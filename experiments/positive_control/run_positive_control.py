"""Does the template cause the collapse? Template vs fact-based artifacts.

The template positive prompt is the same sentence for every user with one
profile item pasted in, so a low-rank vector cloud could be measuring the
template rather than the residual stream. The control swaps that prompt for five
concrete facts the model writes about each user and compares both paths on
geometry (cosine, PCA rank, 2-PC variance) and on downstream accuracy.

Fact extraction, vector extraction and inference all run on the same Qwen3-8B.

    python experiments/positive_control/run_positive_control.py \
        --task LaMP-2 --layer_idx 13 --n_users 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import (
    FactExtractor, LaMPDataset, PersonaVectors,
    chat_kwargs_for, compute_metric, get_decoder_layers, higher_is_better,
    load_model_and_tokenizer, persona_steered_generate, primary_value,
    system_prompt_for, task_info,
)


@torch.no_grad()
def extract_vectors(
    model, tokenizer, samples, *,
    layer_idx: int,
    positive_key: str,
    negative_key: str,
    chat_kwargs: dict,
) -> np.ndarray:
    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs,
    )
    out = []
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            v = pv.extract(
                positive_prompts=s[positive_key],
                negative_prompts=s[negative_key],
                extraction_questions=[s["input_text"]],
            )
            out.append(v.cpu().float().numpy())
        except RuntimeError as e:
            # Zero-fill rather than skip: the two paths must stay index-aligned
            # so the same user can be compared across them.
            print(f"  user {i}: extract failed ({e}) — zero-filling")
            out.append(np.zeros(model.config.hidden_size, dtype=np.float32))
        if (i + 1) % 5 == 0:
            print(f"  vectors {i+1}/{len(samples)} ({time.time()-t0:.0f}s)")
    return np.stack(out, axis=0)


def compute_geometry(vectors: np.ndarray, label: str) -> dict:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-8)
    cos = normalized @ normalized.T
    off_diag = cos[~np.eye(len(cos), dtype=bool)]

    pca = PCA().fit(vectors)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    var_2pc = float(sum(pca.explained_variance_ratio_[:2]))

    if rank_90 <= 3 and off_diag.mean() > 0.6:
        verdict = "TEMPLATE COLLAPSE — rank-2 cluster"
    elif off_diag.mean() < 0.4:
        verdict = "USER-SPECIFIC ✓"
    else:
        verdict = "PARTIAL — some user signal"

    print(f"\n=== {label} (n={len(vectors)}) ===")
    print(f"  cosine off-diag: mean={off_diag.mean():.3f} std={off_diag.std():.3f} "
          f"median={np.median(off_diag):.3f}  frac>0.8={(off_diag > 0.8).mean():.2f}")
    print(f"  PCA: rank@90%={rank_90}, rank@95%={rank_95}, var_2pc={var_2pc:.1%}")
    print(f"  verdict: {verdict}")

    return {
        "label": label,
        "n": len(vectors),
        "cosine": {
            "mean": float(off_diag.mean()),
            "std":  float(off_diag.std()),
            "median": float(np.median(off_diag)),
            "frac_above_0.5": float((off_diag > 0.5).mean()),
            "frac_above_0.8": float((off_diag > 0.8).mean()),
        },
        "pca": {
            "rank_90pct": rank_90,
            "rank_95pct": rank_95,
            "var_2pc": var_2pc,
            "top10": pca.explained_variance_ratio_[:10].tolist(),
        },
        "magnitude": {
            "mean_norm": float(np.linalg.norm(vectors, axis=1).mean()),
            "std_norm":  float(np.linalg.norm(vectors, axis=1).std()),
        },
        "verdict": verdict,
    }


@torch.no_grad()
def run_steering_eval(
    model, tokenizer, samples, vectors: np.ndarray, *,
    layer_idx: int, alpha: float,
    chat_kwargs: dict, system_prompt: str,
    max_new_tokens: int,
    label: str,
) -> tuple[list[str], list[str]]:
    preds, refs = [], []
    for i, (s, v) in enumerate(zip(samples, vectors)):
        pred = persona_steered_generate(
            model, tokenizer,
            user_input=s["input_text"],
            persona_vector=torch.from_numpy(v) if alpha != 0 else None,
            layer_idx=layer_idx, alpha=alpha,
            max_new_tokens=max_new_tokens,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        preds.append(pred)
        refs.append(s["output_text"].strip())
        if (i + 1) % 5 == 0:
            print(f"  {label} {i+1}/{len(samples)}")
    return preds, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--n_users", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/positive_control")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    info = task_info(args.task)
    metric = info["metric"]
    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    print(f"=== Positive control: {args.model} on {args.task} ===")
    print(f"  layer={args.layer_idx} α={args.alpha} n_users={args.n_users}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    n_layers = len(get_decoder_layers(model))
    if not (0 <= args.layer_idx < n_layers):
        raise ValueError(f"layer_idx {args.layer_idx} out of range [0, {n_layers})")

    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)
    print(f"  loaded {len(samples)} unique-user samples")

    print("\n[1] Extracting facts via local Qwen...")
    fact_cache = ROOT / args.output_dir / f"cache_facts_{args.task}.json"
    extractor = FactExtractor(model=model, tokenizer=tokenizer, task=args.task)
    samples_enriched = extractor.build_artifacts_for_dataset(
        samples=samples, n_users=args.n_users, cache_path=str(fact_cache),
    )

    print(f"\n[2] Template-based vector extraction (layer {args.layer_idx})...")
    template_vectors = extract_vectors(
        model, tokenizer, samples_enriched,
        layer_idx=args.layer_idx,
        positive_key="positive_system_prompts",
        negative_key="negative_system_prompts",
        chat_kwargs=chat_kwargs,
    )

    print(f"\n[3] Fact-based vector extraction (layer {args.layer_idx})...")
    fact_vectors = extract_vectors(
        model, tokenizer, samples_enriched,
        layer_idx=args.layer_idx,
        positive_key="fact_positive_prompts",
        negative_key="fact_negative_prompts",
        chat_kwargs=chat_kwargs,
    )

    print("\n[4] Geometry comparison")
    template_geo = compute_geometry(template_vectors, "Template-based")
    fact_geo = compute_geometry(fact_vectors, "Fact-based")

    print("\n[5] Downstream evaluation")
    runs = {}
    for label, vectors, alpha in (
        ("zero_shot", np.zeros_like(template_vectors), 0.0),
        ("template_steering", template_vectors, args.alpha),
        ("fact_steering", fact_vectors, args.alpha),
    ):
        preds, refs = run_steering_eval(
            model, tokenizer, samples_enriched, vectors,
            layer_idx=args.layer_idx, alpha=alpha,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
            max_new_tokens=info["max_new_tokens"], label=label,
        )
        value = compute_metric(metric, preds, refs)
        runs[label] = {"value": value, "primary": primary_value(metric, value)}
        print(f"  {label}: {value}")

    zs_p = runs["zero_shot"]["primary"]
    sign = 1 if higher_is_better(metric) else -1
    for label in ("template_steering", "fact_steering"):
        runs[label]["delta"] = sign * (runs[label]["primary"] - zs_p)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # plot_comparison.py replots from these without re-running extraction.
    npz_path = out_dir / f"vectors_{args.model.split('/')[-1]}_{args.task}.npz"
    np.savez_compressed(npz_path, template=template_vectors, fact=fact_vectors)

    payload = {
        "model": args.model,
        "task": args.task,
        "metric": metric,
        "layer_idx": args.layer_idx,
        "n_users": args.n_users,
        "alpha": args.alpha,
        "extraction_method": "local_llm_no_api",
        "lamp_profile_format": "pre_formatted_strings (titles_p6)",
        "geometry": {
            "template": template_geo,
            "fact_based": fact_geo,
            "cosine_delta": fact_geo["cosine"]["mean"] - template_geo["cosine"]["mean"],
            "rank_90_delta": fact_geo["pca"]["rank_90pct"] - template_geo["pca"]["rank_90pct"],
            "hypothesis_supported": (
                fact_geo["cosine"]["mean"] < template_geo["cosine"]["mean"] - 0.15
                and fact_geo["pca"]["rank_90pct"] > template_geo["pca"]["rank_90pct"]
            ),
        },
        "downstream": runs,
        "samples_for_inspection": [
            {"id": s.get("id"), "facts": s["extracted_facts"]}
            for s in samples_enriched[:3]
        ],
        "vectors_npz": str(npz_path.relative_to(ROOT)),
        "timestamp": datetime.now().isoformat(),
    }
    out_path = out_dir / f"comparison_{args.model.split('/')[-1]}_{args.task}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print(f"  template cosine:       {template_geo['cosine']['mean']:.3f}")
    print(f"  fact-based cosine:     {fact_geo['cosine']['mean']:.3f}")
    print(f"  cosine Δ:              {payload['geometry']['cosine_delta']:+.3f}")
    print(f"  template rank@90%:     {template_geo['pca']['rank_90pct']}")
    print(f"  fact-based rank@90%:   {fact_geo['pca']['rank_90pct']}")
    print(f"  hypothesis supported:  {payload['geometry']['hypothesis_supported']}")
    print(f"  ZS / tmpl / fact:      {zs_p:.3f} / "
          f"{runs['template_steering']['primary']:.3f} / "
          f"{runs['fact_steering']['primary']:.3f}")
    print(f"  Δ vs ZS:  template={runs['template_steering']['delta']:+.3f}  "
          f"fact={runs['fact_steering']['delta']:+.3f}")
    print(f"\nSaved {out_path}")
    print(f"Saved {npz_path}")


if __name__ == "__main__":
    main()
