# Per-User Persona Vectors for LLM Personalization

Rimsky et al. ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)) show that a
character trait like "evil" can be read out of a model's residual stream as a
single direction, by contrasting activations under prompts that do and do not
ask for the trait. This repository asks whether the same construction works when
the trait is a *person*: take one LaMP user, build the positive prompts from
their own history, and see whether the resulting direction steers the model
toward the answer that user would give.

The short answer is no, not at the level of a population. Steering at a single
layer with a single α is not distinguishable from zero-shot on either task where
it should have worked, and on one of them the richer artifacts make it
measurably worse. It does work for individual users, at an α that differs from
user to user — which is the more interesting result, and the one the code here
is set up to measure.

## Where it landed

At full sample size (LaMP-2 n=323 unique users, LaMP-7 n=1497), against a
zero-shot control on the same items:

| Method | LaMP-2 acc | p | LaMP-7 ROUGE-L | p |
|---|---|---|---|---|
| Zero-shot | 0.582 | — | 0.4262 | — |
| Steering, template artifacts (α=1) | 0.557 | 0.17 | 0.4279 | 0.11 |
| Steering, fact artifacts (α=1) | 0.536 | 0.08 | 0.4205 | **0.017** |
| In-context, K=3 profile items | 0.579 | 1.00 | 0.4279 | 0.49 |

LaMP-2 uses exact McNemar on the discordant pairs, LaMP-7 a paired bootstrap
over per-user ROUGE-L (B=10000, seed 42); `experiments/significance_tests.py`
regenerates the table. The only significant effect in it is fact-based steering
hurting LaMP-7. Giving the model the same profile as text instead of as a vector
does not help either, which is worth knowing before blaming the steering
operator for everything.

Getting to that table turned up three things that were not obvious going in.

**The vectors are user-specific; the early geometry said otherwise because it
counted samples, not users.** LaMP-2's dev split contains 1605 items drawn from
323 profiles, so an "n=100 users" run is really about twenty users repeated five
times each. Measured that way the vectors look collapsed — mean off-diagonal
cosine 0.749, 92.7% of variance in two principal components, 59% of pairs above
0.8. Deduplicating by profile hash first (`unique_users=True`) drops the mean
cosine to 0.329 with rank-9 at 90% variance. Whatever is failing, it is not that
the extraction produces the same direction for everyone.

**Replacing the template with concrete facts changes the geometry but not the
output.** The positive control has the model itself write five distinguishing
facts per user and uses those as the positive prompt, which on LaMP-7 pulls the
mean inter-user cosine from 0.413 down to 0.252. Downstream ROUGE-L moves the
other way, 0.4301 for template artifacts against 0.4249 for facts, with
zero-shot at 0.4295 between them. Better-separated vectors, same generations.

**Individual users are steerable, at their own α.** Sweeping α per user on
LaMP-2 with fact artifacts: 188 of 323 users are already right at α=0, and 29 of
the remaining 135 can be flipped to the gold answer by some α (11 with template
artifacts). The α that first does it ranges from 0.25 to 2.0, mean 0.66,
sd 0.44 — so no single global α reaches more than a fraction of them, and the
population average buries the ones it does reach.

## Setup

```bash
pip install -r requirements.txt
source scripts/env.sh          # HF cache locations, venv, token
ln -sfn /path/to/lamp-data data
```

`data/` is expected to contain `LaMP_{1,2,3,4,5,7}/{train,dev}_titles_p6.json`.
The runs in `results/` were produced on V100s, hence float16 and SDPA attention
rather than bf16 and FlashAttention-2.

## Running things

Layer search comes first; everything downstream reads its output through
`src.best_layer()` and falls back to a hand-picked layer if it has not been run.

```bash
python experiments/layer_search/run_layer_search.py --model Qwen/Qwen3-8B --task LaMP-2
python experiments/geometry_analysis/analyze_geometry.py --model Qwen/Qwen3-8B --task LaMP-2 --layer_idx 13
```

The positive control extracts both artifact variants and saves the vectors to
`results/positive_control/vectors_<model>_<task>.npz`. Every later experiment
loads that file rather than re-extracting, so run it before the operator and
per-user sweeps.

```bash
bash scripts/run_positive_control.sh                     # LaMP-2 and LaMP-7
python scripts/extract_template_vectors.py --task LaMP-7 --n_users 1497
python scripts/extract_fact_vectors.py     --task LaMP-7 --n_users 1497
```

Both extraction scripts are resumable: they read the existing `.npz`, extract
only the users past its current length, and write the merged array back.

```bash
python experiments/icl_baseline/run_icl.py --task LaMP-2 --K_grid 3 5 6
python experiments/operators/run_operators.py --task LaMP-7 --operator proj_nuisance --pca_k 5
python experiments/operators/run_routing.py --task LaMP-7
python experiments/operators/run_rank1_edit.py --task LaMP-7
python experiments/variance_analysis/analyze_variance.py --task LaMP-7 --n_users 1497
python experiments/case_study/per_user_alpha_search.py --task LaMP-2 --variant fact
python experiments/significance_tests.py
python scripts/generate_paper_figures.py
```

## Results in detail

### Extraction layer

Sweeping the middle 60% of the stack at stride 2 on LaMP-2 (n=50, α=1):

| Model | Best layer | Depth | Acc at best | Zero-shot |
|---|---|---|---|---|
| Qwen3-8B | 13 | 36% | 0.740 | 0.740 |
| Qwen3-14B | 10 | 25% | 0.760 | 0.760 |

No layer beat zero-shot in either model. What the sweep did settle is that the
hand-picked defaults carried over from the smoke runs — layer 18 for the 8B,
layer 20 for the 14B, both around half depth — sit in a mild degradation zone,
while layers at 25–40% depth at least leave the zero-shot accuracy intact. Layer
13 is used for everything afterwards.

<details>
<summary>Per-layer accuracy</summary>

Qwen3-8B (zero-shot 0.740 / f1w 0.765):

| layer | 7 | 9 | 11 | **13** | 15 | 17 | 19 | 21 | 23 | 25 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| depth | 19% | 25% | 31% | **36%** | 42% | 47% | 53% | 58% | 64% | 69% | 75% |
| acc | 0.700 | 0.720 | 0.700 | **0.740** | 0.740 | 0.700 | 0.660 | 0.680 | 0.660 | 0.680 | 0.680 |
| f1w | 0.728 | 0.749 | 0.730 | **0.762** | 0.767 | 0.749 | 0.719 | 0.711 | 0.709 | 0.719 | 0.719 |

Qwen3-14B (zero-shot 0.760 / f1w 0.797):

| layer | 8 | **10** | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| depth | 20% | **25%** | 30% | 35% | 40% | 45% | 50% | 55% | 60% | 65% | 70% | 75% |
| acc | 0.740 | **0.760** | 0.760 | 0.760 | 0.760 | 0.740 | 0.740 | 0.760 | 0.720 | 0.760 | 0.760 | 0.740 |
| f1w | 0.758 | **0.783** | 0.783 | 0.783 | 0.775 | 0.740 | 0.762 | 0.780 | 0.734 | 0.783 | 0.783 | 0.755 |

</details>

### Steering strength

Qwen3-8B, LaMP-2, layer 13, n=200:

| α | 0.0 | 0.5 | 1.0 | 2.0 | 4.0 | 8.0 | 16.0 |
|---|---|---|---|---|---|---|---|
| acc | 0.760 | 0.770 | **0.780** | 0.745 | 0.545 | 0.160 | 0.000 |
| f1w | 0.781 | 0.790 | **0.798** | 0.774 | 0.652 | 0.194 | 0.000 |

The response curve is the shape the persona-vector paper predicts: a shallow
optimum around α=1 and a collapse of base capability past α=4. The +0.02 at the
peak did not survive a larger sample — at n=500 both backbones came out at
−0.012 against zero-shot — so it is sample variance rather than a real effect,
but the collapse at high α confirms the hook is doing what it should.

Varying the number of extraction questions did nothing (n=50, layer 13, α=1):
k=1 gives 0.740, k=3, 5 and 10 all give 0.700, and k=3 and k=5 produce
bit-identical predictions on every example. More questions do not average the
noise out of the vector because extraction noise is not what limits it.

### Vector geometry

At layer 13, n=30 deduplicated users per task — the sample the committed
`comparison_*.json` files were written from, before the extraction scripts
extended the saved vectors to 323 (LaMP-2) and 1497 (LaMP-7):

| Task | Artifacts | mean cos | rank@90% | var in 2 PCs |
|---|---|---|---|---|
| LaMP-2 | template | 0.329 | 9 | 37.4% |
| LaMP-2 | fact | 0.335 | 7 | 59.6% |
| LaMP-7 | template | 0.413 | 18 | 28.8% |
| LaMP-7 | fact | 0.252 | 19 | 26.7% |

The two artifact constructions are geometrically equivalent on LaMP-2 and
clearly different on LaMP-7, where facts separate users much better. Neither
difference reaches the generations.

The earlier geometry runs in `results/geometry/` predate the deduplication fix
and report mean cosines of 0.749 (Qwen3-8B, layer 13) and 0.921 (Qwen3-14B,
layer 10) on n=100 *samples*. Those numbers measure how often the same profile
recurs in the dev split, not how similar users are, and
`analyze_geometry.py` still runs without `unique_users` — as do the layer
search, α-sweep, n_questions and full-run scripts. Read anything from those five
as a sample-level rather than user-level measurement.

### Alternative operators

Three ways of changing what gets injected, all at layer 13 on the full
unique-user samples, all reported as the best α in the sweep against zero-shot:

| Operator | LaMP-2 (acc) | LaMP-7 (ROUGE-L) |
|---|---|---|
| Project out top-k PCA directions | +0.003 (template, k=5) | +0.001 (template, k=5) |
| Rank-1 `down_proj` edit | 0.000 | +0.002 (template) |
| Cosine-gated token routing | +0.006 (fact) | +0.005 (fact, n=200) |

Removing the population-shared directions, editing the weights instead of the
activations, and steering only the token positions that already point along the
persona direction all land within a few thousandths of the control. The
per-user analysis in `experiments/variance_analysis/` says why the aggregate
sits where it does: on LaMP-7 with template artifacts, 177 users improve and 167
get worse, and none of the three static predictors of steering quality
(alignment with the population mean, relative magnitude, neighbourhood coherence)
correlates with the outcome above |r| = 0.03.

### One user where it works

LaMP-2 user 3, layer 13, fact artifacts, ‖v‖ = 27.6. The profile is six
`MOVIE: "<headline>"` items dominated by women's voices and humour from female
social-media accounts, body positivity and gender issues. The test item is a
politically framed article; the gold label is `politics`.

| α | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 | 2.00 |
|---|---|---|---|---|---|---|---|---|
| top token | `education` 1.00 | `education` 1.00 | `education` 0.699 / `pol` 0.301 | `pol` 1.00 | `pol` 1.00 | `pol` 1.00 | `pol` 1.00 | `pol` 1.00 |
| prediction | education | education | education | politics | politics | politics | politics | politics |

The transition happens between α=0.50 and α=0.75 and it is sharp — the user's
direction rotates the last-token logits far enough to override the model's
content prior in one step of the grid. Figure:
`figures/fig_case_LaMP-2_user003_fact.pdf`.

### Smoke reference

The n=200 runs in `results/main_table/` compare a frozen LLM against
BehavioralTwin, the trained Flan-T5-XXL plus Q-Former pipeline this work started
from:

| Task | BehavioralTwin | Best zero-shot LLM | Best persona LLM |
|---|---|---|---|
| LaMP-1 (acc) | **0.567** | 0.435 (Qwen3-14B) | 0.465 (Mistral) |
| LaMP-2 (acc) | 0.703 | 0.790 (Qwen3-14B) | **0.805** (Qwen3-14B) |
| LaMP-3 (MAE) | **0.251** | 0.630 (Qwen3-14B) | — |
| LaMP-4 (R-L) | **0.179** | 0.137 (Qwen3-14B) | — |
| LaMP-5 (R-L) | **0.437** | 0.360 (Qwen3-14B) | — |
| LaMP-7 (R-L) | 0.403 | 0.413 (Qwen3-14B) | **0.426** (Qwen3-8B) |

Swapping the backbone for a frozen Qwen3-14B and prompting it zero-shot beats
the trained pipeline on LaMP-2 and LaMP-7 without training anything. It loses
badly on the regression and long-generation tasks.

## Repository layout

```
src/
  persona_vectors.py   PersonaVectors, PersonaSteering, PersonaMonitor
  fact_extractor.py    fact-based artifact construction
  dataset.py           LaMPDataset, task registry, prompt templates
  inference.py         model loading, persona-steered generation
  metrics.py           accuracy / regression / ROUGE
  runs.py              locating earlier runs (best_layer)
experiments/
  layer_search/        sweep the middle 60% of the stack
  geometry_analysis/   cosine, magnitude ratio, PCA
  alpha_sweep/         steering-strength response curve
  n_questions/         extraction questions ablation
  full_run/            full-split evaluation, ZS + steering
  positive_control/    template vs fact artifacts, saves the shared .npz
  icl_baseline/        profile as context instead of as a vector
  operators/           PCA projection, rank-1 weight edit, token routing
  variance_analysis/   which users steering helps, and whether that is predictable
  case_study/          single-user trace and per-user α search
  significance_tests.py
results/               one JSON per run, predictions included
figures/               PDFs built by scripts/generate_paper_figures.py
scripts/               environment, vector extraction, figure and paper builds
```

## Reproducibility notes

Seed 42 throughout. Qwen3 runs pass `enable_thinking=False` to the chat template
and put `/no_think` in the system prompt, so nothing is scored against a
truncated chain of thought. Extraction and steering always use the same layer.
Raw predictions and references are written next to the metrics in every result
file, which is what lets `significance_tests.py` pair examples after the fact.

The deduplication distinction matters when comparing numbers across
experiments: `unique_users=True` in the positive control, ICL, operator,
variance and case-study runs; not set in the layer search, geometry, α-sweep,
n_questions and full runs.

## Citation

```bibtex
@inproceedings{persona_user_vectors_2026,
  title={Per-User Persona Vectors: Are Individual Identities Linear
         Directions in LLM Activation Space?},
  author={Romanova, Karina},
  booktitle={Mechanistic Interpretability Workshop at ICML 2026},
  year={2026}
}
```

## Acknowledgements

The persona-vector algorithm is from Rimsky et al. (Anthropic, 2025). The LaMP
benchmark is from Salemi et al. (2024).
