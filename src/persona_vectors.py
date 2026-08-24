"""Persona-vector extraction and steering, adapted to per-user identities.

Follows the algorithm of Rimsky et al. (Anthropic, arXiv:2507.21509) but
substitutes a LaMP user for a character trait: the positive system prompts
describe one user's history, the negatives describe a generic assistant, and
the vector is the difference of mean response activations.

No judge and no filtering step — artifacts in, one vector out.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch
import torch.nn as nn


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Decoder blocks of a Llama / Qwen / Mistral / Gemma2-style causal LM."""
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers
    raise AttributeError(f"Cannot locate decoder layers on {type(model).__name__}")


def _layer_hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace_layer_hidden(out, new_hidden):
    if isinstance(out, tuple):
        return (new_hidden, *out[1:])
    return new_hidden


def _check_layer(model: nn.Module, layer_idx: int) -> None:
    n_layers = len(get_decoder_layers(model))
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {n_layers})")


class PersonaVectors:
    """Extracts one per-user vector from a single residual-stream layer.

    For every (system prompt, question) pair the model generates a response and
    we mean-pool the layer's activations over the response tokens; the vector is
    mean(positive) - mean(negative).

    `chat_template_kwargs` is where Qwen3 needs {"enable_thinking": False}.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        layer_idx: int,
        system_prompt: str | None = None,
        max_new_tokens: int = 50,
        device: torch.device | str | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.chat_template_kwargs = chat_template_kwargs or {}

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        _check_layer(model, layer_idx)

    def _format_chat(self, system: str, user: str) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

    @torch.no_grad()
    def _generate_and_pool(self, system: str, question: str) -> torch.Tensor | None:
        """Mean activation over the response tokens, or None if nothing was generated."""
        prompt = self._format_chat(system, question)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        full_ids = out[0].unsqueeze(0)
        if full_ids.shape[1] <= prompt_len:
            return None

        # One extra forward beats caching hidden states through generate().
        outputs = self.model(full_ids, output_hidden_states=True, use_cache=False)
        hs = outputs.hidden_states[self.layer_idx + 1][0]
        response_hs = hs[prompt_len:]
        if response_hs.numel() == 0:
            return None
        return response_hs.mean(dim=0).detach().float().cpu()

    def extract(
        self,
        positive_prompts: Sequence[str],
        negative_prompts: Sequence[str],
        extraction_questions: Sequence[str],
    ) -> torch.Tensor:
        """One [hidden_dim] vector at self.layer_idx."""
        pos_acts, neg_acts = [], []
        for prompts, acts in ((positive_prompts, pos_acts), (negative_prompts, neg_acts)):
            for sp in prompts:
                for q in extraction_questions:
                    v = self._generate_and_pool(sp, q)
                    if v is not None:
                        acts.append(v)

        if not pos_acts or not neg_acts:
            raise RuntimeError(
                f"Empty bucket: pos={len(pos_acts)} neg={len(neg_acts)} — "
                "model may have produced no response tokens"
            )
        return torch.stack(pos_acts).mean(dim=0) - torch.stack(neg_acts).mean(dim=0)


class PersonaSteering:
    """Adds `alpha * vector` to one layer's residual stream while decoding.

        with PersonaSteering(model, layer_idx=16).hook(vector, alpha=1.0):
            model.generate(...)
    """

    def __init__(self, model: nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self._layers = get_decoder_layers(model)
        _check_layer(model, layer_idx)

    @contextmanager
    def hook(self, vector: torch.Tensor, alpha: float = 1.0, position: str = "all"):
        """`position="all"` steers every token, `"last"` only the final one."""
        if position not in ("all", "last"):
            raise ValueError("position must be 'all' or 'last'")

        v_cpu = vector.detach().float().cpu()
        cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

        def vec_for(device, dtype):
            key = (device, dtype)
            if key not in cache:
                cache[key] = v_cpu.to(device=device, dtype=dtype)
            return cache[key]

        def fwd_hook(module, inputs, output):
            hidden = _layer_hidden(output)
            v = vec_for(hidden.device, hidden.dtype)
            if position == "all":
                hidden = hidden + alpha * v
            else:
                hidden = hidden.clone()
                hidden[..., -1, :] = hidden[..., -1, :] + alpha * v
            return _replace_layer_hidden(output, hidden)

        handle = self._layers[self.layer_idx].register_forward_hook(fwd_hook)
        try:
            yield self
        finally:
            handle.remove()


class PersonaMonitor:
    """Scores how much a prompt already points along a persona direction.

    Projects the last prompt token's hidden state at `layer_idx` onto the
    vector — no generation, so it is cheap enough to run over a whole split.
    """

    def __init__(self, model: nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx

    @torch.no_grad()
    def score(self, vector: torch.Tensor, input_ids: torch.Tensor,
              attention_mask: torch.Tensor | None = None,
              normalize: bool = True) -> torch.Tensor:
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[self.layer_idx + 1]

        if attention_mask is None:
            last_idx = torch.full((input_ids.size(0),), input_ids.size(1) - 1,
                                  dtype=torch.long, device=device)
        else:
            last_idx = attention_mask.long().sum(dim=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=device)
        last_h = hs[batch_idx, last_idx].float()

        v = vector.to(device=device, dtype=last_h.dtype)
        if normalize:
            v = v / (v.norm() + 1e-8)
            last_h = last_h / (last_h.norm(dim=-1, keepdim=True) + 1e-8)
        return (last_h * v).sum(dim=-1).cpu()
