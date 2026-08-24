"""Minimal LaMP wrapper for persona-vector experiments.

Each item carries the raw LaMP fields plus the contrastive system prompts the
extraction needs: positive prompts built from the user's own profile, negative
prompts drawn from a fixed generic set.

The `LaMPDataset` in llm-behavior-fusion is much larger because it feeds a
trained Q-Former; this keeps only what extraction and steering use.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterator

GENERIC_NEGATIVE_PROMPTS: list[str] = [
    "You are a neutral, generic assistant with no particular preferences.",
    "Answer in a generic style, without imitating any specific author.",
    "You are an impartial assistant. Do not reflect any personal voice.",
    "Respond in a default, unstyled manner.",
    "You are a baseline assistant with no user context.",
]

POSITIVE_TEMPLATE = (
    "You are an author whose past work is exemplified by the following item. "
    "Reproduce that author's preferences, style, and topical focus.\n"
    "Item: {profile_excerpt}"
)

TASKS: dict[str, dict] = {
    "LaMP-1": {"folder": "LaMP_1", "metric": "accuracy",   "max_new_tokens": 3},
    "LaMP-2": {"folder": "LaMP_2", "metric": "accuracy",   "max_new_tokens": 3},
    "LaMP-3": {"folder": "LaMP_3", "metric": "regression", "max_new_tokens": 3},
    "LaMP-4": {"folder": "LaMP_4", "metric": "rouge",      "max_new_tokens": 32},
    "LaMP-5": {"folder": "LaMP_5", "metric": "rouge",      "max_new_tokens": 32},
    "LaMP-7": {"folder": "LaMP_7", "metric": "rouge",      "max_new_tokens": 32},
}


def task_info(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Use one of {list(TASKS)}.")
    return TASKS[task]


def _profile_key(sample: dict) -> str:
    profile = json.dumps(sample.get("behavior_profile_text", []), sort_keys=True)
    return hashlib.md5(profile.encode()).hexdigest()


class LaMPDataset:
    """Iterable LaMP split with per-item positive/negative system prompts.

    `split="val"` reads dev_titles_p6.json, `"train"` reads train_titles_p6.json.

    Set `unique_users=True` to keep one item per distinct profile. LaMP-2 has
    roughly five test items per user and LaMP-3 several as well, so without it
    an "n=100 users" run is really n=20 users counted five times.
    """

    def __init__(
        self,
        task: str,
        split: str = "val",
        n_samples: int | None = None,
        data_dir: str = "data",
        n_positive: int = 3,
        n_negative: int = 3,
        excerpt_chars: int = 600,
        unique_users: bool = False,
    ):
        info = task_info(task)
        self.task = task
        self.split = split
        self.data_dir = Path(data_dir)
        self.metric = info["metric"]
        self.max_new_tokens = info["max_new_tokens"]
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.excerpt_chars = excerpt_chars

        fname = "train_titles_p6.json" if split == "train" else "dev_titles_p6.json"
        path = self.data_dir / info["folder"] / fname
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path) as f:
            data = json.load(f)

        if unique_users:
            seen: set[str] = set()
            deduped = []
            for sample in data:
                key = _profile_key(sample)
                if key not in seen:
                    seen.add(key)
                    deduped.append(sample)
            data = deduped

        self.data = data[:n_samples] if n_samples is not None else data

    def __len__(self) -> int:
        return len(self.data)

    def _build_positive(self, profile_texts: list[str]) -> list[str]:
        if not profile_texts:
            return []
        return [
            POSITIVE_TEMPLATE.format(
                profile_excerpt=profile_texts[i % len(profile_texts)][: self.excerpt_chars]
            )
            for i in range(self.n_positive)
        ]

    def _build_negative(self) -> list[str]:
        return [GENERIC_NEGATIVE_PROMPTS[i % len(GENERIC_NEGATIVE_PROMPTS)]
                for i in range(self.n_negative)]

    def __getitem__(self, idx: int) -> dict:
        s = self.data[idx]
        profile = s.get("behavior_profile_text") or []
        return {
            "input_text": s["input_text"],
            "output_text": s["output_text"],
            "behavior_profile_text": profile,
            "positive_system_prompts": self._build_positive(profile),
            "negative_system_prompts": self._build_negative(),
        }

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    def sample_train_inputs(self, k: int, seed: int = 42) -> list[str]:
        """k inputs from the *train* split, to use as extraction questions.

        Independent of self.split, so a val-split dataset can still draw
        questions the model was never evaluated on.
        """
        path = self.data_dir / task_info(self.task)["folder"] / "train_titles_p6.json"
        if not path.exists():
            return []
        with open(path) as f:
            train = json.load(f)
        if k >= len(train):
            return [d["input_text"] for d in train]
        return [d["input_text"] for d in random.Random(seed).sample(train, k)]
