import os

# Both are only honoured before CUDA / the tokenizers threadpool come up, and
# importing anything below pulls in torch. scripts/env.sh exports them too; this
# keeps scripts correct when run without sourcing it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .persona_vectors import PersonaVectors, PersonaSteering, PersonaMonitor, get_decoder_layers
from .dataset import LaMPDataset, task_info, TASKS
from .metrics import (
    compute_metric, compute_accuracy, compute_regression, compute_rouge,
    primary_value, primary_label, higher_is_better,
)
from .inference import (
    load_model_and_tokenizer, persona_steered_generate, build_chat_prompt,
    chat_kwargs_for, system_prompt_for, is_qwen3,
)
from .fact_extractor import (
    FactExtractor, format_profile_from_lamp,
    FACT_EXTRACTION_PROMPT, DOMAIN_NEGATIVE_PROMPTS, TASK_FRAMING,
)
from .runs import ROOT, RESULTS, FIGURES, best_layer
