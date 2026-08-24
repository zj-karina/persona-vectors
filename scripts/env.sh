# Source from project root: `source scripts/env.sh`
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export PROJECT_ROOT
export EXT_ROOT="${EXT_ROOT:-/mnt/opt/alexw/zjkarina}"
export VENV_DIR="${EXT_ROOT}/venv311"

if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "${VENV_DIR}/bin/activate"
fi

export HF_HOME="${EXT_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${EXT_ROOT}/torch_cache"

export RUN_LOG_DIR="${EXT_ROOT}/logs/icml2026"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" \
         "$TORCH_HOME" "$RUN_LOG_DIR"

# V100 → no FA2, no native bf16
export TORCH_DTYPE="float16"
export ATTN_IMPL="sdpa"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -f "$HOME/.hf_token" ]; then
    # shellcheck disable=SC1090
    . "$HOME/.hf_token"
    export HF_TOKEN
fi

echo "[icml2026 env] PROJECT_ROOT=$PROJECT_ROOT  HF_HOME=$HF_HOME  RUN_LOG_DIR=$RUN_LOG_DIR"
