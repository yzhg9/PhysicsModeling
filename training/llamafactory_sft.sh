#!/usr/bin/env bash
# SFT launcher.
set -euo pipefail
: "${LLAMAFACTORY_ROOT:?Set LLAMAFACTORY_ROOT to your separately installed framework checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to your base model}"
: "${SFT_DATASET_DIR:?Set SFT_DATASET_DIR to the dataset registration directory}"
: "${SFT_DATASET:?Set SFT_DATASET to the registered dataset name}"
: "${SFT_TEMPLATE:?Set SFT_TEMPLATE to the template supported by your selected model}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$LLAMAFACTORY_ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "$LLAMAFACTORY_ROOT"
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "$MODEL_PATH" \
    --template "$SFT_TEMPLATE" \
    --dataset_dir "$SFT_DATASET_DIR" \
    --dataset "$SFT_DATASET" \
    --finetuning_type lora \
    --report_to none \
    --output_dir "${OUTPUT_DIR:-$REPO_ROOT/outputs/sft}" "$@"
