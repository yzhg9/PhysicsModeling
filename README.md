# Decoupled Physical Modeling and Execution for Physics Reasoning

Code for paper "Decoupled Physical Modeling and Execution for Physics Reasoning"
<!-- We use **explicit physical modeling as an intermediate representation** for
multimodal physics reasoning. The model first identifies the physical system,
relevant principles, and constraints, then uses this representation to guide
mathematical execution and derive the final answer. Physical modeling also connects
**supervision and reward design**: SFT teaches the model to construct physical
representations before execution, while rubric-guided GRPO uses rubrics derived from
verified modeling traces to assess the physical grounding of the reasoning,
alongside answer correctness and format validity. -->

## Installation

Run the commands below from the repository root. Create a Python 3.11 environment
for inference and evaluation:

```bash
conda create -n physreasoning python=3.11 -y
conda activate physreasoning
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


Export the runtime settings before starting Python:

```bash
export PHYSREASONING_PROMPTS_FILE=/path/to/runtime.json
```

Configure model endpoints and credentials through command-line arguments or exported
environment variables. Generation and model-based judging support OpenAI-compatible
inference endpoints.

## Data preparation

Download the source datasets from their versioned Hugging Face repositories:

```bash
python -m scripts.download_data --datasets all --output-dir data/raw
python -m scripts.filter_data --stage source \
  --input-dir data/raw --output-dir data/prepared \
  --text-limit 12000 --seed 42
```

Supported sources are PhysReason, PhyX, SeePhys, SCP-116K, MegaScience,
PhysUniBench, and MMK12. Use `--list` to list their repository IDs, or select sources
with `--datasets physreason phyx seephys`. Each download records the resolved revision
and file checksums; `--revision dataset=revision` selects another revision.

Source preparation selects English physics questions, separates PhysReason
sub-questions, validates and exports images, removes exact duplicates, and samples
textual problems. PhysUniBench English multiple-choice questions and MMK12 physics
test questions are used for evaluation.

Outputs are `text_pool.jsonl`, `multimodal_pool.jsonl`, `evaluation.jsonl`, and an
`images/` directory. `summary.json` and `decisions.jsonl` record filtering counts and
rejection reasons. Choose a new output directory for each preparation run.

After generating modeling candidates with `scripts.generate_modeling`, validate
them with a quality model, then execute and judge the candidates that pass:

```bash
python -m scripts.generate_modeling \
  --input data/prepared/multimodal_pool.jsonl --output outputs/modeling.jsonl \
  --model model-name --modeling_provider vllm \
  --api_base http://localhost:8000/v1 --api_key EMPTY \
  --mm_root data/prepared --num_attempts 5

python -m scripts.filter_data --stage modeling \
  --input outputs/modeling.jsonl --output-dir data/verified \
  --model model-name --solver_model model-name --judge_model model-name \
  --provider vllm --api_base http://localhost:8000/v1 --api_key EMPTY \
  --mm_root data/prepared
```

The quality check evaluates internal coherence, multimodal consistency, modeling
specificity, and physical consistency. Its JSON response contains the boolean fields
`internal_coherence`, `multimodal_consistency`, `modeling_specificity`, and
`physical_consistency`. The runtime configuration key is
`scripts.filter_data.MODELING_QUALITY_PROMPT`, with `{question}` and `{modeling}`
substitutions. Execution validation uses the existing solver and `judge_one` and
requires full correctness. By default, one validated attempt per question is kept;
`--keep-all-attempts` retains every passing attempt.

The command writes `accepted.jsonl`, `sft.jsonl`, and filtering reports. Register
`sft.jsonl` in LLaMA-Factory using the Alpaca format, with its image column when present.

## Evaluation

The default evaluation command runs **physical modeling, mathematical execution,
and correctness judging**. Both generation stages use `--model` unless overridden
by `--modeling_model` or `--solver_model`.

```bash
python evaluate.py \
  --input data/prepared/evaluation.jsonl --mm_root data/prepared \
  --model model-name --provider vllm \
  --api_base http://localhost:8000/v1 --api_key EMPTY \
  --output_dir outputs/evaluation \
  --judge_model model-name --judge_provider vllm \
  --judge_api_base http://localhost:8002/v1 \
  --rtol 0.03
```


Use `--num_attempts` to sample multiple trajectories per question. Stage-specific
endpoints can be set with `--modeling_api_base` and `--solver_api_base`.

### Score existing predictions

```bash
python evaluate.py --mode score \
  --input outputs/evaluation/solutions.jsonl --output outputs/results.json \
  --judge_model model-name --judge_provider vllm \
  --judge_api_base http://localhost:8002/v1 --judge_api_key EMPTY
```

Predictions are read from `pred_answer`, `PRED`, or `pred`, or extracted from a boxed
answer in `response` or `solution`.

## Run the reasoning stages separately

```bash
# Physical modeling
python -m scripts.generate_modeling \
  --input /path/to/questions.jsonl --output outputs/modeling.jsonl \
  --model model-name --modeling_provider vllm \
  --api_base http://localhost:8000/v1 --api_key EMPTY

# Mathematical execution
python -m scripts.solve_with_modeling \
  --input outputs/modeling.jsonl --output outputs/solutions.jsonl \
  --model model-name --solver_provider vllm \
  --api_base http://localhost:8001/v1 --api_key EMPTY
```

The modeling stage adds `modeling`; execution adds `solution`, `response`, and
`pred_answer`. Output uses `<modeling>`, `<solution>`, and `<answer>` blocks, with a
boxed final answer.

## Training

The paper uses **LLaMA-Factory for SFT** and **verl for GRPO**. LLaMA-Factory is
installed separately; the modified verl source is included in `verl/`.

### 1. SFT cold start

Install LLaMA-Factory in a dedicated environment:

```bash
conda create -n physreasoning-sft python=3.11 -y
conda activate physreasoning-sft
python -m pip install --upgrade pip
export LLAMAFACTORY_ROOT=/path/to/LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git "$LLAMAFACTORY_ROOT"
python -m pip install -e "$LLAMAFACTORY_ROOT"
llamafactory-cli help
```

Register the training set in `dataset_info.json` under `SFT_DATASET_DIR`, set
`SFT_DATASET` to its registered name, and choose `SFT_TEMPLATE` for the model family.
See the official [installation guide](https://github.com/hiyouga/LlamaFactory#installation)
and [dataset format guide](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md).

Train first on textual physics problems and then on multimodal physics problems,
carrying the learned weights from the first stage into the second. The paper uses
2 epochs per stage, learning rate `1e-4`, and LoRA rank `64` with alpha `128`.

Use the corresponding model initialization and registered training set for each stage:

```bash
export LLAMAFACTORY_ROOT=/path/to/LLaMA-Factory
export MODEL_PATH=/path/to/stage-initialization
export SFT_DATASET_DIR=/path/to/datasets
export SFT_DATASET=your_registered_dataset
export SFT_TEMPLATE=model-template
export OUTPUT_DIR=/path/to/stage-output

bash training/llamafactory_sft.sh \
  --learning_rate 1e-4 --num_train_epochs 2 \
  --lora_rank 64 --lora_alpha 128
```

### 2. Rubric-guided GRPO

Create a separate environment and install the bundled verl with the vLLM backend
and multimodal utilities. Prepare CUDA and FlashAttention as described in the
[bundled installation guide](verl/docs/start/install.rst):

```bash
conda create -n physreasoning-grpo python=3.12 -y
conda activate physreasoning-grpo
python -m pip install --upgrade pip
python -m pip install -e "./verl[vllm,geo]"
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The CUDA availability check should print `True` before GPU training.

Initialize from the multimodal SFT model. For a LoRA adapter, set `MODEL_PATH` to its
base model and `SFT_ADAPTER_PATH` to the SFT adapter. The paper uses learning rate
`5e-5`, 1 epoch, and LoRA rank `64` with alpha `128`:

```bash
export MODEL_PATH=/path/to/base-model
export SFT_ADAPTER_PATH=/path/to/multimodal-sft-adapter
export TRAIN_FILES="['/path/to/train.parquet']"
export VAL_FILES="['/path/to/validation.parquet']"

bash training/verl_grpo.sh \
  actor_rollout_ref.actor.optim.lr=5e-5 \
  trainer.total_epochs=1
```

The GRPO launcher uses the bundled `verl/` source automatically. See
[the verl changes](verl/PHYSREASONING_CHANGES.md) for the project-specific modules.
Additional framework options can be passed as arguments to the launchers.

## Repository structure

```text
phys_utils/          Model client, parsing, hybrid correctness evaluation, and utilities
scripts/             Source download, filtering, modeling generation, and mathematical execution
training/            LLaMA-Factory SFT and verl GRPO launchers
verl/               Full modified verl source, configuration, tests, and upstream notices
evaluate.py          Modeling -> execution -> judging, or scoring existing predictions
```

