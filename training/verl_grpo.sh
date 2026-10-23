#!/usr/bin/env bash
# Portable single-node GRPO launch template.
# Run inside a Linux/CUDA environment with the bundled verl installed.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="$REPO_ROOT/verl"
export VERL_ROOT
: "${VERL_ROOT:?Bundled verl source directory is required}"
if [[ ! -f "$VERL_ROOT/verl/trainer/main_ppo.py" ]]; then
    printf '%s\n' "Bundled verl source not found: $VERL_ROOT" >&2
    exit 1
fi
: "${MODEL_PATH:?Set MODEL_PATH to your own base model}"
: "${TRAIN_FILES:?Set TRAIN_FILES to a Hydra list of your own training parquet paths}"
: "${VAL_FILES:?Set VAL_FILES to a Hydra list of your own validation parquet paths}"
export PYTHONPATH="$REPO_ROOT:$VERL_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SFT_ARGS=()
if [[ -n "${SFT_ADAPTER_PATH:-}" ]]; then
    SFT_ARGS+=("actor_rollout_ref.model.lora_adapter_path=$SFT_ADAPTER_PATH")
fi

cd "$VERL_ROOT"
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    "data.train_files=$TRAIN_FILES" \
    "data.val_files=$VAL_FILES" \
    data.train_batch_size=80 \
    data.val_batch_size=64 \
    data.max_prompt_length=3072 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.image_key=images \
    "reward.custom_reward_function.path=$VERL_ROOT/verl/utils/reward_score/physics.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=physics_manager \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules=all-linear \
    "actor_rollout_ref.model.exclude_modules=.*visual.*" \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_model_len=6500 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name=PhysReasoning \
    trainer.experiment_name=public-custom-run \
    "trainer.n_gpus_per_node=${N_GPUS:-2}" \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=4 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    "trainer.default_local_dir=${OUTPUT_DIR:-$REPO_ROOT/outputs/training}" \
    "${SFT_ARGS[@]}" "$@"
