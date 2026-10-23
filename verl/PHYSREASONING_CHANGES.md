# PhysReasoning changes

This directory contains the supplied verl source tree, based on upstream
`ed913d1414612a1247fd4e70bb4b091777379033` from
[verl-project/verl](https://github.com/verl-project/verl).

Project changes:

- `verl/experimental/agent_loop/modeling_only_agent_loop.py` and its registration:
  restrict the response mask to the physical modeling block.
- `verl/experimental/reward_loop/reward_manager/physics_manager.py` and its
  registration: physics reward management and optional remote execution.
- `verl/utils/reward_score/physics.py`: correctness, rubric, format, and reflection rewards.
- `verl/trainer/ppo/ray_trainer.py`: training reward metrics.
- `verl/utils/dataset/vision_utils.py`: image dictionary handling.

The project extensions use `<modeling>` tags and import `phys_utils` from the
repository root. Run training with `training/verl_grpo.sh` from the parent project.

The upstream `LICENSE` and `Notice.txt` apply to verl. The upstream README,
configuration, installation files, scripts, tests, and examples are retained.
