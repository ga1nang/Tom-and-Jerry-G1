# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch PPO implementation that teaches the Unitree G1 humanoid to walk. **mjlab is used only as the simulator.** mjlab also installs `rsl_rl` and its own `train`/`play` scripts. This project deliberately does not use them, so don't replace `ppo.py` with rsl_rl. The code is meant to be read top to bottom, so it has plain-language comments and no framework indirection. Keep it that way.

## Commands

The environment is managed with uv (Python 3.12, `mjlab==1.6.0` is the only direct dependency). A CUDA GPU is used automatically if one is available.

```bash
uv sync                                                    # install deps into .venv
uv run python train_walk.py                                # full run: 4096 envs, 1500 iterations -> runs/g1_walk
uv run python train_walk.py --num_envs 64 --iterations 2 --log_dir runs/smoke   # quick smoke test
uv run python train_walk.py --resume runs/g1_walk/model_1499.pt --iterations 500  # continue training
uv run python eval_walk.py --checkpoint runs/g1_walk/model_1499.pt               # count falls over 64 robots
uv run python eval_walk.py --checkpoint runs/g1_walk/model_1499.pt --video videos/walk  # record an mp4 (1 robot)
uv run python eval_walk.py --checkpoint runs/g1_walk/model_1499.pt --speed 2.0  # fixed straight command; prints commanded vs achieved speed
uv run python train_walk.py --speed_stages 0:4 300:5 --fell_over -200 --pose_running_scale 1.5 --speed_shortfall -1  # speed curriculum + reward overrides (see --help)
uv run tensorboard --logdir runs
uv run list-envs                                           # list mjlab task IDs usable with --task
```

There are no tests, linter, or formatter configured. `main.py` is the unused `uv init` stub. Code uses **2-space indentation** (the same style as mjlab).

## Architecture

There are three layers, and each one only talks to the layer next to it:

- **`ppo.py`** is environment-agnostic PPO. It expects any vectorized env with `reset() -> (actor_obs, critic_obs)` and `step(a) -> (actor_obs, critic_obs, reward, terminated, truncated, info)`, all torch tensors with a leading `num_envs` dimension. The env must **auto-reset finished episodes itself**. The pipeline is `collect_rollout` → `compute_gae` → `ppo_update`, run by `train`.
- **`mjlab_env.py`** (`MjlabEnv`) adapts mjlab's `ManagerBasedRlEnv` to that interface. It picks the `"actor"`/`"critic"` observation groups and passes `extras["log"]` as `info`. `play=True` loads the task's evaluation config (no pushes, no observation noise, effectively infinite episode length). With `video_dir` set, it wraps the env in mjlab's `VideoRecorder`.
- **`train_walk.py` / `eval_walk.py`** are thin CLI scripts that connect the two. The default task is `Mjlab-Velocity-Flat-Unitree-G1`. Its config (rewards, terminations, observations) lives in mjlab under `.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/`.

Non-obvious details that matter when changing things:

- **Asymmetric actor-critic:** the critic sees privileged sim observations and the actor does not. Each network has its own `RunningNorm`. The norms are updated only during rollout collection and are saved as buffers in the `state_dict`.
- **Timeout bootstrapping:** on `truncated`, `collect_rollout` adds `gamma * V(s)` to the reward. `done = terminated | truncated` then cuts GAE at that step. Don't "fix" this by dropping one or the other.
- **Adaptive LR:** the learning rate is adjusted for every minibatch from the KL against the stored rollout `mu`/`sigma`. It is kept in `state["lr"]` and saved to checkpoints.
- **`PPOConfig` defaults intentionally mirror mjlab's rsl_rl config** for this task (`mjlab/tasks/velocity/config/g1/rl_cfg.py`). Use that file as the reference when tuning.
- **Checkpoints** (`model_{it}.pt`, every `save_every` iterations plus the final one) store `cfg` and `dims`. This lets `load_actor_critic` rebuild the network without creating an env. Renaming `ActorCritic` submodules or parameters breaks loading of existing checkpoints in `runs/`. `--resume` restores model, optimizer, LR, and iteration, but uses the *current* `PPOConfig()`, not the saved one. `--iterations` counts additional iterations. Task, curriculum and reward flags aren't in the checkpoint; `train_walk.py` appends each run's flags as a line to `<log_dir>/args.jsonl`.
- **Task overrides live in `mjlab_env.py`**: `set_speed_stages` (`--stage_iters`, mjlab's 1/2/3 m/s ladder), `set_custom_speed_stages` (`--speed_stages ITER:VMAX`, any top speed), `set_fixed_speed` and `override_rewards` edit the deep-copied env cfg before `ManagerBasedRlEnv` is built. The speed curriculum (`curriculum["command_vel"]`) counts env steps since the *process* started (24 per iteration) and only applies on robot resets. mjlab's own stages start at iterations 5000/10000, so a default 1500-iteration run never trains above 1 m/s forward. Reward weights are multiplied by `step_dt` (0.02 s).
- **Standing-still trap at high commanded speed:** `track_linear_velocity` is `exp(-err²/0.25)`, which is about 0 once the robot is more than about 1 m/s off the command. A robot that stands still still collects the yaw-tracking, upright and pose rewards and pays no foot or action costs. So when the command is out of reach, the policy learns to stand still. `g1_sprint` did exactly this at 4-5 m/s. `--speed_shortfall` (a linear penalty per m/s of missing speed) is the fix. Training curves average over all commands and hide this, so check `eval_walk.py --speed <top speed>` after raising the speed. `EXPERIMENTS.md` has the run history and the per-term numbers.
- **`log_keys`** in `train_walk.py` must exactly match keys in mjlab's `extras["log"]` (e.g. `Episode_Termination/fell_over`). Keys that don't match are silently skipped. `Episode_Reward/<term>` is added automatically for every term with a nonzero weight (`MjlabEnv.reward_terms`). Only `print_keys` are shown on the console line.
- **Eval fall counting** relies on play mode having no time limit, so every `terminated` is a fall. `eval_walk.py` hardcodes `0.02` s per step for the video length. That equals the task's `step_dt` (timestep 0.005 × decimation 4), so update it if the task's timing changes.

`runs/` (TensorBoard logs plus about 5 MB per checkpoint) and `videos/` are **not** in `.gitignore`, so don't commit them by accident.
