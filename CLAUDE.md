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
uv run python eval_walk.py --checkpoint runs/g1_walk/model_1499.pt --turn 0.7   # turn in place (add --speed to turn while moving)
uv run python train_walk.py --speed_stages 0:4 300:5 --fell_over -200 --pose_running_scale 1.5 --speed_shortfall -1 --yaw_error -1  # speed curriculum + reward overrides (see --help)
uv run python train_chase.py --legs runs/g1_turn/model_4349.pt                   # runner vs chaser game on frozen legs -> runs/chase
uv run python eval_chase.py --runner flee --chaser pursue                         # scripted baselines; --checkpoint for the learned policy, --video DIR (--camera chaser|overhead)
uv run python eval_chase.py --checkpoint runs/chase/model_499.pt --terrain        # the game on the grass/mud/rocky patchwork (terrain.py)
uv run python eval_terrain.py                                                     # the legs alone on each terrain type: falls, speed, turning, sinking
uv run tensorboard --logdir runs
uv run list-envs                                           # list mjlab task IDs usable with --task
```

There are no tests, linter, or formatter configured. `main.py` is the unused `uv init` stub. Code uses **2-space indentation** (the same style as mjlab).

## Architecture

There are three layers, and each one only talks to the layer next to it:

- **`ppo.py`** is environment-agnostic PPO. It expects any vectorized env with `reset() -> (actor_obs, critic_obs)` and `step(a) -> (actor_obs, critic_obs, reward, terminated, truncated, info)`, all torch tensors with a leading `num_envs` dimension. The env must **auto-reset finished episodes itself**. The pipeline is `collect_rollout` → `compute_gae` → `ppo_update`, run by `train`.
- **`mjlab_env.py`** (`MjlabEnv`) adapts mjlab's `ManagerBasedRlEnv` to that interface. It picks the `"actor"`/`"critic"` observation groups and passes `extras["log"]` as `info`. `play=True` loads the task's evaluation config (no pushes, no observation noise, effectively infinite episode length). With `video_dir` set, it wraps the env in mjlab's `VideoRecorder`.
- **`train_walk.py` / `eval_walk.py`** are thin CLI scripts that connect the two. The default task is `Mjlab-Velocity-Flat-Unitree-G1`. Its config (rewards, terminations, observations) lives in mjlab under `.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/`.
- **`chase_env.py` (`ChaseEnv`)** is a zero-sum runner vs chaser game, built on top of `MjlabEnv`. It has the same `reset`/`step` interface, so `train_chase.py` uses `ppo.train` unchanged. A frozen locomotion checkpoint (`--legs`) turns velocity commands into joint actions. The game policy outputs a command (3 numbers, mapped to the legs' training range) every 0.1 s.
- **`terrain.py` (`TerrainArena`)** is the optional `--terrain` arena: a 5 × 5 patchwork of normal, grass, mud and rocky cells, a new layout per game, with per-terrain grip, softness, mud drag, solid rocks and a forward speed cap. `MjlabEnv(spec_fn=..., expand_fields=...)` lets it add geometry before mjlab compiles the scene and make model fields per-world. `eval_terrain.py` tests the legs alone on each type.

Non-obvious details that matter when changing things:

- **Asymmetric actor-critic:** the critic sees privileged sim observations and the actor does not. Each network has its own `RunningNorm`. The norms are updated only during rollout collection and are saved as buffers in the `state_dict`.
- **Timeout bootstrapping:** on `truncated`, `collect_rollout` adds `gamma * V(s)` to the reward. `done = terminated | truncated` then cuts GAE at that step. Don't "fix" this by dropping one or the other.
- **Adaptive LR:** the learning rate is adjusted for every minibatch from the KL against the stored rollout `mu`/`sigma`. It is kept in `state["lr"]` and saved to checkpoints.
- **`PPOConfig` defaults intentionally mirror mjlab's rsl_rl config** for this task (`mjlab/tasks/velocity/config/g1/rl_cfg.py`). Use that file as the reference when tuning.
- **Checkpoints** (`model_{it}.pt`, every `save_every` iterations plus the final one) store `cfg` and `dims`. This lets `load_actor_critic` rebuild the network without creating an env. Renaming `ActorCritic` submodules or parameters breaks loading of existing checkpoints in `runs/`. `--resume` restores model, optimizer, LR, and iteration, but uses the *current* `PPOConfig()`, not the saved one. `--iterations` counts additional iterations. Task, curriculum and reward flags aren't in the checkpoint; `train_walk.py` appends each run's flags as a line to `<log_dir>/args.jsonl`.
- **Task overrides live in `mjlab_env.py`**: `set_speed_stages` (`--stage_iters`, mjlab's 1/2/3 m/s ladder), `set_custom_speed_stages` (`--speed_stages ITER:VMAX`, any top speed), `set_fixed_speed` and `override_rewards` edit the deep-copied env cfg before `ManagerBasedRlEnv` is built. The speed curriculum (`curriculum["command_vel"]`) counts env steps since the *process* started (24 per iteration) and only applies on robot resets. mjlab's own stages start at iterations 5000/10000, so a default 1500-iteration run never trains above 1 m/s forward. Reward weights are multiplied by `step_dt` (0.02 s).
- **The exp() tracking traps:** mjlab's tracking rewards are `exp(-err²/std²)`. They pay about 0 once the error is large, so they give no learning signal there.
  - **Standing still at high speed:** `track_linear_velocity` is about 0 once the robot is more than about 1 m/s off the command. A robot that stands still still collects the upright and pose rewards and pays no foot or action costs, so when the command is out of reach, the policy learns to stand still. `g1_sprint` did this at 4-5 m/s. `--speed_shortfall` (a linear penalty per m/s of missing speed) fixes it.
  - **Not turning:** `track_angular_velocity` also counts the torso's roll and pitch rates, which are large when running. So it pays about the same whether the robot turns or not, and `g1_run` → `g1_sprint2` gradually stopped turning (yaw rate 0 for a 0.7 rad/s command). `--yaw_error` (a linear penalty per rad/s of yaw-rate error) is the fix.
  - Training curves average over all commands and hide both problems. After changing locomotion rewards, check `eval_walk.py --speed <top speed>` and `--turn 0.7`. `EXPERIMENTS.md` has the run history and the numbers.
- **`ChaseEnv` ghost worlds:** runner i is mjlab world i and chaser i is world G+i, so the two robots never touch, and a catch means their pelvises come within `catch_radius`.
  - Every world uses the same coordinate frame: the arena is centred at (0, 0), and `MjlabEnv.place_robots` writes positions that ignore `env_origins`.
  - mjlab must never change the commands (`external_commands=True`), and `set_command` recomputes the actor observations so the legs see a new command on the same step.
  - The 20 s limit is a game rule, so it's returned as `terminated`, never `truncated`.
  - One network plays both roles, so the logged `mean_return` is about 0 by construction. Watch the `Game/*` win rates instead.
  - Game checkpoints don't record which `--legs` were used; `args.jsonl` does.
  - The video overlay works by adding `ChaseEnv` itself to mjlab's `manager_visualizers`, which mjlab draws on every rendered frame.
  - The default `--camera chaser` moves mjlab's private `_offline_renderer._cam` from `debug_vis` each frame, so a camera change shows up one frame late. If an mjlab upgrade breaks it, use `--camera overhead`.
- **Terrain (`terrain.py`) and MuJoCo Warp.** Each of these was found the hard way (`EXPERIMENTS.md`, "Terrain arena"):
  - **Per-world fields.** Only fields expanded with `sim.expand_model_fields` can differ between worlds; everything else is one shared copy. The two worlds of a game must get the same layout.
  - **Contact priority decides whose grip counts.** The G1's foot geoms have priority 1, so the ground's friction and softness are ignored unless the ground's priority is higher. The plane gets priority 2, and on normal ground it copies each robot's own (randomized) foot friction.
  - **Don't build the ground from separate box tiles.** Feet catch on the seams: falls at 5 m/s went from 8% to 16%. Instead, each world's plane takes the grip and softness of the cell under its robot every 0.02 s (`step_effects`), which works because each world holds one robot. The colored tiles are visual only.
  - **Geoms fixed to the world never move per world.** MuJoCo Warp computes their positions once, when the data is created, so per-world `geom_pos` does nothing. Movable obstacles must be mocap bodies (`data.mocap_pos`). Also set their per-world `body_pos`/`body_quat`: a world reset puts mocap bodies back there.
  - **`--terrain` keeps the 23-number game input,** so `chase/model_499` loads but is blind to the terrain. Rocks are not in mjlab's `terrain` body, so foot-contact sensors don't count them as ground.
- **`log_keys`** in `train_walk.py` must exactly match keys in mjlab's `extras["log"]` (e.g. `Episode_Termination/fell_over`). Keys that don't match are silently skipped. `Episode_Reward/<term>` is added automatically for every term with a nonzero weight (`MjlabEnv.reward_terms`). Only `print_keys` are shown on the console line.
- **Eval fall counting** relies on play mode having no time limit, so every `terminated` is a fall. `eval_walk.py` hardcodes `0.02` s per step for the video length. That equals the task's `step_dt` (timestep 0.005 × decimation 4), so update it if the task's timing changes.

`runs/` (TensorBoard logs plus about 5 MB per checkpoint) and `videos/` are git-ignored. Results worth keeping go in `EXPERIMENTS.md`.

`README.md` is the shareable overview: the training story, the game, a step-by-step guide and the command reference. Its GIFs live in `docs/`. It quotes numbers from `EXPERIMENTS.md`, so update both together. It also lists every file with its line count.
