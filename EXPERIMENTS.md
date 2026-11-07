# Experiments

What was trained, how it did, and what went wrong. Task: `Mjlab-Velocity-Flat-Unitree-G1`
(flat ground). Every number below comes from `eval_walk.py --speed V`: 64 robots, 20 s each,
the first 2 s after every spawn skipped. "Falls" means falls per robot-minute.

## Runs

| run | starts from | iterations | flags | result |
|---|---|---|---|---|
| `g1_walk` | scratch | 0-1499 | defaults, 4096 envs | 0.93 m/s @ 1, 1.77 @ 2, 2.26 @ 3 (59 falls @ 3). mjlab's speed stages start at iteration 5000, so this run never got commands above 1 m/s. |
| `g1_run` | `g1_walk/model_1499` | 1500-2999 | `--stage_iters 0 500 --fell_over -200 --pose_running_scale 1.5` | 2.98 @ 3, 3.89 @ 4 (0 falls), 4.41 @ 5 (4.2 falls), even though training commands stopped at 3 m/s |
| `g1_sprint` | `g1_run/model_2999` | 3000-3999 | `--num_envs 8192 --num_minibatches 8 --speed_stages 0:4 300:5 --fell_over -200 --pose_running_scale 1.5` | 3.00 @ 3 (0 falls), **0.35 @ 4, 0.00 @ 5 (0 falls): stands still** (see below) |

## Failure: the policy stands still when told to run at 4-5 m/s (`g1_sprint`)

**What happens.** `g1_sprint/model_3999` tracks 3 m/s exactly. Told to go 4 or 5 m/s, it plants
its feet and stays in place: no falls, but almost no forward motion. Its starting point,
`g1_run/model_2999`, ran at 4.4 m/s under the same command. Training never showed this,
because `Episode_Reward/track_linear_velocity` averages over all commands from -2 to 5 m/s
and stayed flat around 1.1.

**Why: under this reward, standing still is the better deal.** Reward per second (weight × value,
before mjlab's ×0.02) at a 5 m/s command:

| term | `g1_run/2999` (runs 4.43 m/s) | `g1_sprint/3999` (stands still) |
|---|---|---|
| track_linear_velocity | 0.37 | 0.00 |
| track_angular_velocity | 0.76 | 2.00 |
| upright | 0.93 | 0.98 |
| pose | 0.67 | 0.88 |
| action_rate, foot_clearance, foot_slip, other costs | -1.49 | 0.00 |
| **total** | **1.23** (plus about 4.7 falls per robot-minute at -4 each) | **3.86** |

- `track_linear_velocity` is `exp(-error² / 0.25)`, so it pays almost nothing once the robot is
  more than about 1 m/s off the command. Running at 4.43 instead of 5 earns only 0.37 of the 2.0
  maximum. Standing earns 0, but so does jogging at 3 m/s (`exp(-16)`), so the term gives no
  signal anywhere between standing and nearly reaching the command.
- A robot standing still collects the other terms almost for free. It has perfect yaw tracking,
  stays upright and holds the default pose. The foot and action-rate costs are all zero when
  nothing moves.
- Standing wins by 2.6 per second even before counting falls. So `--fell_over -200` isn't the
  root cause, but it makes trying to run more costly. At 4 m/s, running would actually score
  higher (3.74 against 3.34), but a standing robot has no tracking reward to find that out.
- How these were measured: a throwaway script, not in the repo, ran each policy in play mode
  at a fixed command for 15 s and averaged `reward_manager._step_reward` (per-term reward per
  second) over the robots that had been up for 2 s or more.

**Fix: `--speed_shortfall W`** (`mjlab_env.speed_shortfall`). It adds a penalty of
W × max(0, |command| − velocity along the command), in m/s. Unlike the exp kernel, every m/s
of missing speed costs the same, so the penalty keeps pushing all the way from 0 to the
commanded speed. Going faster than commanded and standing commands aren't penalized. With
W = -1 at 5 m/s, the totals flip: the running policy scores 0.72 (shortfall -0.55) and the
standing one -1.14 (shortfall -5.00). Running still comes out about 1 per second ahead after
counting its falls.

**Next run (not run yet):** repeat `g1_sprint` with the penalty, starting again from the policy
that still runs:

```bash
uv run python train_walk.py --resume runs/g1_run/model_2999.pt --log_dir runs/g1_sprint2 \
  --num_envs 8192 --num_minibatches 8 --iterations 1000 \
  --speed_stages 0:4 300:5 --fell_over -200 --pose_running_scale 1.5 --speed_shortfall -1
```

**Lesson:** after any run that raises the top speed, check `eval_walk.py --speed` at the new top
speed. The training curves average over all commands and hide a policy that stands still at
the top of the range.
