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
| `g1_sprint2` | `g1_run/model_2999` | 3000-3999 | same as `g1_sprint` + `--speed_shortfall -1` | **5.05 @ 5, 4.08 @ 4, 3.04 @ 3**, 0 falls at every speed and with random commands. **It can't turn** (see below). |
| `g1_turn` | `g1_sprint2/model_3999` | 4000-4499 | `--num_envs 8192 --num_minibatches 8 --speed_stages 0:5 --fell_over -200 --pose_running_scale 1.5 --speed_shortfall -1 --yaw_error -1` | Turns a little: 0.12-0.22 rad/s for a 0.7 command. **It collapsed after about iteration 4400**: `model_4499` falls 254 times per robot-minute at 5 m/s. Best checkpoint: `model_4349` (4.96 @ 5, 0 falls). |

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

**Result (`g1_sprint2`):** fixed. This repeats `g1_sprint` with only the penalty added, starting
again from the policy that still runs. It reaches 5.05 m/s at 5, 4.08 at 4 and 3.04 at 3, with
0 falls at every speed and with random commands.

```bash
uv run python train_walk.py --resume runs/g1_run/model_2999.pt --log_dir runs/g1_sprint2 \
  --num_envs 8192 --num_minibatches 8 --iterations 1000 \
  --speed_stages 0:4 300:5 --fell_over -200 --pose_running_scale 1.5 --speed_shortfall -1
```

**Lesson:** after any run that raises the top speed, check `eval_walk.py --speed` at the new top
speed. The training curves average over all commands and hide a policy that stands still at
the top of the range.

## Failure: the legs don't turn (`g1_run` → `g1_sprint2`)

**What happens.** We found this when the runner vs chaser game's robots couldn't steer.
Commanded to turn at 0.7 rad/s, `g1_sprint2/model_3999` doesn't turn at any speed:

| command | `g1_run/2999` yaw rate | `g1_sprint2/3999` yaw rate |
|---|---|---|
| 0 m/s, 0.7 rad/s (turn in place) | 0.15 | 0.00 |
| 0.5 m/s, 0.7 rad/s | 0.44 | 0.00 |
| 1 m/s, -0.7 rad/s | -0.38 | -0.03 |
| 3 m/s, 0.7 rad/s | 0.20 | 0.06 |
| 5 m/s, 0.7 rad/s | 0.08 (falls) | -0.01 |

These were measured with a throwaway script (8 robots per command, 6 s, the first 2 s skipped).
`eval_walk.py --turn 0.7` now prints the same check. Forward speed still matches the command
exactly. `eval_walk.py` used to print only forward speed, so nobody noticed.

**When it started.** Turning got worse over the whole run history. Logged
`Episode_Reward/track_angular_velocity` (2.0 is perfect tracking):
- `g1_run`: 0.75, falling to 0.37
- `g1_sprint`: about 0.3
- `g1_sprint2`: 0.16, falling to 0.055

**Why: the same exp() trap as standing still.** mjlab's turning reward is
`exp(-(yaw error² + roll rate² + pitch rate²) / 0.5)`. A running torso rocks in roll and pitch,
and that alone pushes the term close to 0 at speed: at 5 m/s with no turn commanded, the
running policy got 0.76 of 2.0 (table above). So turning or not barely changes the reward, and
there's nothing to learn from. Turning in place is rarely commanded, because forward commands
are uniform over -2 to 5 m/s.

**Fix: `--yaw_error W`** (`mjlab_env.yaw_error`). It adds a penalty of W × |commanded − actual
yaw rate|, in rad/s, and ignores roll and pitch. Like `--speed_shortfall`, it's linear, so
every bit of turning pays. At 5 m/s with no turn commanded, the running gait's yaw wobble
already measures 0.42 rad/s, so with W = -1 the penalty also discourages pelvis twist and may
cost a little top speed.

**Result (`g1_turn`): only partly fixed, and the run collapsed at the end.** It fine-tuned
`g1_sprint2` with the penalty added and nothing else changed (command below). 32-64 robots per
check:

| checkpoint | 5 m/s straight | 4 m/s straight | turn in place (0.7) | 3 m/s + turn 0.7 |
|---|---|---|---|---|
| `g1_sprint2/3999` | 5.05, 0 falls | 4.08, 0 falls | 0.00 rad/s | 0.04 rad/s |
| `g1_turn/4049` | 5.01, 0 falls | | | 0.11 rad/s |
| `g1_turn/4249` | 4.92, 0.7 falls | | | 0.20 rad/s |
| `g1_turn/4349` (best) | 4.96, 0 falls | 4.01, 0 falls | 0.12 rad/s | 0.22 rad/s |
| `g1_turn/4399` | 4.99, 10.5 falls | | | |
| `g1_turn/4499` (last) | **254 falls** (falls within 0.3 s of spawning) | **185 falls** | 0.17 rad/s | 0.22 rad/s |

- Turning improved, but only to about a third of the command. A robot turning at 0.22 rad/s
  while running at 3 m/s needs a circle 27 m across, bigger than the 20 m game arena.
- Training falls (`Episode_Termination/fell_over`) went from about 1.5 to 6 between iterations
  4400 and 4499, and episodes got shorter. Meanwhile KL (about 0.015), learning rate and
  action std stayed normal. So the reward pushed the policy there; the optimizer didn't blow up.
  The likely cause, not verified: turning at 0.7 rad/s while running at 4-5 m/s means about
  3.5 m/s² of sideways acceleration, and the penalty keeps pushing for it.
- Random commands and forward-speed tracking are fine at `model_4349`: 0 falls.

Command used:

```bash
uv run python train_walk.py --resume runs/g1_sprint2/model_3999.pt --log_dir runs/g1_turn \
  --num_envs 8192 --num_minibatches 8 --iterations 500 \
  --speed_stages 0:5 --fell_over -200 --pose_running_scale 1.5 --speed_shortfall -1 --yaw_error -1
```

**Lessons:**
- Check every command axis (`--speed`, `--turn`), not just the one being pushed.
- Check a few checkpoints, not only the last one. The run can go bad near the end, and only
  the evaluation catches it.

## Game: runner vs chaser (`chase_env.py`)

The sprint legs are frozen, and a game policy picks each robot's command 10 times per second.
Rules are in `chase_env.py`'s docstring.

**Status:** the env is built and checked:
- nobody moving gives 100% runner wins at exactly 20 s;
- starting 0.5 m apart is an instant catch;
- running into the wall is `runner_out`;
- a tipped-over robot is `chaser_fell`;
- rewards sum to exactly 0;
- training, resume and the overhead video all work.

Training speed: 10.4 s per iteration at 2048 games (4096 robots), 1.3 GB of GPU memory.

With `g1_sprint2` legs, no robot could steer: scripted `flee` vs `pursue` came out 92% runner
wins and 8% catches, which isn't a real baseline. All results below use
`g1_turn/model_4349` legs, which turn at about 0.2 rad/s.

**Run `chase`:** `train_chase.py --legs runs/g1_turn/model_4349.pt --log_dir runs/chase`, with
defaults of 2048 games and 500 iterations. It took 84 minutes at 10.1 s per iteration.

Results with 256 games each and the mean (deterministic) action. Percentages are outcomes of
each game:

| chaser vs runner | caught | time up | runner fell | chaser fell | runner out | chaser out |
|---|---|---|---|---|---|---|
| scripted `pursue` vs scripted `flee` (baseline) | 27.7 | 68.0 | 1.6 | 2.3 | 0.0 | 0.4 |
| **learned** (`model_499`) vs `flee` | **87.9** | 5.9 | 2.7 | 0.8 | 0.0 | 2.7 |
| `pursue` vs **learned** (`model_499`) | 3.5 | **77.7** | 2.0 | 2.0 | 14.8 | 0.0 |
| learned vs learned (`model_499`) | 42.6 | 9.4 | 4.3 | 5.5 | 37.1 | 1.2 |
| learned (`model_249`) vs `flee` | 83.6 | 12.1 | 0.4 | 1.6 | 0.0 | 2.3 |
| `pursue` vs learned (`model_249`) | 2.7 | 60.5 | 6.2 | 1.2 | 29.3 | 0.0 |

- **Both learned roles beat the scripted ones.**
  - The learned chaser wins 91% of games against `flee`, where `pursue` wins 29%.
  - The learned runner wins 80% against `pursue`, where `flee` wins 70%.
- **The runner's weak spot is the walls.** It is almost never caught by `pursue` (3.5%), but
  leaves the arena in 15% of games, down from 29% at iteration 249. Turning at about 0.2 rad/s,
  it can't steer back in time near a wall.
- **The learned chaser exploits that.** Against the learned runner, 37% of games end with the
  runner pushed out of the arena.
- **The two learned sides are unbalanced:** the chaser side wins 84% of learned vs learned
  games. Win rates between them swing a lot during training (chaser 24-57%), which is normal
  for self-play, so judge progress against the scripted players.
- **Action std grew from 0.5 to 1.04.** Actions are clamped to [-1, 1], so beyond about 1 the
  noise mostly produces saturated, random commands. A smaller `entropy_coef` or squashing the
  actions with tanh should help the next run.
- **Video:** `videos/chase/rl-video-step-0.mp4` shows learned vs learned. The chaser drives the
  runner to a wall, and the runner drifts out.

**Next ideas** (none run yet):
- legs that turn faster (see "Failure: the legs don't turn");
- lower entropy, or tanh actions;
- train against an opponent pool sampled from older checkpoints, against cycling.
