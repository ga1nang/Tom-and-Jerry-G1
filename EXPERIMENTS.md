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

**The shaping reward makes leaving the arena cheaper than being caught.**
- Each decision, the runner's shaping pays +1 per metre it opens and -1 per metre the chaser
  closes. Over a game, that adds up to (distance at the end − distance at the start).
- On the decision where a game ends, the shaping is skipped (`torch.where(ended, 0.0, ...)`
  in `ChaseEnv.step`). So a runner that loses keeps all the distance it gained.

Measured with `model_499`, 512 games each, with the distance at the last decision before
each game ended:

| chaser | outcome | share | distance at the end | runner's shaping | runner's total |
|---|---|---|---|---|---|
| `pursue` | caught | 2.1% | 1.0 m | -4.5 | **-14.5** |
| `pursue` | runner out | 12.1% | **9.4 m** | +4.3 | **-5.7** |
| learned | caught | 38.1% | 0.9 m | -4.2 | -14.2 |
| learned | runner out | 37.9% | 4.1 m | -1.1 | -11.1 |

- **Against `pursue`, leaving the arena scores 9 points better than being caught.** The runner
  leaves with the chaser 9.4 m behind, so nothing forces it out. It runs straight away,
  collecting shaping on every decision, and doesn't slow down for the wall.
- **Discounting turns the shaping into a "stay far away" reward.** With γ = 0.995, the
  discounted shaping is a time-weighted average of the distance over the game, minus the
  starting distance.

**Fix, not tried yet: proper potential-based shaping.** Discount the new distance with γ, and
refund the remaining distance when the game ends:

```python
chaser_reward = shaping * torch.where(ended, self.dist, self.dist - gamma * dist)
```

Every outcome's total shaping then equals the starting distance. The shaping still guides
learning but no longer changes which outcome is better, so leaving and being caught both cost
exactly -10.

**Next ideas** (none run yet):
- legs that turn faster (see "Failure: the legs don't turn");
- lower entropy, or tanh actions;
- train against an opponent pool sampled from older checkpoints, against cycling;
- potential-based shaping (above), to stop rewarding the runner for leaving the arena far
  from the chaser.

## Terrain arena (evaluation only): grass, mud and rocks (`--terrain`)

`terrain.py` turns the arena into a 5 × 5 patchwork of 4 m cells: 5 normal, 6 grass, 6 mud and
8 rocky cells (20/24/24/32%). Every game draws a new layout, with no two side-by-side cells of
the same type. The first game results below used 8 normal and 5 rocky cells (32/24/24/20%).
The rockier layout, the current default, comes at the end of this section. Nothing was trained for this: the legs are `g1_turn/model_4349` and the game
policy is `chase/model_499`, which can't see the terrain (its 23 inputs are unchanged).

| terrain | grip | softness (standing, feet sink) | extra | forward speed cap (game rule) |
|---|---|---|---|---|
| normal | the robot's own foot grip (0.3-1.2, as on the flat arena) | none | | 5 m/s |
| grass | 0.5 | 0.4 cm | | 4 m/s |
| mud | 0.4 | 1.0 cm | drag on the pelvis, 0.5/s × mass × velocity | 2 m/s |
| rocky | own foot grip | none | 5 solid rocks per cell: 0.3-0.8 m wide, 8-30 cm tall | 3 m/s |

**How it's built, and three things that went wrong on the way:**
- **The ground's grip was ignored at first.** The G1's foot geoms have contact priority 1, so
  their own friction wins over the ground's. The ground now has priority 2. Normal ground
  copies each robot's own foot friction, so it behaves exactly like the flat arena.
- **Box tiles tripped the robots at the seams.** At first every cell was a box geom with its
  own grip and softness. Starting from standing at 5 m/s for 3.2 s (128 robots × 2 runs),
  falls were:
  - plain plane: 8%;
  - one big box: 11%;
  - 4 m tiles: 16%.

  Now each world's plane takes the grip and softness of the cell under its robot, every 0.02 s
  (each world has only one robot). The colored tiles are visual only.
- **Rocks didn't move.** MuJoCo Warp computes the position of geoms fixed to the world only
  once, when the data is created (`mujoco_warp/_src/smooth.py`), so writing a per-world
  `geom_pos` did nothing. The robots walked through a "wall" of rocks.
  - The rocks are now mocap bodies, moved through `data.mocap_pos`.
  - Their per-world `body_pos` is set too, because a world reset puts mocap bodies back there.
  - After the fix, a low wall of rocks trips 100% of the robots at 1 m/s.

**Calibrating mud** (32 robots per variant). Speed after the first second, then the share
that fell:

| mud variant | 2 m/s | 3 m/s | 5 m/s |
|---|---|---|---|
| grip 0.3, hard, no drag | 2.03, 0% | 2.98, 0% | 1.88, **100%** |
| own grip, hard, drag 1.0/s | 1.48, 0% | 2.11, 0% | 2.92, 9% |
| grip 0.3, soft, drag 1.0/s (first try) | 1.50, 0% | 1.84, 38% | 0.00, 100% |
| **grip 0.4, soft, drag 0.5/s (chosen)** | **1.78, 0%** | **2.58, 0%** | **2.94, 84%** |

Low grip is what makes mud deadly at speed, and drag alone mostly slows the robot down. The
chosen mud is safe at its 2 m/s cap and dangerous if a robot runs into it fast.

**The legs alone** (`eval_terrain.py`). 64 robots per terrain, each running straight across a
whole arena of that type, with no speed caps. The rocky lanes have the same rock density as a
rocky cell. Each cell gives the speed (or turn rate) and the share that fell:

| commanded | normal | grass | mud | rocky |
|---|---|---|---|---|
| 1 m/s | 0.99, 0% | 1.03, 0% | 0.85, 0% | 0.96, **86%** |
| 3 m/s | 3.03, 0% | 3.06, 0% | 2.58, 0% | 2.93, **91%** |
| 5 m/s | 4.77, 3% | 4.60, 0% | 3.01, **84%** | 4.32, **80%** |
| 3 m/s + turn 0.7 rad/s | 0.21 rad/s, 0% | 0.22, 0% | 0.20, 0% | 0.16, 75% |

Grass and mud are fine at their speed caps. **The legs are blind, so they trip over rocks at
any speed.**

**The game on terrain**, first layout: 8 normal and 5 rocky cells (256 games each; percentages
are outcomes):

| chaser vs runner | arena | caught | time up | runner fell | chaser fell | runner out | chaser out |
|---|---|---|---|---|---|---|---|
| `pursue` vs `flee` | flat | 20-27 | 71-77 | 1.6 | 1.2 | 0.0 | 0.0 |
| `pursue` vs `flee` | terrain | 16.0 | 7.4 | **48.4** | 28.1 | 0.0 | 0.0 |
| learned vs `flee` | terrain | 41.0 | 1.6 | 33.2 | 22.7 | 0.0 | 1.6 |
| `pursue` vs learned | terrain | 1.2 | 11.3 | 43.0 | 39.1 | 5.5 | 0.0 |
| learned vs learned | terrain | 16.4 | 1.6 | 42.6 | 26.6 | 12.5 | 0.4 |

The flat row is three runs of the same command (evaluations aren't seeded), to show the
run-to-run spread. The recorded flat result was 27.7 / 68.0.

- **Rocks decide the game.** Between 136 and 200 of the falls in each match-up happened on a
  rocky cell, against at most 5 on normal, grass or mud. Robots spend only 10-15% of their time
  on rocky cells, because games there end fast.
- **The speed caps work as intended.** Grass and mud cost speed, but almost never a fall.
- **Nothing here can see the terrain.** The scripted players, the game policy and the legs are
  all blind to it, so "time up" wins nearly vanish: few games last 20 s without someone
  tripping.

**Rockier layout: 5 normal, 6 grass, 6 mud and 8 rocky cells** (current default, 40 rocks).

**The layout generator was biased.** Backtracking fills the grid in reading order, so the most
common type ran short of room early and piled up in the last rows. Share of rocky cells per
grid row, over the 4096 layouts of the bank (row 5 is the strip at y = 6-10 m):

| generator | row 1 | row 2 | row 3 | row 4 | row 5 |
|---|---|---|---|---|---|
| backtracking only | 0.25 | 0.25 | 0.28 | 0.37 | 0.45 |
| backtracking, then 2000 random swaps (now) | 0.33 | 0.31 | 0.32 | 0.31 | 0.33 |

- The first layout had the same bias for normal ground: up to 52% of a last-row cell, against
  32% overall. So the first game table was played with more normal ground on one side.
- The fix shuffles each layout: swap two random cells, and keep the swap if the layout is
  still valid. Every swap can be undone just as easily, so the swaps make all valid layouts
  about equally likely. Each cell is now 28-36% rocky, slightly more at corners, which have
  fewer neighbours.
- The backtracking stays as a varied starting point. Starting all 4096 layouts from one
  layout still showed it after 4000 swaps (one cell 23% rocky, another 40%).

**The game** (256 games each, same policies as above):

| chaser vs runner | caught | time up | runner fell | chaser fell | runner out | chaser out | falls on rocky / elsewhere |
|---|---|---|---|---|---|---|---|
| `pursue` vs `flee` | 11.3 | 1.2 | **50.0** | 37.1 | 0.0 | 0.0 | 216 / 7 |
| learned vs `flee` | 30.5 | 0.4 | 44.5 | 24.6 | 0.0 | 0.0 | 168 / 9 |
| `pursue` vs learned | 1.6 | 2.3 | 48.0 | 43.4 | 4.7 | 0.0 | 225 / 9 |
| learned vs learned | 9.8 | 0.4 | 43.4 | 34.4 | 11.7 | 0.4 | 185 / 14 |

- **More rocks, more falls.** Falls end 69-91% of games, against 56-82% on the first layout.
  Catches drop by 26-40% (`pursue` vs learned was already near 1%), and "time up" wins almost
  vanish (at most 2.3%).
- Robots spend 17-24% of their time on rocky cells, against 10-15% before.

**The legs alone, repeated.** `eval_terrain.py` keeps its rocky lanes at 25 rocks
(`LANE_ROCKS`, about a rocky cell's density), not the arena's 40. The rocks now have new sizes,
drawn from the same ranges. Two runs (evaluations aren't seeded, 64 robots per terrain), the
share that fell:

| commanded | normal | grass | mud | rocky |
|---|---|---|---|---|
| 1 m/s | 0%, 0% | 0%, 0% | 0%, 0% | 61%, 72% |
| 3 m/s | 0%, 0% | 0%, 0% | 0%, 0% | 86%, 94% |
| 5 m/s | 3%, 11% | 0%, 0% | 75%, 72% | 86%, 84% |
| 3 m/s + turn 0.7 rad/s | 0%, 0% | 0%, 0% | 0%, 0% | 70%, 81% |

The speeds and turn rates are close to the first table's. Within this run-to-run spread, the legs
behave as before.

Commands:

```bash
uv run python eval_terrain.py
uv run python eval_chase.py --runner flee --chaser pursue --terrain
uv run python eval_chase.py --checkpoint runs/chase/model_499.pt --terrain                 # also --runner flee / --chaser pursue
uv run python eval_chase.py --checkpoint runs/chase/model_499.pt --terrain --video videos/terrain --camera overhead
```

**Next ideas** (none run yet):
- Give the game policy the terrain: the cell types around it and the nearest rocks. Then
  retrain it with `--terrain`, so it learns to avoid rocky cells, or to herd its opponent into
  them.
- Make the scripted players avoid rocky cells, so they stay a fair yardstick.
- Legs that can see. mjlab's rough-terrain task has a height scan, but that changes the legs'
  inputs, so they'd have to be retrained.
