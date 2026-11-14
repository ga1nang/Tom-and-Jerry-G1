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

**Old vs new layout in the game.** 1024 games per match-up, same policies:
- **old:** 8 normal and 5 rocky cells, from the biased generator;
- **new:** 5 normal and 8 rocky cells, shuffled.

Evaluations aren't seeded; with 1024 games, each figure is within about ±3 points (95%). Draws
and "chaser out", at most 1.3% each, are left out. "Someone fell" adds up both falls. "Chaser
wins, any way" adds up the catches, the runner's falls and the runner leaving the arena.

| chaser vs runner | layout | caught | time up | runner fell | chaser fell | runner out | someone fell | chaser wins, any way |
|---|---|---|---|---|---|---|---|---|
| `pursue` vs `flee` | old | 14.5 | 6.6 | 45.6 | 33.0 | 0.1 | 78.6 | 60.2 |
| | new | 12.0 | 1.6 | 47.2 | 39.2 | 0.1 | **86.4** | 59.3 |
| learned vs `flee` | old | 44.0 | 2.2 | 31.7 | 20.8 | 0.0 | 52.5 | 75.7 |
| | new | 31.0 | 0.6 | 40.4 | 27.4 | 0.2 | **67.8** | 71.6 |
| `pursue` vs learned | old | 1.6 | 11.2 | 42.4 | 35.7 | 9.1 | 78.1 | 53.1 |
| | new | 1.1 | 3.2 | 45.0 | 45.0 | 5.7 | **90.0** | 51.8 |
| learned vs learned | old | 16.9 | 1.3 | 41.4 | 27.0 | 13.0 | 68.4 | 71.3 |
| | new | 10.9 | 0.5 | 48.4 | 31.6 | 8.4 | **80.0** | 67.7 |

- **Falls replace catches and time-ups.** With 8 rocky cells, falls end 68-90% of games,
  against 53-79% on the old layout.
  - Catches drop by up to 13 points: the learned chaser against `flee` goes from 44.0 to 31.0.
  - "Time up" wins drop to at most 3.2%.
  - Runners leave the arena less often, because more games end in a fall first.
- **Who wins hardly changes.** The chaser's wins, counting every way of winning, move by at
  most 4 points. Both robots are equally blind, so the extra rocks trip both sides about as
  often. Rocks decide more of the games, but they favour neither role.
- **Robots spend 17-22% of their time on rocky cells, against 11-14% before.** That's about
  the 8 : 5 ratio of rocky cells, because nobody steers around them.
- **90-98% of all falls happen on rocky cells,** on both layouts.
- **The 1024-game runs of the old layout agree with the 256-game table above** within noise,
  so that table still stands.

**Which change did it?** The rockier layout and the layout fix came together, so all four
combinations were run (1024 games each). Share of games that ended in a fall:

| chaser vs runner | 5 rocky, biased (old) | 5 rocky, shuffled | 8 rocky, biased | 8 rocky, shuffled (now) |
|---|---|---|---|---|
| `pursue` vs `flee` | 78.6 | 76.1 | 84.8 | 86.4 |
| learned vs `flee` | 52.5 | 48.2 | 68.2 | 67.8 |
| `pursue` vs learned | 78.1 | 75.2 | 91.5 | 90.0 |
| learned vs learned | 68.4 | 58.2 | 79.2 | 80.0 |

- **The extra rocky cells explain the difference.** They add 6-22 points of falls with either
  generator.
- **The layout fix matters little.** With 8 rocky cells, it moves falls by less than 2 points,
  which is noise. With 5 rocky cells, it lowered falls by 2.5-10 points, the most for learned vs
  learned. The cause wasn't checked.
- To reproduce the old layout, set `CELL_COUNTS = (8, 6, 6, 5)` in `terrain.py` and call
  `make_layouts(..., swaps=0)`.

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

## Stand-up after falls (`--standup`)

On the terrain arena, 68-91% of games end with a fall, and a fall ends the game at once. That
hides everything else, including the question of how long a game should be. `--standup CKPT`
turns a fall into lost time: a frozen stand-up policy gets the robot up again (`standup.py`),
and the robot loses only if it isn't standing 5 s after its 0.6 s settle (`--standup_s`).
Nothing in this project was retrained: the legs are `g1_turn/model_4349` and the game policy is
`chase/model_499`, with its 23 inputs unchanged.

**The stand-up policy.** `runs/standup/model_11000.pt` is a copy of
`../HoST/mjlab_port/runs/model_11000.pt`: HoST (Huang et al., RSS 2025) ported to mjlab 1.6.0
on the same 29-DoF G1, trained 11000 iterations on its `Host-G1-Rocks` task (dropped on its
back onto rocky ground). Its own evaluation (`scripts/eval.py` there, 300 robots; success =
pelvis above 0.7 m within 5 s, never below 0.5 m afterwards):

| start | success | median time to 0.7 m |
|---|---|---|
| on its back, flat (`Host-G1-Ground`) | 99.3% | 1.72 s |
| on its front, flat (`Host-G1-Prone`) | 69.3% | 4.16 s |
| on its back, rocks (`Host-G1-Rocks`) | 88.3% | 1.96 s |

**How it's switched in.** Each robot has a state machine:
- **WALK:** the legs drive. Tilting past 70° (mjlab's own `fell_over` angle) is a fall.
- **STANDUP:** 0.6 s limp (HoST's "unactuated window": zero actions and inputs while the body
  settles), then the stand-up policy. Pelvis above 0.7 m with the torso within 20° of upright,
  for 0.4 s, hands the robot back.
- **STEADY:** the legs again, told to stand still for 0.5 s, then WALK.

The stand-up policy differs from the legs in almost everything, and each difference had to be
reproduced:
- **Its control law.** Targets are the current joint angles plus β·(kp_host/kp_mjlab)·action,
  recomputed every 0.005 s physics substep. The 6 joints its 23-joint model lacks are held at 0.
- **Its damping.** It was trained with HoST's joint damping. The damping is a per-world model
  field (`actuator_biasprm`), swapped when a robot falls and swapped back at the hand-over.
- **Its input:** 6 frames of 76 numbers, oldest first, all 0 during the window.
- **One action term, not two.** The legs' "last action" input reads mjlab's whole action
  vector, so a second action term would change the legs' input size. Instead the legs' term is
  replaced by one that overrides the targets of the worlds getting up (`StandupAction`).
- **mjlab's `fell_over` termination is removed.** It respawns a fallen robot standing.
- **Contact buffers:** a lying robot touches the ground in many places, so nconmax 64 and
  njmax 600 (the flat task: 300). No test below overflowed.
- **The network matches.** `load_standup_actor` rebuilds the rsl_rl actor as a plain MLP. On
  1000 random inputs it gives exactly the output of HoST's TorchScript export.
- **Not reproduced:** the per-robot gain, motor-strength and offset randomization and the action
  delay of training. These run at their nominal values, which are inside the training range.

**Drops** (`eval_standup.py --robots 512 --speeds 1 3 5`). 256 robots dropped from 0.5 m on
their back and 256 on their front, on flat ground. After the hand-over they're told to walk
at 1 m/s:

| start | HoST success | handed to the legs in time (5.6 s) | ... by the end (8.6 s) | median time to the hand-over | walks on, no new fall |
|---|---|---|---|---|---|
| back | 100% | 100% | 100% | 2.0 s | 100% |
| front | 91% | 89% | 95% | 4.5 s | 95% |

From the back this matches HoST's own 99.3%, so the wiring is right. (Our time includes the
drop and the 0.4 s upright hold.) From the front it does better than HoST's 69%, whose prone
start uses a different pose and height. Every robot that was handed over walked on without
falling again.

**Trips on rocks.** Robots run along a rocky lane (25 rocks over 15 × 5 m, as in
`eval_terrain.py`) for 12 s. Every fall goes to the stand-up policy, and a robot that gets up
runs on at the same speed. "Back / front / side" is how the robot lies when its 0.6 s window
ends. Rates are over falls that ended (up, or given up) before the 12 s were over; 25-43% of
robots were still down at the end.

| speed | robots that fell | falls: back / front / side | got up in time: back / front / side | get-up time: back / front / side | fell again within 2 s of getting up |
|---|---|---|---|---|---|
| 1 m/s | 76% | 22 / 178 / 96 | 91% / 63% / 72% | 2.7 / 4.3 / 3.2 s | 3% of falls |
| 3 m/s | 89% | 40 / 324 / 151 | 98% / 72% / 80% | 2.5 / 4.1 / 3.1 s | 20% |
| 5 m/s | 88% | 101 / 300 / 184 | 97% / 85% / 95% | 2.6 / 4.1 / 3.1 s | 25% |

- **Tripping throws the robots forward.** 51-63% of finished falls end face down, and only
  7-17% on the back, which is the only start the stand-up policy was trained on.
- **Face down among rocks is the weak case:** 63-85% get up in time, against 89% face down on
  flat ground and 91-98% on the back.
- **Robots trip again right after getting up.** At 3-5 m/s, a fifth to a quarter of falls are
  followed by another within 2 s of the hand-over: the robot sets off at full speed into the
  same rocks. At 1 m/s it's 3%.
- We haven't looked into why trips at 1 m/s get up less often than at 5 m/s.

**The game**, 1024 games per row, on the terrain arena, policies' mean actions. With stand-up,
"runner fell" means the runner gave up: it was still down 5 s after its window. "Any way" adds
up all the ways each side wins. The last two columns are over every game played: each robot's
falls per game, and the share of falls it got up from in time.

20 s games (the default):

| chaser vs runner | stand-up | caught | time up | runner fell | chaser fell | runner out | chaser wins, any way | runner wins, any way | falls per game (runner, chaser) | got up in time |
|---|---|---|---|---|---|---|---|---|---|---|
| `pursue` vs `flee` | no | 10.6 | 1.1 | 48.1 | 40.1 | 0.0 | 58.7 | 41.2 | | |
| | yes | 33.1 | 30.4 | 16.9 | 18.0 | 0.5 | 50.5 | 49.6 | 0.89, 0.85 | 74% in 3.6 s |
| learned vs `flee` | no | 30.8 | 0.5 | 39.7 | 28.5 | 0.0 | 70.5 | 29.4 | | |
| | yes | 67.9 | 10.8 | 9.7 | 9.6 | 0.7 | 78.3 | 21.8 | 0.66, 0.61 | 79% in 3.4 s |
| `pursue` vs learned | no | 1.1 | 2.1 | 47.6 | 43.5 | 5.7 | 54.4 | 45.6 | | |
| | yes | 7.0 | 49.9 | 14.6 | 19.0 | 9.1 | 30.7 | 69.3 | 1.00, 0.94 | 78% in 3.4 s |
| learned vs learned | no | 10.5 | 1.0 | 48.7 | 31.0 | 8.6 | 67.8 | 32.1 | | |
| | yes | 48.4 | 20.0 | 7.1 | 10.8 | 13.1 | 68.6 | 31.3 | 0.83, 0.78 | 85% in 3.3 s |

60 s games:

| chaser vs runner | stand-up | caught | time up | runner fell | chaser fell | runner out | chaser wins, any way | runner wins, any way | falls per game (runner, chaser) | got up in time |
|---|---|---|---|---|---|---|---|---|---|---|
| `pursue` vs `flee` | no | 11.0 | 0.0 | 51.1 | 37.9 | 0.0 | 62.1 | 37.9 | | |
| | yes | 45.5 | 2.5 | 21.9 | 26.9 | 1.5 | 68.9 | 31.2 | 1.19, 1.12 | 73% in 3.7 s |
| learned vs `flee` | no | 31.4 | 0.0 | 41.3 | 26.5 | 0.1 | 72.8 | 27.2 | | |
| | yes | 72.0 | 0.1 | 11.1 | 14.9 | 1.2 | 84.3 | 15.7 | 0.77, 0.75 | 78% in 3.4 s |
| `pursue` vs learned | no | 1.8 | 0.0 | 47.0 | 44.5 | 6.5 | 55.3 | 44.5 | | |
| | yes | 14.0 | 7.5 | 23.4 | 36.3 | 17.7 | 55.1 | 44.9 | 1.48, 1.39 | 75% in 3.5 s |
| learned vs learned | no | 10.8 | 0.0 | 47.7 | 28.6 | 12.9 | 71.4 | 28.6 | | |
| | yes | 52.0 | 0.1 | 9.6 | 14.0 | 23.9 | 85.5 | 14.6 | 0.95, 0.87 | 83% in 3.3 s |

Draws and "chaser out" (at most 1.8%) are left out. Each figure is within about ±3 points (95%).

- **Without stand-up nothing changed.** The rows without stand-up match the 1024-game table of
  the previous section within ±3 points, so the old code path behaves as before.
- **Falls stop deciding the game.** At 20 s, games ending in a fall drop from 68-91% to 18-35%.
  Catches (7-68%) and time-ups (11-50%) take their place.
- **Skill shows again.** Against `flee`, the learned chaser now wins 78% (any way) where
  `pursue` wins 51%: a 28-point edge, up from 12. Against `pursue`, the learned runner cuts the
  chaser's wins from 51% to 31%: a 20-point edge, up from 4. On the flat arena the edges were
  larger still (88% vs 28% caught, 78% vs 68% time up). The rocks still cost every robot 0.6-1.0
  falls per 20 s game, and about one fall in five ends the game.
- **Who wins between the two learned roles hardly moves** (68% vs 69% for the chaser at 20 s):
  both sides trip alike.

**20 s or 60 s?** (The question that started this.)
- **Without stand-up, game length doesn't matter.** Almost no game reaches 20 s (time-ups at
  most 2.1%, and 0% at 60 s). The runner's wins, any way, move by 1-4 points.
- **With stand-up, a longer game is much harder for the runner.** Time-ups drop from 11-50% to
  0.1-7.5%, and the runner's wins fall by 6-24 points. That's mostly the rules: the chaser needs
  one catch at any moment, while the runner has to last three times as long, falling 0.8-1.5
  times per game.
- **Training on 20 s games adds a little on top.** The scripted runner `flee`, which ignores
  time, loses 18 points against `pursue` going from 20 to 60 s. The learned runner loses 24.
  Its extra losses show up as leaving the arena (9% → 18%; against the learned chaser, 13% →
  24%). Its time-left input still runs from 1 to 0, just three times slower, so it's in range,
  but its timing is off.
- So compare policies at the length they were trained on, and train with `--game_s 60` (and a
  larger γ) if long games are the goal.

**Commands:**

```bash
uv run python eval_standup.py --robots 512 --speeds 1 3 5
uv run python eval_chase.py --checkpoint runs/chase/model_499.pt --terrain --num_games 1024 --standup runs/standup/model_11000.pt   # add --runner flee / --chaser pursue / --game_s 60
uv run python eval_chase.py --checkpoint runs/chase/model_499.pt --terrain --standup runs/standup/model_11000.pt --video videos/standup_chaser --game_s 40
```

**Next ideas** (none run yet):
- Retrain the game policy with `--standup --terrain`. It may also need to know when its
  opponent is down: the actor sees only its own tilt.
- Train the stand-up policy on the falls that actually happen: face down and on the side,
  among rocks. HoST's port already has a prone task.
- After a hand-over, run slowly for a moment, or steer away from the rocks, to cut the refalls.

## Speed cap (`--max_speed`)

**Why.** The game lets both robots command up to 5 m/s forward, but a real G1 walks at about
2 m/s, and the runner lost 83-85% of games. `--max_speed V` clamps the forward command of both
robots after the policy's output is mapped to a command (a game rule, like the terrain cap), so
`chase/model_499` runs unchanged. The scripted players use `V` as their top speed.

**Result.** 128 games, 20 m arena, 20 s, `chase/model_499`, no retraining. Win rates count a
fall or leaving the arena as a loss for that side. With `--standup`
(`runs/standup/model_11000.pt`):

| setup | max_speed | chaser wins | runner wins | runner leaves arena | mean catch time |
|---|---|---|---|---|---|
| policy vs policy | 5 | 83% | 17% | 43.8% | 8.5 s |
| policy vs policy | 2 | 57% | 43% | 19.5% | 11.4 s |
| `flee` vs `pursue` | 5 | 18% | 82% | 0% | 7.1 s |
| `flee` vs `pursue` | 2 | 30% | 70% | 0% | 9.5 s |

Without `--standup` the numbers are within a few points (policy vs policy: runner wins 15% at
5 m/s and 41% at 2 m/s), because falls are rare (0-0.08 per game).

- Leaving the arena, not falling, is the runner's main loss: the legs turn at about 0.2 rad/s,
  so a 5 m/s turning circle (about 27 m) doesn't fit in the arena. At 2 m/s it is about 9 m.
- The learned runner (43%) is still weaker than the scripted `flee` (70%) at 2 m/s. The policy was
  trained with the 5 m/s range, so it hasn't adapted to the cap.

**Commands:**

```bash
uv run python eval_chase.py --checkpoint runs/chase/model_499.pt --standup runs/standup/model_11000.pt --max_speed 2 --num_games 128
uv run python eval_chase.py --runner flee --chaser pursue --standup runs/standup/model_11000.pt --max_speed 2 --num_games 128
```

**Next idea** (not run yet): fine-tune with `train_chase.py --resume runs/chase/model_499.pt
--max_speed 2 --standup runs/standup/model_11000.pt`, so the policy trains under the rules it
is evaluated with.
