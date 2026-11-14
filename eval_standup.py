"""The stand-up policy (standup.py) with the legs, without a game: do fallen robots get up and walk on?

drop: robots are dropped from 0.5 m lying on their back or front, on flat ground, the way the
      stand-up policy was trained. It reports HoST's success (the pelvis gets above 0.7 m within
      5 s after the 0.6 s window and doesn't drop below 0.5 m in that time), the hand-over to
      the legs and the get-up time. Then the legs are told to walk at 1 m/s: "walks on" = no
      new fall before the end (at least 2 s of walking).
trip: robots run along a rocky lane (as in eval_terrain.py) at each of --speeds for 12 s. Every
      fall goes to the stand-up policy, and a robot that gets up walks on at the same speed.
"""

import argparse
import math

import torch
import warp as wp

import standup
from mjlab_env import MjlabEnv
from ppo import load_actor_critic
from standup import POSTURES, STANDUP, STEADY, WALK, WINDOW, Recovery
from terrain import CELLS, FIELDS, ROCKY, TerrainArena

TASK = "Mjlab-Velocity-Flat-Unitree-G1"

p = argparse.ArgumentParser()
p.add_argument("--standup", default="runs/standup/model_11000.pt", help="stand-up checkpoint (HoST, mjlab port)")
p.add_argument("--legs", default="runs/g1_turn/model_4349.pt", help="locomotion checkpoint")
p.add_argument("--robots", type=int, default=256, help="robots per test")
p.add_argument("--speeds", type=float, nargs="+", default=[3.0, 5.0], help="trip test speeds (m/s)")
p.add_argument("--standup_s", type=float, default=5.0, help="time to get up after the window (s)")
args = p.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
n = args.robots
arena = TerrainArena(20.0, device)
mj = MjlabEnv(TASK, n, device, play=True, external_commands=True, cfg_fn=standup.configure,
              spec_fn=arena.spec_fn, expand_fields=FIELDS + ("actuator_biasprm",))
arena.bind(mj)
recovery = Recovery(mj, args.standup, device, standup_s=args.standup_s)
legs, _ = load_actor_critic(args.legs, device)
legs.eval()
worlds = torch.arange(n, device=device)
arena.set_cells(worlds, torch.full((n, CELLS * CELLS), ROCKY, device=device))  # same physics as normal
dt = mj.step_dt
overflow = [0]  # most worlds whose contact or constraint buffers overflowed in one step (should stay 0)


def start(xy, lying=None):
  """Respawn every robot at xy facing +x, standing, or lying (bool per robot: True = on its back)."""
  arena.place_rocks(worlds, torch.full((n, arena.num_rocks, 2), float("nan"), device=device),
                    torch.zeros(n, arena.num_rocks, device=device))  # park all rocks
  obs = mj.place_robots(worlds, xy, torch.zeros(n, device=device))
  if lying is not None:  # pitch the pelvis by -90° (back) or +90° (front), 0.5 m up
    s = math.sqrt(0.5)
    pose = torch.zeros(n, 7, device=device)
    pose[:, :2], pose[:, 2], pose[:, 3] = xy, 0.5, s
    pose[:, 5] = torch.where(lying, -s, s)
    mj.robot.write_root_link_pose_to_sim(pose, env_ids=worlds)
    mj.env.unwrapped.sim.forward()
    obs = mj.actor_obs()
  recovery.reset(worlds)
  recovery.reset_stats()
  return obs


def run(obs, speed, seconds, watch=None):
  """Step everyone for `seconds`, told to go at `speed` while walking (0 while down).
  watch(t) is called after every step. Returns the number of falls of each robot."""
  with torch.no_grad():
    for step in range(round(seconds / dt)):
      arena.step_effects()
      cmd = torch.zeros(n, 3, device=device)
      cmd[:, 0] = torch.where(recovery.mode == WALK, speed, 0.0)
      obs = mj.set_command(cmd)
      obs, _, _, _, _, _ = mj.step(recovery.act(legs.act_deterministic(obs)))
      recovery.update()
      overflow[0] = max(overflow[0], int((wp.to_torch(mj.env.unwrapped.sim.wp_data.overflow) != 0).sum()))
      if watch:
        watch((step + 1) * dt)
  return recovery.robot_falls


def stats_line():
  r = recovery
  done = r.stood + r.gave_up_count
  parts = []
  for k, name in enumerate(POSTURES):
    if done[k] > 0:
      parts.append(f"{name} {int(done[k])}: up {r.stood[k] / done[k]:4.0%} in {r.stand_s[k] / r.stood[k].clamp(min=1):.1f} s")
  total = done.sum().clamp(min=1)
  return (f"falls {r.falls} | up in time {r.stood.sum() / total:.0%}, gave up {r.gave_up_count.sum() / total:.0%}"
          f" | fell again within 2 s of getting up: {r.refalls} | " + " | ".join(parts))


print(f"stand-up: {args.standup} | legs: {args.legs} | {n} robots per test | "
      f"time to get up: 0.6 s window + {args.standup_s:.0f} s")

# ---- drop ----
on_back = torch.arange(n, device=device) % 2 == 0
y = (torch.arange(n, device=device) % 16 - 7.5) * 1.0  # robots don't touch (own worlds), spread anyway
obs = start(torch.stack([torch.zeros(n, device=device), y], dim=1), lying=on_back)
window_s, host_s = (WINDOW + 1) * dt, 5.0
rose_at = torch.full((n,), float("nan"), device=device)
dropped = torch.zeros(n, dtype=torch.bool, device=device)
handed_at = torch.full((n,), float("nan"), device=device)


def watch_drop(t):
  z = mj.robot.data.root_link_pos_w[:, 2]
  if window_s < t <= window_s + host_s:  # HoST's success: rise above 0.7 m, then stay above 0.5 m
    rose_at[torch.isnan(rose_at) & (z > 0.7)] = t
    dropped[~torch.isnan(rose_at) & (z < 0.5)] = True
  handed_at[torch.isnan(handed_at) & (recovery.mode == STEADY)] = t


falls = run(obs, 1.0, window_s + args.standup_s + 0.5 + 2.0 + 0.5, watch_drop)
print("\ndrop from 0.5 m, then walk at 1 m/s")
for name, group in (("back", on_back), ("front", ~on_back)):
  host = (~torch.isnan(rose_at) & ~dropped)[group].float().mean()
  handed = ~torch.isnan(handed_at[group])
  on = handed & (falls[group] == 1)
  t = handed_at[group][handed]
  print(f"  {name:5s} | HoST success {host:4.0%} | handed to the legs {handed.float().mean():4.0%}"
        f" (median {t.median().item() if len(t) else float('nan'):.1f} s after the drop)"
        f" | walks on {on.float().mean():4.0%}")
print("  " + stats_line())

# ---- trip ----
print("\ntrip: run a rocky lane (25 rocks over 15 x 5 m) for 12 s")
for speed in args.speeds:
  y = (torch.rand(n, device=device) * 2 - 1) * 6.0
  obs = start(torch.stack([torch.full_like(y, -8.0), y], dim=1))
  arena.place_lane_rocks(worlds, y)
  falls = run(obs, speed, 12.0)
  print(f"  {speed:.0f} m/s | robots that fell {(falls > 0).float().mean():4.0%}"
        f" | still down at the end {(recovery.mode == STANDUP).float().mean():4.0%} | " + stats_line())
print("(postures: how a robot lies when its 0.6 s window ends; up = handed back to the legs within the time limit)")
print(f"MuJoCo Warp buffer overflow: at most {overflow[0]} worlds in one step")
mj.close()
