"""The legs alone on each terrain type of the game arena: falls, speed, turning, sinking.

Every robot gets a whole arena of one terrain type (terrain.py) and is told to go straight
across it from x = -8 m. On rocky ground, its rocks are scattered along its lane at about
the density of a rocky cell. No game and no speed caps: this tests the physics only.
"""

import argparse

import torch

from mjlab_env import MjlabEnv
from ppo import load_actor_critic
from terrain import CELLS, FIELDS, ROCKY, TERRAINS, TerrainArena

TASK = "Mjlab-Velocity-Flat-Unitree-G1"
WARMUP_S = 1.0  # robots start standing: ignore their speed for the first second
LANE_ROCKS = 25  # rocks in each rocky lane (15 x 5 m): about a rocky cell's 5 per 16 m²

p = argparse.ArgumentParser()
p.add_argument("--legs", default="runs/g1_turn/model_4349.pt", help="locomotion checkpoint")
p.add_argument("--robots", type=int, default=64, help="robots per terrain type")
p.add_argument("--speeds", type=float, nargs="+", default=[1.0, 3.0, 5.0])
p.add_argument("--arena", type=float, default=20.0)
args = p.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
R, K = args.robots, len(TERRAINS)
n = R * K  # robots i*R .. (i+1)*R-1 are on terrain type i
arena = TerrainArena(args.arena, device)
mj = MjlabEnv(TASK, n, device, play=True, external_commands=True,
              spec_fn=arena.spec_fn, expand_fields=FIELDS)
arena.bind(mj)
legs, _ = load_actor_critic(args.legs, device)
legs.eval()
worlds = torch.arange(n, device=device)
types = worlds // R
# Foot collision capsules: their lowest point shows how far the feet sink into the ground.
m = mj.env.unwrapped.sim.mj_model
feet = [i for i in range(m.ngeom) if "foot" in m.geom(i).name and m.geom(i).name.endswith("_collision")]
foot_radius = torch.as_tensor(m.geom_size[feet, 0], dtype=torch.float, device=device)
arena.set_cells(worlds, types[:, None].expand(n, CELLS * CELLS))


def run(speed, turn, seconds):
  """Everyone starts at x = -8 m facing +x; returns per-type fall fraction, speed, yaw rate, foot depth."""
  y = (torch.rand(n, device=device) * 2 - 1) * 6.0
  # Rocks (rocky robots only): along each robot's lane, from 2 m ahead of it to the far side.
  # The arena's other rocks stay parked (NaN).
  rock_xy = torch.full((n, arena.num_rocks, 2), float("nan"), device=device)
  rocky = types == ROCKY
  rock_xy[rocky, :LANE_ROCKS, 0] = torch.empty(int(rocky.sum()), LANE_ROCKS, device=device).uniform_(-6.0, 9.0)
  rock_xy[rocky, :LANE_ROCKS, 1] = y[rocky, None] + torch.empty(int(rocky.sum()), LANE_ROCKS, device=device).uniform_(-2.5, 2.5)
  arena.place_rocks(worlds, rock_xy, torch.rand(n, arena.num_rocks, device=device) * 2 * torch.pi)
  obs = mj.place_robots(worlds, torch.stack([torch.full_like(y, -8.0), y], dim=1), torch.zeros(n, device=device))
  obs = mj.set_command(torch.tensor([speed, 0.0, turn], device=device).repeat(n, 1))

  fell = torch.zeros(n, dtype=torch.bool, device=device)
  speed_sum, yaw_sum, depth_sum, count = (torch.zeros(n, device=device) for _ in range(4))
  steps = round(seconds / mj.step_dt)
  with torch.no_grad():
    for step in range(steps):
      arena.step_effects()
      obs, _, _, terminated, _, _ = mj.step(legs.act_deterministic(obs))
      fell |= terminated  # a fallen robot is respawned elsewhere by mjlab: stop counting it
      if step * mj.step_dt >= WARMUP_S:
        up = (~fell).float()
        data = mj.robot.data
        speed_sum += up * data.root_link_lin_vel_b[:, 0]
        yaw_sum += up * data.root_link_ang_vel_b[:, 2]
        lowest = (mj.env.unwrapped.sim.data.geom_xpos[:, feet, 2] - foot_radius).min(dim=1).values
        depth_sum += up * lowest
        count += up
  mean = lambda x: torch.stack([(x[types == k].sum() / count[types == k].sum().clamp(min=1)) for k in range(K)])
  falls = torch.stack([fell[types == k].float().mean() for k in range(K)])
  return falls, mean(speed_sum), mean(yaw_sum), mean(depth_sum)


# Standing still: how far the feet sink into each terrain (lowest foot point below z = 0).
_, _, _, depth = run(0.0, 0.0, 3.0)
print(f"legs: {args.legs} | {R} robots per terrain | falls = robots that fell during the run")
print("standing, feet sink: " + " | ".join(f"{name} {-100 * depth[k].item():.1f} cm" for k, name in enumerate(TERRAINS)))

header = "".join(f" | {name:>16s}" for name in TERRAINS)
print(f"\n{'straight, commanded':22s}{header}")
for v in args.speeds:
  falls, speed, _, _ = run(v, 0.0, min(8.0, max(3.0, 16.0 / v)))
  cells = "".join(f" | {s.item():4.2f} m/s {f.item():4.0%}" for s, f in zip(speed, falls))
  print(f"{v:4.1f} m/s{'':13s}{cells}")

falls, _, yaw, _ = run(3.0, 0.7, 5.0)
cells = "".join(f" | {w.item():4.2f} r/s {f.item():4.0%}" for w, f in zip(yaw, falls))
print(f"{'3 m/s + turn 0.7 rad/s':22s}{cells}")
print("(each cell: mean achieved speed or turn rate after the first second, then the share of robots that fell)")
mj.close()
