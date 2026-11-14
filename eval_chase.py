"""Play runner vs chaser games: learned policy and/or scripted players. Prints who wins and how."""

import argparse
import glob
import os

import torch

from chase_env import CHASER_FELL, CHASER_WIN, OUTCOMES, RUNNER_FELL, ChaseEnv, add_game_args, game_kwargs
from ppo import load_actor_critic
from terrain import TERRAINS

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default=None, help="game policy (not needed if both sides are scripted)")
p.add_argument("--runner", choices=("policy", "flee"), default="policy")
p.add_argument("--chaser", choices=("policy", "pursue"), default="policy")
p.add_argument("--num_games", type=int, default=256)
p.add_argument("--sample", action="store_true", help="sample actions instead of using the mean")
p.add_argument("--video", default=None, help="folder for an mp4 of one game (overhead camera)")
p.add_argument("--camera", choices=("chaser", "overhead"), default="chaser",
               help="video view: behind the chaser looking at the runner, or above the whole arena")
add_game_args(p)
args = p.parse_args()
if "policy" in (args.runner, args.chaser) and not args.checkpoint:
  p.error("--checkpoint is needed when a side is played by the policy")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
G = 1 if args.video else args.num_games
env = ChaseEnv(G, device, video_dir=args.video, camera=args.camera, **game_kwargs(args))
ac = None
if args.checkpoint:
  ac, _ = load_actor_critic(args.checkpoint, device)
  ac.eval()

actor_obs, _ = env.reset()
first = torch.full((G,), -1, dtype=torch.long, device=device)  # result of each slot's first game
first_len = torch.zeros(G, device=device)
first_falls = torch.zeros(G, 2, device=device)  # --standup: falls of the runner and the chaser in each first game
# --terrain: decisions each role spent on each terrain type, and where robots fell (first games only).
time_on = torch.zeros(2, len(TERRAINS), device=device)
fell_on = torch.zeros(len(TERRAINS), device=device)
# Video: play for game_s seconds whatever happens. Otherwise: until every slot finished a game.
decisions = round(args.game_s / 0.1)
step = 0
with torch.no_grad():
  while (step < decisions) if args.video else (first < 0).any():
    actions = torch.zeros(2 * G, 3, device=device)
    if ac is not None:
      actions = ac.dist(actor_obs).sample() if args.sample else ac.act_deterministic(actor_obs)
    if args.runner == "flee":
      actions[:G] = env.flee()
    if args.chaser == "pursue":
      actions[G:] = env.pursue()
    playing = first < 0
    actor_obs, _, _, _, _, _ = env.step(actions)
    new = (first < 0) & (env.outcome >= 0)
    if env.terrain is not None:  # terrain_under: the ground under each robot when this decision started
      for role, under in enumerate((env.terrain_under[:G], env.terrain_under[G:])):
        time_on[role] += torch.bincount(under[playing], minlength=len(TERRAINS))
      fell_on += torch.bincount(env.terrain_under[:G][new & (env.outcome == RUNNER_FELL)], minlength=len(TERRAINS))
      fell_on += torch.bincount(env.terrain_under[G:][new & (env.outcome == CHASER_FELL)], minlength=len(TERRAINS))
    first[new] = env.outcome[new]
    first_len[new] = env.length_s[new]
    if env.recovery:
      first_falls[new] = env.falls[new].float()
    step += 1

done = first >= 0
counts = torch.bincount(first[done], minlength=len(OUTCOMES)).float()
n = int(done.sum())
print(f"runner: {args.runner} | chaser: {args.chaser} | games: {n} | arena {args.arena:.0f} m"
      + (" | terrain" if args.terrain else "") + f" | {args.game_s:.0f} s"
      + (f" | stand-up, {args.standup_s:.0f} s to get up" if args.standup else ""))
for k, name in enumerate(OUTCOMES):
  print(f"  {name:11s} {counts[k].item() / max(n, 1):6.1%}")
caught = first == CHASER_WIN
if caught.any():
  print(f"  mean catch time: {first_len[caught].mean().item():.1f} s")
if env.recovery:
  print(f"  falls per game: runner {first_falls[done, 0].mean().item():.2f}, chaser {first_falls[done, 1].mean().item():.2f}")
  r = env.recovery  # these count every game played, not just the first ones
  finished = (r.stood + r.gave_up_count).sum().clamp(min=1)  # falls that ended in getting up or giving up
  print(f"  of all falls: up in time {r.stood.sum() / finished:.0%} (in {r.stand_s.sum() / r.stood.sum().clamp(min=1):.1f} s),"
        f" gave up {r.gave_up_count.sum() / finished:.0%}, fell again within 2 s of getting up {r.refalls / max(r.falls, 1):.0%}")
if args.terrain:
  share = time_on / time_on.sum(dim=1, keepdim=True)
  print("  time on each terrain (runner / chaser): "
        + " | ".join(f"{name} {share[0, k]:.0%} / {share[1, k]:.0%}" for k, name in enumerate(TERRAINS)))
  print("  falls by the terrain the robot was on: "
        + " | ".join(f"{name} {int(fell_on[k])}" for k, name in enumerate(TERRAINS)))
env.close()
if args.video:
  vids = sorted(glob.glob(os.path.join(args.video, "**", "*.mp4"), recursive=True), key=os.path.getmtime)
  print("video:", vids[-1] if vids else "none written")
