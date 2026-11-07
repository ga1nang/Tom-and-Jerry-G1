"""Play runner vs chaser games: learned policy and/or scripted players. Prints who wins and how."""

import argparse
import glob
import os

import torch

from chase_env import CHASER_WIN, OUTCOMES, ChaseEnv, add_game_args, game_kwargs
from ppo import load_actor_critic

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default=None, help="game policy (not needed if both sides are scripted)")
p.add_argument("--runner", choices=("policy", "flee"), default="policy")
p.add_argument("--chaser", choices=("policy", "pursue"), default="policy")
p.add_argument("--num_games", type=int, default=256)
p.add_argument("--sample", action="store_true", help="sample actions instead of using the mean")
p.add_argument("--video", default=None, help="folder for an mp4 of one game (overhead camera)")
add_game_args(p)
args = p.parse_args()
if "policy" in (args.runner, args.chaser) and not args.checkpoint:
  p.error("--checkpoint is needed when a side is played by the policy")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
G = 1 if args.video else args.num_games
env = ChaseEnv(G, device, video_dir=args.video, **game_kwargs(args))
ac = None
if args.checkpoint:
  ac, _ = load_actor_critic(args.checkpoint, device)
  ac.eval()

actor_obs, _ = env.reset()
first = torch.full((G,), -1, dtype=torch.long, device=device)  # result of each slot's first game
first_len = torch.zeros(G, device=device)
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
    actor_obs, _, _, _, _, _ = env.step(actions)
    new = (first < 0) & (env.outcome >= 0)
    first[new] = env.outcome[new]
    first_len[new] = env.length_s[new]
    step += 1

done = first >= 0
counts = torch.bincount(first[done], minlength=len(OUTCOMES)).float()
n = int(done.sum())
print(f"runner: {args.runner} | chaser: {args.chaser} | games: {n} | arena {args.arena:.0f} m")
for k, name in enumerate(OUTCOMES):
  print(f"  {name:11s} {counts[k].item() / max(n, 1):6.1%}")
caught = first == CHASER_WIN
if caught.any():
  print(f"  mean catch time: {first_len[caught].mean().item():.1f} s")
env.close()
if args.video:
  vids = sorted(glob.glob(os.path.join(args.video, "**", "*.mp4"), recursive=True), key=os.path.getmtime)
  print("video:", vids[-1] if vids else "none written")
