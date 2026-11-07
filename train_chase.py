"""Train the runner and the chaser (one shared network) with our PPO, on top of the frozen sprint legs."""

import argparse
import json
import os
import time

import torch

from chase_env import OUTCOMES, ChaseEnv, add_game_args, game_kwargs
from ppo import PPOConfig, train

p = argparse.ArgumentParser()
p.add_argument("--num_games", type=int, default=2048, help="games played at once (2 robots each)")
p.add_argument("--iterations", type=int, default=500)
p.add_argument("--log_dir", default="runs/chase")
p.add_argument("--resume", default=None, help="game-policy checkpoint to continue from")
add_game_args(p)
args = p.parse_args()

# Checkpoints don't store these flags (or which legs were used), so keep one line per run.
os.makedirs(args.log_dir, exist_ok=True)
with open(os.path.join(args.log_dir, "args.jsonl"), "a") as f:
  f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **vars(args)}) + "\n")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)
env = ChaseEnv(args.num_games, device, **game_kwargs(args))
# A game is 200 decisions (20 s at 10 Hz), so look further ahead than for walking.
# The inputs are ~25 numbers, so a smaller network is plenty.
cfg = PPOConfig(gamma=0.995, hidden=(256, 128), init_std=0.5)
print_keys = ("Game/chaser_win", "Game/runner_win", "Game/runner_fell", "Game/chaser_fell",
              "Game/runner_out", "Game/chaser_out")
train(
  env,
  iterations=args.iterations,
  cfg=cfg,
  device=device,
  log_dir=args.log_dir,
  resume=args.resume,
  log_keys=tuple(f"Game/{name}" for name in OUTCOMES) + ("Game/length_s",),
  print_keys=print_keys,
)
env.close()
