"""Train the G1 to walk with our own PPO (ppo.py), using mjlab only as the simulator."""

import argparse

import torch

from mjlab_env import MjlabEnv
from ppo import PPOConfig, train

p = argparse.ArgumentParser()
p.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-G1")
p.add_argument("--num_envs", type=int, default=4096)
p.add_argument("--iterations", type=int, default=1500)
p.add_argument("--log_dir", default="runs/g1_walk")
p.add_argument("--resume", default=None, help="checkpoint to continue from")
args = p.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)
env = MjlabEnv(args.task, args.num_envs, device)
train(
  env,
  iterations=args.iterations,
  cfg=PPOConfig(),
  device=device,
  log_dir=args.log_dir,
  resume=args.resume,
  log_keys=("Episode_Termination/fell_over", "Episode_Reward/track_linear_velocity"),
)
env.close()
