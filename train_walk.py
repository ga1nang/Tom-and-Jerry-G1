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
p.add_argument("--stage_iters", type=int, nargs=2, metavar=("A", "B"), default=None,
               help="commands up to 2 m/s from iteration A and 3 m/s from iteration B of this run "
                    "(mjlab: 5000 and 10000)")
args = p.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)
cfg = PPOConfig()
env = MjlabEnv(
  args.task, args.num_envs, device,
  stage_iters=args.stage_iters, steps_per_iter=cfg.num_steps,
)
train(
  env,
  iterations=args.iterations,
  cfg=cfg,
  device=device,
  log_dir=args.log_dir,
  resume=args.resume,
  log_keys=("Episode_Termination/fell_over", "Episode_Reward/track_linear_velocity",
            "Curriculum/command_vel/lin_vel_x_max"),
)
env.close()
