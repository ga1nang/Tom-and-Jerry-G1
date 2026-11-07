"""Run a policy trained with ppo.py: count falls and optionally record a video."""

import argparse
import glob
import os

import torch

from mjlab_env import MjlabEnv
from ppo import load_actor_critic

p = argparse.ArgumentParser()
p.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-G1")
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seconds", type=float, default=20.0)
p.add_argument("--video", default=None, help="folder for an mp4 (uses 1 robot)")
args = p.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ac, _ = load_actor_critic(args.checkpoint, device)
ac.eval()

num_envs = 1 if args.video else args.num_envs
env = MjlabEnv(args.task, num_envs, device, play=True, video_dir=args.video,
               video_steps=int(args.seconds / 0.02))
steps = int(args.seconds / env.step_dt)

actor_obs, _ = env.reset()
falls, reward_sum = 0, 0.0
with torch.no_grad():
  for _ in range(steps):
    action = ac.act_deterministic(actor_obs)  # mean action, no exploration noise
    actor_obs, _, reward, terminated, truncated, _ = env.step(action)
    falls += int(terminated.sum().item())  # play mode has no time limit, so this counts falls
    reward_sum += reward.mean().item()

robot_minutes = num_envs * args.seconds / 60
print(f"robots: {num_envs} | simulated: {args.seconds:.0f} s each")
print(f"falls: {falls}  ({falls / robot_minutes:.2f} per robot-minute)")
print(f"mean reward per step: {reward_sum / steps:.3f}")
env.close()
if args.video:
  vids = sorted(glob.glob(os.path.join(args.video, "**", "*.mp4"), recursive=True), key=os.path.getmtime)
  print("video:", vids[-1] if vids else "none written")
