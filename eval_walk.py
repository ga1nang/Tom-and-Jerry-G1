"""Run a policy trained with ppo.py: count falls, measure speed, optionally record a video."""

import argparse
import glob
import os

import torch

from mjlab_env import MjlabEnv
from ppo import load_actor_critic

WARMUP_S = 2.0  # robots (re)spawn standing still: ignore their speed for the first 2 s

p = argparse.ArgumentParser()
p.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-G1")
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seconds", type=float, default=20.0)
p.add_argument("--video", default=None, help="folder for an mp4 (uses 1 robot)")
p.add_argument("--speed", type=float, default=None,
               help="command every robot to go straight at this speed (m/s); default: random commands")
p.add_argument("--turn", type=float, default=None,
               help="with --speed (default 0): also turn at this rate (rad/s), e.g. 0.7")
args = p.parse_args()
if args.turn is not None and args.speed is None:
  args.speed = 0.0

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ac, _ = load_actor_critic(args.checkpoint, device)
ac.eval()

num_envs = 1 if args.video else args.num_envs
env = MjlabEnv(args.task, num_envs, device, play=True, video_dir=args.video,
               video_steps=int(args.seconds / 0.02), fixed_speed=args.speed, fixed_turn=args.turn or 0.0)
steps = int(args.seconds / env.step_dt)

actor_obs, _ = env.reset()
falls, reward_sum = 0, 0.0
cmd_sum, speed_sum, n_speed = 0.0, 0.0, 0
turn_cmd_sum, turn_sum, turn_err_sum = 0.0, 0.0, 0.0
with torch.no_grad():
  for _ in range(steps):
    action = ac.act_deterministic(actor_obs)  # mean action, no exploration noise
    actor_obs, _, reward, terminated, truncated, _ = env.step(action)
    falls += int(terminated.sum().item())  # play mode has no time limit, so this counts falls
    reward_sum += reward.mean().item()
    # Per robot, so a robot that fell and respawned is skipped again while it speeds up.
    commanded, actual = env.forward_speed()
    settled = env.time_since_reset() >= WARMUP_S
    cmd_sum += commanded[settled].sum().item()
    speed_sum += actual[settled].sum().item()
    n_speed += int(settled.sum().item())
    turn_cmd, turn = env.turn_rate()
    turn_cmd_sum += turn_cmd[settled].sum().item()
    turn_sum += turn[settled].sum().item()
    turn_err_sum += (turn_cmd - turn)[settled].abs().sum().item()

robot_minutes = num_envs * args.seconds / 60
print(f"robots: {num_envs} | simulated: {args.seconds:.0f} s each")
print(f"falls: {falls}  ({falls / robot_minutes:.2f} per robot-minute)")
print(f"mean reward per step: {reward_sum / steps:.3f}")
if n_speed:
  print(f"forward speed (skipping first {WARMUP_S:.0f} s after each spawn): "
        f"commanded {cmd_sum / n_speed:.2f} m/s | achieved {speed_sum / n_speed:.2f} m/s")
  print(f"turning: commanded {turn_cmd_sum / n_speed:.2f} rad/s | achieved {turn_sum / n_speed:.2f} rad/s | "
        f"mean error {turn_err_sum / n_speed:.2f} rad/s")
else:
  print(f"forward speed: not measured (no robot stayed up for {WARMUP_S:.0f} s)")
env.close()
if args.video:
  vids = sorted(glob.glob(os.path.join(args.video, "**", "*.mp4"), recursive=True), key=os.path.getmtime)
  print("video:", vids[-1] if vids else "none written")
