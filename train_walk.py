"""Train the G1 to walk with our own PPO (ppo.py), using mjlab only as the simulator."""

import argparse
import json
import os
import time

import torch

from mjlab_env import MjlabEnv
from ppo import PPOConfig, train


def speed_stage(text):
  """Parse "300:5" into (300, 5.0): from iteration 300, forward commands up to 5 m/s."""
  it, vmax = text.split(":")
  return int(it), float(vmax)


p = argparse.ArgumentParser()
p.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-G1")
p.add_argument("--num_envs", type=int, default=4096)
p.add_argument("--iterations", type=int, default=1500)
p.add_argument("--log_dir", default="runs/g1_walk")
p.add_argument("--resume", default=None, help="checkpoint to continue from")
p.add_argument("--num_steps", type=int, default=PPOConfig.num_steps, help="steps per robot between updates")
p.add_argument("--num_minibatches", type=int, default=PPOConfig.num_minibatches)
# Speed curriculum, counted from the start of this run (also after --resume).
stages = p.add_mutually_exclusive_group()
stages.add_argument("--stage_iters", type=int, nargs=2, metavar=("A", "B"), default=None,
                    help="commands up to 2 m/s from iteration A and 3 m/s from iteration B "
                         "(mjlab: 5000 and 10000)")
stages.add_argument("--speed_stages", type=speed_stage, nargs="+", metavar="ITER:VMAX", default=None,
                    help="commands up to VMAX m/s from iteration ITER, e.g. 0:4 300:5 (first ITER must be 0)")
# Reward weights. Every mjlab reward is multiplied by step_dt = 0.02 s.
p.add_argument("--air_time", type=float, default=0.0, help="weight of the feet air-time reward")
p.add_argument("--action_rate", type=float, default=None, help="weight of action_rate_l2 (mjlab: -0.1)")
p.add_argument("--fell_over", type=float, default=None,
               help="weight of an extra penalty when a robot falls, e.g. -200 (= -4 per fall)")
p.add_argument("--pose_running_scale", type=float, default=None,
               help="multiply the running pose std of hip_pitch, knees and arms, e.g. 1.5")
p.add_argument("--speed_shortfall", type=float, default=None,
               help="weight of a penalty per m/s slower than the command, e.g. -1 "
                    "(stops the policy from standing still when told to run fast)")
args = p.parse_args()
for flag in ("fell_over", "speed_shortfall"):
  if getattr(args, flag) is not None and getattr(args, flag) > 0:
    p.error(f"--{flag} is a penalty weight and must be <= 0")

# Checkpoints don't store these flags, so keep one line per run (resumes included).
os.makedirs(args.log_dir, exist_ok=True)
with open(os.path.join(args.log_dir, "args.jsonl"), "a") as f:
  f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **vars(args)}) + "\n")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)
cfg = PPOConfig(num_steps=args.num_steps, num_minibatches=args.num_minibatches)
env = MjlabEnv(
  args.task, args.num_envs, device,
  stage_iters=args.stage_iters, speed_stages=args.speed_stages, steps_per_iter=cfg.num_steps,
  rewards=dict(
    air_time=args.air_time, action_rate=args.action_rate,
    fell_over_weight=args.fell_over, pose_running_scale=args.pose_running_scale,
    speed_shortfall_weight=args.speed_shortfall,
  ),
)
print_keys = (
  "Episode_Termination/fell_over",
  "Episode_Reward/track_linear_velocity",
  "Curriculum/command_vel/lin_vel_x_max",
)
train(
  env,
  iterations=args.iterations,
  cfg=cfg,
  device=device,
  log_dir=args.log_dir,
  resume=args.resume,
  log_keys=print_keys + tuple(f"Episode_Reward/{name}" for name in env.reward_terms),
  print_keys=print_keys,
)
env.close()
