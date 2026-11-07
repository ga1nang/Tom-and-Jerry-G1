"""Thin adapter: use mjlab only as the simulator, with the interface ppo.py expects."""

from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.wrappers import VideoRecorder

# Joints that move the most when running: longer leg strides, bigger arm swing.
RUNNING_JOINTS = ("hip_pitch", "knee", "shoulder", "elbow")


def fell_over(env):
  """1 for robots that the "fell_over" termination ended this step, else 0."""
  # Safe to read inside a reward: mjlab computes terminations before rewards.
  return env.termination_manager.get_term("fell_over").float()


def set_speed_stages(cfg, stage_iters, steps_per_iter):
  """Speed curriculum: commands up to 2 m/s from iteration A, up to 3 m/s from iteration B.

  Same stages as mjlab, which starts them at iteration 5000 and 10000. mjlab counts
  env steps since this process started, so a resumed run starts again from stage 0.
  A stage takes effect at the next robot reset after it is reached.
  """
  a, b = stage_iters
  if not 0 <= a <= b:
    raise ValueError(f"stage_iters needs 0 <= A <= B, got {stage_iters}")
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
    {"step": a * steps_per_iter, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
    {"step": b * steps_per_iter, "lin_vel_x": (-2.0, 3.0)},
  ]


def set_fixed_speed(cfg, speed):
  """Every robot is told the same thing: go straight ahead at `speed` m/s."""
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (speed, speed)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_standing_envs = 0.0
  twist.rel_forward_envs = 0.0  # these robots get abs(speed) clamped to >= 0.3
  twist.rel_world_envs = 0.0
  # Heading robots turn toward a random heading by overwriting ang_vel_z.
  twist.heading_command = False
  twist.ranges.heading = None  # mjlab rejects a heading range without heading_command
  cfg.curriculum.pop("command_vel", None)  # it would overwrite the ranges above


def override_rewards(cfg, air_time=None, action_rate=None, fell_over_weight=None, pose_running_scale=None):
  """Change reward terms before the env is built. None keeps mjlab's setting.

  Weights are multiplied by step_dt (0.02 s), like every mjlab reward:
  fell_over_weight=-200 costs 4 per fall.
  """
  if air_time is not None:
    cfg.rewards["air_time"].weight = air_time  # rewards each foot in the air for 0.05-0.5 s
  if action_rate is not None:
    cfg.rewards["action_rate_l2"].weight = action_rate
  if fell_over_weight is not None:
    cfg.rewards["fall_penalty"] = RewardTermCfg(func=fell_over, weight=fell_over_weight)
  if pose_running_scale is not None:
    # The pose reward uses std_running once |command| >= 1.5 (larger std = more freedom).
    std = cfg.rewards["pose"].params["std_running"]
    for pattern in std:
      if any(joint in pattern for joint in RUNNING_JOINTS):
        std[pattern] *= pose_running_scale


class MjlabEnv:
  def __init__(
    self, task, num_envs, device, play=False, video_dir=None, video_steps=1000,
    stage_iters=None, steps_per_iter=24, fixed_speed=None, rewards=None,
  ):
    # play=True: evaluation version of the task (no pushes, no sensor noise, no time limit).
    cfg = load_env_cfg(task, play=play)
    cfg.scene.num_envs = num_envs
    if stage_iters is not None:
      set_speed_stages(cfg, stage_iters, steps_per_iter)
    if fixed_speed is not None:
      set_fixed_speed(cfg, fixed_speed)
    if rewards:
      override_rewards(cfg, **rewards)
    if video_dir:
      cfg.viewer.width, cfg.viewer.height = 640, 480
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if video_dir else None)
    if video_dir:
      self.env = VideoRecorder(
        self.env, video_folder=Path(video_dir), step_trigger=lambda s: s == 0,
        video_length=video_steps, disable_logger=True,
      )
    self.num_envs = num_envs
    self.num_actions = self.env.unwrapped.action_manager.total_action_dim
    self.step_dt = self.env.unwrapped.step_dt

  def forward_speed(self):
    """Commanded and actual forward speed (m/s, pelvis frame) of every robot."""
    env = self.env.unwrapped
    commanded = env.command_manager.get_command("twist")[:, 0]
    actual = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return commanded, actual

  def time_since_reset(self):
    """Seconds since each robot last (re)spawned."""
    return self.env.unwrapped.episode_length_buf * self.step_dt

  def reset(self):
    obs, _ = self.env.reset()
    return obs["actor"], obs["critic"]

  def step(self, actions):
    obs, reward, terminated, truncated, extras = self.env.step(actions)
    return obs["actor"], obs["critic"], reward, terminated, truncated, extras.get("log", {})

  def close(self):
    self.env.close()
