"""Thin adapter: use mjlab only as the simulator, with the interface ppo.py expects."""

from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.wrappers import VideoRecorder


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


class MjlabEnv:
  def __init__(
    self, task, num_envs, device, play=False, video_dir=None, video_steps=1000,
    stage_iters=None, steps_per_iter=24,
  ):
    # play=True: evaluation version of the task (no pushes, no sensor noise, no time limit).
    cfg = load_env_cfg(task, play=play)
    cfg.scene.num_envs = num_envs
    if stage_iters is not None:
      set_speed_stages(cfg, stage_iters, steps_per_iter)
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

  def reset(self):
    obs, _ = self.env.reset()
    return obs["actor"], obs["critic"]

  def step(self, actions):
    obs, reward, terminated, truncated, extras = self.env.step(actions)
    return obs["actor"], obs["critic"], reward, terminated, truncated, extras.get("log", {})

  def close(self):
    self.env.close()
