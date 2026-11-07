"""Thin adapter: use mjlab only as the simulator, with the interface ppo.py expects."""

from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.wrappers import VideoRecorder


class MjlabEnv:
  def __init__(self, task, num_envs, device, play=False, video_dir=None, video_steps=1000):
    # play=True: evaluation version of the task (no pushes, no sensor noise, no time limit).
    cfg = load_env_cfg(task, play=play)
    cfg.scene.num_envs = num_envs
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
