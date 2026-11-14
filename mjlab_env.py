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


def speed_shortfall(env):
  """How many m/s slower than commanded each robot moves along the command direction.

  Standing still when told to run at 5 m/s scores 5; running at 4.4 m/s scores 0.6;
  going faster than commanded scores 0. Unlike track_linear_velocity, whose
  exp(-error²/0.25) is ~0 for anything more than ~1 m/s off, this keeps paying for
  every extra m/s, so standing still is never the safe choice.
  """
  command = env.command_manager.get_command("twist")[:, :2]
  velocity = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
  target = command.norm(dim=1)
  progress = (velocity * command).sum(dim=1) / target.clamp(min=1e-6)  # 0 for a zero command
  return (target - progress).clamp(min=0.0)


def yaw_error(env):
  """How far (rad/s) each robot's turning speed is from the commanded one.

  mjlab's track_angular_velocity also counts the torso's roll and pitch wobble, which is
  large when running, so its exp() pays ~nothing at speed whether the robot turns or not.
  This penalty only looks at the yaw rate and grows linearly, so turning always pays.
  """
  command = env.command_manager.get_command("twist")[:, 2]
  return (command - env.scene["robot"].data.root_link_ang_vel_b[:, 2]).abs()


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


def set_custom_speed_stages(cfg, speed_stages, steps_per_iter):
  """Speed curriculum beyond mjlab's 3 m/s: [(iteration, vmax), ...].

  From each iteration of this run on, forward commands go up to vmax m/s, backward
  up to min(2, vmax) m/s, and turning up to 0.7 rad/s (as in mjlab's last stage).
  Counted and applied like set_speed_stages.
  """
  iters = [it for it, _ in speed_stages]
  if iters[0] != 0 or iters != sorted(iters):
    raise ValueError(f"speed_stages must start at iteration 0 and be in order, got {speed_stages}")
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": it * steps_per_iter, "lin_vel_x": (-min(2.0, vmax), vmax), "ang_vel_z": (-0.7, 0.7)}
    for it, vmax in speed_stages
  ]


def set_fixed_speed(cfg, speed, turn=0.0):
  """Every robot is told the same thing: go ahead at `speed` m/s while turning at `turn` rad/s."""
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (speed, speed)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (turn, turn)
  twist.rel_standing_envs = 0.0
  twist.rel_forward_envs = 0.0  # these robots get abs(speed) clamped to >= 0.3
  twist.rel_world_envs = 0.0
  # Heading robots turn toward a random heading by overwriting ang_vel_z.
  twist.heading_command = False
  twist.ranges.heading = None  # mjlab rejects a heading range without heading_command
  cfg.curriculum.pop("command_vel", None)  # it would overwrite the ranges above


def set_external_commands(cfg):
  """mjlab never picks or changes the commands; the caller writes them (MjlabEnv.set_command)."""
  set_fixed_speed(cfg, 0.0)
  cfg.commands["twist"].resampling_time_range = (1e9, 1e9)  # never draw a new random command


def override_rewards(
  cfg, air_time=None, action_rate=None, fell_over_weight=None, pose_running_scale=None,
  speed_shortfall_weight=None, yaw_error_weight=None,
):
  """Change reward terms before the env is built. None keeps mjlab's setting.

  Weights are multiplied by step_dt (0.02 s), like every mjlab reward:
  fell_over_weight=-200 costs 4 per fall; speed_shortfall_weight=-1 costs 1 per
  second per m/s of missing speed; yaw_error_weight=-1 costs 1 per second per rad/s
  of turning error.
  """
  if speed_shortfall_weight is not None:
    cfg.rewards["speed_shortfall"] = RewardTermCfg(func=speed_shortfall, weight=speed_shortfall_weight)
  if yaw_error_weight is not None:
    cfg.rewards["yaw_error"] = RewardTermCfg(func=yaw_error, weight=yaw_error_weight)
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
    stage_iters=None, speed_stages=None, steps_per_iter=24, fixed_speed=None, rewards=None,
    external_commands=False, viewer=None, fixed_turn=0.0, spec_fn=None, expand_fields=(), cfg_fn=None,
  ):
    # play=True: evaluation version of the task (no pushes, no sensor noise, no time limit).
    cfg = load_env_cfg(task, play=play)
    cfg.scene.num_envs = num_envs
    if spec_fn is not None:  # add or change geometry before mjlab compiles the scene
      cfg.scene.spec_fn = spec_fn
    if stage_iters is not None:
      set_speed_stages(cfg, stage_iters, steps_per_iter)
    if speed_stages is not None:
      set_custom_speed_stages(cfg, speed_stages, steps_per_iter)
    if fixed_speed is not None:
      set_fixed_speed(cfg, fixed_speed, fixed_turn)
    if external_commands:
      set_external_commands(cfg)
    if rewards:
      override_rewards(cfg, **rewards)
    if cfg_fn is not None:  # any other change to the task, e.g. standup.configure
      cfg_fn(cfg)
    if video_dir:
      cfg.viewer.width, cfg.viewer.height = 640, 480
    for key, value in (viewer or {}).items():  # camera settings, e.g. {"distance": 20.0}
      setattr(cfg.viewer, key, value)
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if video_dir else None)
    # Model fields that may differ between worlds (e.g. "geom_pos"); all others are shared.
    self.env.unwrapped.sim.expand_model_fields(tuple(expand_fields))
    if video_dir:
      self.env = VideoRecorder(
        self.env, video_folder=Path(video_dir), step_trigger=lambda s: s == 0,
        video_length=video_steps, disable_logger=True,
      )
    self.num_envs = num_envs
    self.num_actions = self.env.unwrapped.action_manager.total_action_dim
    self.step_dt = self.env.unwrapped.step_dt

  @property
  def reward_terms(self):
    """Names of the reward terms with a nonzero weight (mjlab skips the others)."""
    rm = self.env.unwrapped.reward_manager
    return [name for name in rm.active_terms if rm.get_term_cfg(name).weight != 0]

  def forward_speed(self):
    """Commanded and actual forward speed (m/s, pelvis frame) of every robot."""
    env = self.env.unwrapped
    commanded = env.command_manager.get_command("twist")[:, 0]
    actual = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return commanded, actual

  def turn_rate(self):
    """Commanded and actual turning speed (rad/s, about the pelvis z axis) of every robot."""
    env = self.env.unwrapped
    commanded = env.command_manager.get_command("twist")[:, 2]
    actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return commanded, actual

  def time_since_reset(self):
    """Seconds since each robot last (re)spawned."""
    return self.env.unwrapped.episode_length_buf * self.step_dt

  @property
  def robot(self):
    """mjlab's robot entity: robot.data has positions, velocities, heading, ..."""
    return self.env.unwrapped.scene["robot"]

  def actor_obs(self):
    """Recompute the actor observations from the current state (e.g. after a new command)."""
    return self.env.unwrapped.observation_manager.compute_group("actor")

  def set_command(self, cmd):
    """Give every robot a new (forward m/s, sideways m/s, turn rad/s) command; returns fresh actor obs.

    Needs external_commands=True, otherwise mjlab overwrites it with its own random commands.
    """
    self.env.unwrapped.command_manager.get_term("twist").vel_command_b[:] = cmd
    return self.actor_obs()

  def place_robots(self, ids, xy, yaw):
    """Respawn the robots in worlds `ids` standing still at xy (m), facing yaw (rad).

    mjlab respawns them as usual (default joint angles, random spot), then we move them.
    Returns fresh actor obs for all robots.
    """
    env = self.env.unwrapped
    env.reset(env_ids=ids)
    pose = torch.zeros(len(ids), 7, device=xy.device)
    pose[:, :2] = xy
    pose[:, 2] = self.robot.data.root_link_pos_w[ids, 2]  # keep mjlab's spawn height
    pose[:, 3] = torch.cos(yaw / 2)  # quaternion (w, x, y, z) of a turn by yaw about the z axis
    pose[:, 6] = torch.sin(yaw / 2)
    self.robot.write_root_link_pose_to_sim(pose, env_ids=ids)
    self.robot.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=xy.device), env_ids=ids)
    env.sim.forward()  # update body positions from the new pose
    env.sim.sense()  # and the IMU readings the actor observes
    return self.actor_obs()

  def reset(self):
    obs, _ = self.env.reset()
    return obs["actor"], obs["critic"]

  def step(self, actions):
    obs, reward, terminated, truncated, extras = self.env.step(actions)
    return obs["actor"], obs["critic"], reward, terminated, truncated, extras.get("log", {})

  def close(self):
    self.env.close()
