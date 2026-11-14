"""Getting up after a fall: a frozen HoST stand-up policy, switched in per robot.

The stand-up policy was trained in ../HoST/mjlab_port (HoST, RSS 2025, ported to mjlab's
29-DoF G1, the same robot as here). It has its own inputs, its own way of turning actions
into joint targets, and its own joint damping, so a robot that falls is handed over to it
completely, then handed back to the legs once it stands again. A state machine per robot:

  WALK     the legs drive. Tilting past 70° (mjlab's own fell_over angle) is a fall -> STANDUP
  STANDUP  0.6 s limp (HoST's "unactuated window": zero actions while the body settles),
           then the stand-up policy. Pelvis above 0.7 m and upright for 0.4 s -> STEADY
  STEADY   the legs drive again, told to stand still, for 0.5 s -> WALK

How it plugs into mjlab:
  - configure(cfg) swaps the legs' action term for StandupAction, which writes the legs'
    joint targets everywhere except in worlds where the stand-up policy has control. mjlab
    calls it every physics substep, as the stand-up policy needs. It also removes mjlab's
    fell_over termination (otherwise a fallen robot is respawned standing) and enlarges the
    contact buffers (a robot lying on the ground has many more contacts than one walking).
  - The joint damping (D gain) is a per-world model field, so build MjlabEnv with
    expand_fields including "actuator_biasprm". Recovery swaps it while a robot gets up.
  - Recovery runs the state machine and the stand-up network. Call act() on the legs'
    actions before every env step and update() after it.

Deviations from the stand-up training: no per-robot gain, motor-strength or offset
randomization and no action delay (training sampled them around these nominal values).
"""

import math
from dataclasses import fields

import torch
from torch import nn

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

# ---------------------------------------------------------------------------- #
# The stand-up policy's setup, copied from ../HoST/mjlab_port/host_mjlab (robot.py,
# tasks.py, mdp/actions.py, mdp/observations.py). Change only together with a retrain.
# ---------------------------------------------------------------------------- #
# The 23 joints the stand-up policy drives (HoST's 23-DoF G1), in its action order.
CONTROLLED_JOINTS = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint",
)
# The 6 joints the 29-DoF model adds: held at 0 while the stand-up policy drives.
LOCKED_JOINTS = ("waist_roll_joint", "waist_pitch_joint", "left_wrist_pitch_joint",
                 "left_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
# HoST's PD gains, matched by substring of the joint name (the last match wins). The
# stiffness is only used to scale actions; the damping replaces mjlab's while getting up.
HOST_STIFFNESS = {"hip": 150.0, "knee": 200.0, "ankle": 40.0, "shoulder": 100.0, "elbow": 100.0,
                  "waist": 100.0, "wrist": 100.0}
HOST_DAMPING = {"hip": 4.0, "knee": 6.0, "ankle": 2.0, "shoulder": 4.0, "elbow": 4.0,
                "waist": 4.0, "wrist": 4.0}
BETA = 0.25  # action bound at the end of HoST's curriculum
BETA_NOISE = 0.025  # HoST always adds U(-0.025, 0.025) to the beta input, also when evaluating
WINDOW = 30  # policy steps of the unactuated window (0.6 s): zero actions and zero inputs
HISTORY = 6  # input frames, oldest first
FRAME = 3 + 3 + 3 * len(CONTROLLED_JOINTS) + 1  # numbers per frame: 76
ANG_VEL_SCALE, JOINT_VEL_SCALE = 0.25, 0.05
NCONMAX, NJMAX = 64, 600  # contact and constraint buffers per world for full-body contact

# ---------------------------------------------------------------------------- #
# The state machine's thresholds.
# ---------------------------------------------------------------------------- #
WALK, STANDUP, STEADY = range(3)
FALL_ANGLE = math.radians(70.0)  # the same as mjlab's fell_over termination
UP_HEIGHT = 0.7  # pelvis height (m) that counts as standing (HoST's success threshold)
UP_ANGLE = math.radians(20.0)  # ... with the torso at most this far from upright
UP_HOLD_S = 0.4  # ... for this long
STEADY_S = 0.5  # after the hand-over, the legs stand still this long before walking on
REFALL_S = 2.0  # a fall this soon after a hand-over counts as a "refall"
POSTURES = ("back", "front", "side")  # how a robot lies at the end of the unactuated window


def gain(joint, table):
  value = None
  for key, v in table.items():
    if key in joint:
      value = v
  return value


def load_standup_actor(path, device):
  """The stand-up actor from an rsl_rl checkpoint of ../HoST/mjlab_port, as a plain MLP.

  456 inputs -> 512 -> 256 -> 128 -> 23 actions in [-1, 1], ELU, no input normalization.
  Its mean action is used; the exploration std in the checkpoint is ignored.
  """
  state = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
  state = {k.removeprefix("mlp."): v for k, v in state.items() if k.startswith("mlp.")}
  sizes = [state["0.weight"].shape[1]] + [state[f"{i}.weight"].shape[0] for i in (0, 2, 4, 6)]
  layers = []
  for i in range(4):
    layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ELU() if i < 3 else nn.Tanh()]
  actor = nn.Sequential(*layers).to(device)
  actor.load_state_dict(state)
  return actor.eval()


# ---------------------------------------------------------------------------- #
# mjlab side
# ---------------------------------------------------------------------------- #
class StandupActionCfg(JointPositionActionCfg):
  def build(self, env):
    return StandupAction(self, env)


class StandupAction(JointPositionAction):
  """The legs' joint targets, except in worlds where `active` is set: there the controlled
  joints get HoST's target, current position + delta, recomputed every physics substep, and
  the locked joints get 0. Recovery sets `active` and `delta` (rad) before every env step."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.controlled = torch.tensor(self._entity.find_joints(CONTROLLED_JOINTS, preserve_order=True)[0],
                                   device=self.device)
    self.locked = torch.tensor(self._entity.find_joints(LOCKED_JOINTS, preserve_order=True)[0],
                               device=self.device)
    self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.delta = torch.zeros(self.num_envs, len(CONTROLLED_JOINTS), device=self.device)

  def apply_actions(self):
    super().apply_actions()  # the legs' targets, for every world
    if not self.active.any():
      return
    data, on = self._entity.data, self.active[:, None]
    legs = data.joint_pos_target
    target = torch.where(on, data.joint_pos[:, self.controlled] + self.delta, legs[:, self.controlled])
    self._entity.set_joint_position_target(target, joint_ids=self.controlled)
    locked = torch.where(on, 0.0, legs[:, self.locked])
    self._entity.set_joint_position_target(locked, joint_ids=self.locked)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    self.active[slice(None) if env_ids is None else env_ids] = False


def configure(cfg):
  """MjlabEnv(cfg_fn=configure): make the task ready for Recovery (see the module docstring)."""
  legs = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = StandupActionCfg(**{f.name: getattr(legs, f.name) for f in fields(legs)})
  cfg.terminations.pop("fell_over")
  cfg.sim.nconmax, cfg.sim.njmax = NCONMAX, NJMAX


# ---------------------------------------------------------------------------- #
# The state machine
# ---------------------------------------------------------------------------- #
class Recovery:
  """Per-robot WALK / STANDUP / STEADY switching for an MjlabEnv built with configure().

  standup_s: a robot still down this long after its unactuated window has "given up"
  (update() reports it; the caller decides what that means, it keeps trying anyway).
  """

  def __init__(self, mj, standup_ckpt, device, standup_s=5.0):
    env = mj.env.unwrapped
    self.mj, self.device, self.dt = mj, device, mj.step_dt
    self.term = env.action_manager.get_term("joint_pos")
    assert isinstance(self.term, StandupAction), "build MjlabEnv with cfg_fn=standup.configure"
    self.actor = load_standup_actor(standup_ckpt, device)
    self.limit_steps = WINDOW + round(standup_s / self.dt)

    # Actuator (ctrl) index of each controlled joint, and mjlab's gains, read from the model.
    ctrl_of = {}
    for act in mj.robot.actuators:
      ctrl_of.update(zip(act.target_names, act.global_ctrl_ids.tolist()))
    self.ctrl = torch.tensor([ctrl_of[j] for j in CONTROLLED_JOINTS], device=device)
    self.model = env.sim.model
    assert self.model.actuator_biasprm.shape[0] == mj.num_envs, "expand_fields needs 'actuator_biasprm'"
    kp_mjlab = self.model.actuator_gainprm[0, self.ctrl, 0]
    self.kd_mjlab = -self.model.actuator_biasprm[0, self.ctrl, 2].clone()
    self.kd_host = torch.tensor([gain(j, HOST_DAMPING) for j in CONTROLLED_JOINTS], device=device)
    # The P term is kp_mjlab * (target - q) = kp_mjlab * delta. HoST's is kp_host * beta * a.
    self.action_to_delta = BETA * torch.tensor([gain(j, HOST_STIFFNESS) for j in CONTROLLED_JOINTS],
                                               device=device) / kp_mjlab

    n, d = mj.num_envs, len(CONTROLLED_JOINTS)
    self.mode = torch.full((n,), WALK, dtype=torch.long, device=device)
    self.counter = torch.zeros(n, dtype=torch.long, device=device)  # steps since the fall
    self.up_steps = torch.zeros(n, dtype=torch.long, device=device)  # steps standing so far
    self.steady_steps = torch.zeros(n, dtype=torch.long, device=device)
    self.since_up = torch.full((n,), 10**9, dtype=torch.long, device=device)  # steps since the hand-over
    self.history = torch.zeros(n, HISTORY, FRAME, device=device)
    self.last_action = torch.zeros(n, d, device=device)
    self.posture = torch.full((n,), -1, dtype=torch.long, device=device)
    self.robot_falls = torch.zeros(n, dtype=torch.long, device=device)  # falls of each robot since its reset
    self.reset_stats()

  def reset_stats(self):
    """Counters since the start (or the last call): falls, refalls, and per posture of the
    fallen robot, how many stood up in time, gave up, and their total get-up time (s, from
    the fall to the hand-over)."""
    self.falls = 0
    self.refalls = 0
    self.stood = torch.zeros(len(POSTURES), device=self.device)
    self.gave_up_count = torch.zeros(len(POSTURES), device=self.device)
    self.stand_s = torch.zeros(len(POSTURES), device=self.device)

  @property
  def down(self):
    """Robots the legs aren't free to steer: getting up, or standing still after it."""
    return self.mode != WALK

  def _frame(self):
    """One 76-number input frame per robot (HoST's host_proprio)."""
    data, j = self.mj.robot.data, self.term.controlled
    beta = BETA + (torch.rand(self.mj.num_envs, 1, device=self.device) * 2 - 1) * BETA_NOISE
    frame = torch.cat([
      data.root_link_ang_vel_b * ANG_VEL_SCALE, data.projected_gravity_b,
      data.joint_pos[:, j], data.joint_vel[:, j] * JOINT_VEL_SCALE,
      self.last_action, beta,
    ], dim=1)
    return frame.clamp(-100.0, 100.0)

  def act(self, legs_actions):
    """Before every env step: run the stand-up policy for robots that are getting up.
    Returns the legs' actions with those robots' set to 0, so the legs' "last action"
    input is a plain 0 (the default pose) when they take over again."""
    standing_up = self.mode == STANDUP
    self.term.active[:] = standing_up
    if not standing_up.any():
      return legs_actions
    visible = (self.counter > WINDOW)[:, None]  # inputs and actions are 0 in the window
    frame = self._frame() * visible
    self.history = torch.cat([self.history[:, 1:], frame[:, None]], dim=1)
    with torch.no_grad():
      a = self.actor(self.history.flatten(1)) * visible
    self.last_action = torch.where(standing_up[:, None], a, self.last_action)
    self.term.delta[:] = a * self.action_to_delta
    self.counter += standing_up.long()
    return torch.where(standing_up[:, None], 0.0, legs_actions)

  def _set_damping(self, worlds, kd):
    self.model.actuator_biasprm[worlds[:, None], self.ctrl[None, :], 2] = -kd

  def update(self):
    """After every env step: move robots between modes. Returns the robots that have just
    given up (still down standup_s after the window)."""
    data = self.mj.robot.data
    g = data.projected_gravity_b
    tilt = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
    self.since_up += 1

    # Falls (also during STEADY: a refall).
    fell = (self.mode != STANDUP) & (tilt > FALL_ANGLE)
    if fell.any():
      w = fell.nonzero().squeeze(1)
      self.falls += len(w)
      self.robot_falls[w] += 1
      self.refalls += int((self.since_up[w] * self.dt < REFALL_S).sum())
      self.mode[w], self.counter[w], self.up_steps[w], self.posture[w] = STANDUP, 0, 0, -1
      self.history[w], self.last_action[w] = 0.0, 0.0
      self._set_damping(w, self.kd_host)

    getting_up = self.mode == STANDUP
    # How it lies when the window ends: on its back (body x axis pointing up), front, or side.
    window_end = getting_up & (self.counter == WINDOW + 1)
    if window_end.any():
      lying = torch.where(g[:, 0] < -0.5, 0, torch.where(g[:, 0] > 0.5, 1, 2))
      self.posture = torch.where(window_end, lying, self.posture)

    up = getting_up & (self.counter > WINDOW) & (data.root_link_pos_w[:, 2] > UP_HEIGHT) & (tilt < UP_ANGLE)
    self.up_steps = torch.where(up, self.up_steps + 1, 0)
    stood = up & (self.up_steps * self.dt >= UP_HOLD_S)
    if stood.any():
      w = stood.nonzero().squeeze(1)
      in_time = w[self.counter[w] <= self.limit_steps]  # later ones were already counted as given up
      self.stood += torch.bincount(self.posture[in_time], minlength=len(POSTURES))
      self.stand_s += torch.bincount(self.posture[in_time], weights=self.counter[in_time] * self.dt,
                                     minlength=len(POSTURES))
      self.mode[w], self.steady_steps[w], self.since_up[w] = STEADY, 0, 0
      self._set_damping(w, self.kd_mjlab)

    steady = self.mode == STEADY
    self.steady_steps += steady.long()
    self.mode[steady & (self.steady_steps * self.dt >= STEADY_S)] = WALK

    gave_up = (self.mode == STANDUP) & (self.counter == self.limit_steps)
    if gave_up.any():
      self.gave_up_count += torch.bincount(self.posture[gave_up], minlength=len(POSTURES))
    return gave_up

  def reset(self, worlds):
    """Robots in `worlds` were respawned standing: back to WALK with mjlab's damping."""
    self.mode[worlds], self.counter[worlds], self.up_steps[worlds] = WALK, 0, 0
    self.since_up[worlds], self.robot_falls[worlds] = 10**9, 0
    self.term.active[worlds] = False
    self._set_damping(worlds, self.kd_mjlab)
