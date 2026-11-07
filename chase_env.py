"""Runner vs chaser: a two-player zero-sum game between two G1 robots.

The frozen sprint policy (the "legs") turns velocity commands into joint targets.
The game policy only picks each robot's command, 10 times per second.

Rules, checked after every 0.02 s physics step (the first one that applies wins):
  1. A robot fell over or left the arena: it loses (both on the same step: draw).
  2. The two pelvises are closer than catch_radius: the chaser wins.
  3. game_s seconds have passed: the runner wins.

Each robot lives in its own mjlab world, so the two robots never touch ("ghost"
robots: a catch is a tag, not a tackle). All worlds share one coordinate frame:
the arena is the square |x|, |y| < arena / 2 around (0, 0), and we place the
robots ourselves at the start of every game.

The env looks like any other env to ppo.py: agents 0..G-1 are the runners and
agents G..2G-1 the chasers of games 0..G-1; agent i drives the robot in world i.
One network plays both roles; an observation tells it which one it is.
"""

import math

import torch

from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab.viewer.viewer_config import ViewerConfig
from mjlab_env import MjlabEnv
from ppo import load_actor_critic

TASK = "Mjlab-Velocity-Flat-Unitree-G1"
# Commands the sprint policy was trained on (forward -2..5 m/s, sideways ±1 m/s, turning ±0.7 rad/s).
CMD_LOW = (-2.0, -1.0, -0.7)
CMD_HIGH = (5.0, 1.0, 0.7)
WIN = 10.0  # reward for winning (the loser gets -WIN)
OUTCOMES = ("chaser_win", "runner_win", "runner_fell", "chaser_fell", "runner_out", "chaser_out", "draw")
CHASER_WIN, RUNNER_WIN, RUNNER_FELL, CHASER_FELL, RUNNER_OUT, CHASER_OUT, DRAW = range(len(OUTCOMES))
# +1 if the outcome is a win for the chaser, -1 for the runner, 0 for a draw (same order as OUTCOMES).
CHASER_SIGN = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0)


def add_game_args(p):
  """Game settings shared by train_chase.py and eval_chase.py."""
  p.add_argument("--legs", default="runs/g1_sprint2/model_3999.pt", help="frozen locomotion policy")
  p.add_argument("--arena", type=float, default=20.0, help="side of the square arena (m)")
  p.add_argument("--catch_radius", type=float, default=0.8, help="pelvis distance that counts as a catch (m)")
  p.add_argument("--game_s", type=float, default=20.0, help="the runner wins after this many seconds")
  p.add_argument("--shaping", type=float, default=1.0, help="chaser reward per metre closed (runner: the opposite)")


def game_kwargs(args):
  return dict(legs_ckpt=args.legs, arena=args.arena, catch_radius=args.catch_radius,
              game_s=args.game_s, shaping=args.shaping)


def rotate(v, heading):
  """World-frame xy vectors -> the frame of a robot facing `heading` (x forward, y left)."""
  c, s = torch.cos(heading), torch.sin(heading)
  return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], dim=1)


class ChaseEnv:
  def __init__(
    self, num_games, device, legs_ckpt, arena=20.0, catch_radius=0.8, game_s=20.0,
    decision_s=0.1, shaping=1.0, start_dist=(3.0, 8.0), video_dir=None,
  ):
    self.G = num_games
    self.num_envs = 2 * num_games  # agents, as ppo.py sees them
    self.num_actions = 3  # forward speed, sideways speed, turning speed
    self.device = device
    self.half = arena / 2
    self.catch_radius = catch_radius
    self.shaping = shaping
    self.start_dist = start_dist

    # Video: one game (worlds 0 and 1) seen from a fixed camera above the arena.
    viewer = None
    if video_dir:
      viewer = dict(origin_type=ViewerConfig.OriginType.WORLD, lookat=(0.0, 0.0, 0.0),
                    distance=arena * 1.1, elevation=-60.0, max_extra_envs=1)
    self.mj = MjlabEnv(TASK, self.num_envs, device, play=True, external_commands=True,
                       video_dir=video_dir, video_steps=int(game_s / 0.02), viewer=viewer)
    self.substeps = round(decision_s / self.mj.step_dt)  # physics steps per decision
    self.game_steps = round(game_s / self.mj.step_dt)
    self.game_s = game_s

    self.legs, _ = load_actor_critic(legs_ckpt, device)
    self.legs.eval()

    g = torch.arange(num_games, device=device)
    self.opp = torch.cat([g + num_games, g])  # world of each agent's opponent
    self.role = torch.cat([-torch.ones(num_games, device=device), torch.ones(num_games, device=device)])
    self.cmd_low = torch.tensor(CMD_LOW, device=device)
    self.cmd_high = torch.tensor(CMD_HIGH, device=device)
    self.chaser_sign = torch.tensor(CHASER_SIGN, device=device)
    self.command = torch.zeros(self.num_envs, 3, device=device)  # current command of every robot
    self.steps = torch.zeros(num_games, dtype=torch.long, device=device)  # physics steps since game start
    self.dist = torch.zeros(num_games, device=device)  # pelvis distance at the last decision
    # Result of the games that ended in the last step(): index into OUTCOMES, -1 = still playing.
    self.outcome = torch.full((num_games,), -1, dtype=torch.long, device=device)
    self.length_s = torch.zeros(num_games, device=device)

  # ------------------------------------------------------------------------- #
  # Game state
  # ------------------------------------------------------------------------- #
  def _pos(self):
    return self.mj.robot.data.root_link_pos_w[:, :2]

  def _new_games(self, games):
    """Start new games: runner somewhere inside the arena, chaser 3-8 m away, random headings."""
    n, G, dev = len(games), self.G, self.device
    runner_xy = (torch.rand(n, 2, device=dev) * 2 - 1) * (self.half - 3.0)
    angle = torch.rand(n, device=dev) * 2 * math.pi
    dist = torch.empty(n, device=dev).uniform_(*self.start_dist)
    offset = dist[:, None] * torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    # Keep the chaser at least 1 m from the walls (so it may start a bit closer than 3 m).
    chaser_xy = (runner_xy + offset).clamp(-(self.half - 1.0), self.half - 1.0)
    yaw = torch.rand(2 * n, device=dev) * 2 * math.pi - math.pi
    worlds = torch.cat([games, games + G])
    self.mj.place_robots(worlds, torch.cat([runner_xy, chaser_xy]), yaw)
    self.steps[games] = 0
    self.command[worlds] = 0.0
    self.dist[games] = (runner_xy - chaser_xy).norm(dim=1)

  def _observe(self):
    """Everything in the agent's own frame: x = where it faces, y = to its left."""
    d, o = self.mj.robot.data, self.opp
    pos, heading = self._pos(), d.heading_w
    vel, yaw_rate = d.root_link_lin_vel_w[:, :2], d.root_link_ang_vel_w[:, 2:3]
    rel = rotate(pos[o] - pos, heading)  # where the opponent is
    rel_heading = heading[o] - heading  # which way the opponent faces
    wall = self.half - pos.abs().max(dim=1).values  # distance to the nearest wall
    time_left = (1 - self.steps.float() / self.game_steps).repeat(2)
    command = self.command / self.cmd_high  # roughly in [-1, 1]
    actor = torch.cat([
      self.role[:, None], time_left[:, None],
      rel, rel.norm(dim=1, keepdim=True), torch.cos(rel_heading)[:, None], torch.sin(rel_heading)[:, None],
      rotate(vel, heading), yaw_rate,
      rotate(vel[o], heading), yaw_rate[o],
      rotate(-pos, heading), wall[:, None], wall[o][:, None],  # arena centre, walls
      command, d.projected_gravity_b,  # own last command, own tilt
    ], dim=1)
    # The critic also sees the opponent's tilt and its current command.
    critic = torch.cat([actor, d.projected_gravity_b[o], command[o]], dim=1)
    return actor, critic

  # ------------------------------------------------------------------------- #
  # ppo.py interface
  # ------------------------------------------------------------------------- #
  def reset(self):
    self.mj.reset()
    self._new_games(torch.arange(self.G, device=self.device))
    return self._observe()

  def step(self, actions):
    G = self.G
    a = actions.clamp(-1.0, 1.0)
    self.command = self.cmd_low + (a + 1) / 2 * (self.cmd_high - self.cmd_low)
    legs_obs = self.mj.set_command(self.command)

    outcome = torch.full((G,), -1, dtype=torch.long, device=self.device)
    for _ in range(self.substeps):
      with torch.no_grad():
        legs_obs, _, _, fell, _, _ = self.mj.step(self.legs.act_deterministic(legs_obs))
      self.steps += 1
      pos = self._pos()  # a robot that fell has already been respawned by mjlab, but fell says so
      lost = fell | (pos.abs().max(dim=1).values > self.half)
      fell_r, fell_c, lost_r, lost_c = fell[:G], fell[G:], lost[:G], lost[G:]
      result = torch.full_like(outcome, -1)
      result = torch.where(lost_r & ~lost_c, torch.where(fell_r, RUNNER_FELL, RUNNER_OUT), result)
      result = torch.where(lost_c & ~lost_r, torch.where(fell_c, CHASER_FELL, CHASER_OUT), result)
      result = torch.where(lost_r & lost_c, DRAW, result)
      caught = (pos[:G] - pos[G:]).norm(dim=1) < self.catch_radius
      result = torch.where((result < 0) & caught, CHASER_WIN, result)
      result = torch.where((result < 0) & (self.steps >= self.game_steps), RUNNER_WIN, result)
      outcome = torch.where(outcome < 0, result, outcome)  # a game keeps its first result

    # Zero-sum reward, from the chaser's point of view: shaping + win/loss.
    ended = outcome >= 0
    pos = self._pos()
    dist = (pos[:G] - pos[G:]).norm(dim=1)
    chaser_reward = torch.where(ended, 0.0, self.shaping * (self.dist - dist))  # metres closed
    chaser_reward += WIN * self.chaser_sign[outcome.clamp(min=0)] * ended
    reward = torch.cat([-chaser_reward, chaser_reward])
    self.dist = dist

    self.outcome = outcome
    self.length_s = self.steps * self.mj.step_dt
    info = {}
    if ended.any():
      results = outcome[ended]
      for k, name in enumerate(OUTCOMES):
        info[f"Game/{name}"] = (results == k).float().mean().item()
      info["Game/length_s"] = self.length_s[ended].mean().item()
      self._new_games(ended.nonzero().squeeze(-1))

    terminated = ended.repeat(2)  # the game is over for both players
    truncated = torch.zeros_like(terminated)  # the 20 s limit is a game rule, not a cut-off
    actor, critic = self._observe()
    return actor, critic, reward, terminated, truncated, info

  # ------------------------------------------------------------------------- #
  # Scripted opponents (baselines for eval_chase.py), as actions in [-1, 1]
  # ------------------------------------------------------------------------- #
  def _steer(self, ids, direction, speed):
    """Turn robots `ids` toward a world-frame direction and run at up to `speed` m/s."""
    heading = self.mj.robot.data.heading_w[ids]
    error = wrap_to_pi(torch.atan2(direction[:, 1], direction[:, 0]) - heading)
    turn = (2.0 * error).clamp(CMD_LOW[2], CMD_HIGH[2])
    forward = speed * torch.cos(error).clamp(min=0.2)  # slow down while facing the wrong way
    cmd = torch.stack([forward, torch.zeros_like(forward), turn], dim=1)
    return 2 * (cmd - self.cmd_low) / (self.cmd_high - self.cmd_low) - 1

  def pursue(self):
    """Chasers: run at full speed to where the runner will be (up to 1 s ahead)."""
    G, pos = self.G, self._pos()
    runner_vel = self.mj.robot.data.root_link_lin_vel_w[:G, :2]
    lead = ((pos[:G] - pos[G:]).norm(dim=1, keepdim=True) / CMD_HIGH[0]).clamp(max=1.0)
    target = pos[:G] + runner_vel * lead
    return self._steer(torch.arange(G, 2 * G, device=self.device), target - pos[G:], CMD_HIGH[0])

  def flee(self):
    """Runners: run at full speed away from the chaser, pushed back from walls closer than 4 m."""
    G, pos = self.G, self._pos()
    away = pos[:G] - pos[G:]
    away = away / away.norm(dim=1, keepdim=True).clamp(min=1e-6)
    near_wall = (pos[:G].abs() - (self.half - 4.0)).clamp(min=0.0) / 4.0  # 0 far away, 1 at the wall
    direction = away - 2.0 * torch.sign(pos[:G]) * near_wall
    return self._steer(torch.arange(G, device=self.device), direction, CMD_HIGH[0])

  def close(self):
    self.mj.close()
