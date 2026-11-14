"""Runner vs chaser: a two-player zero-sum game between two G1 robots.

The frozen sprint policy (the "legs") turns velocity commands into joint targets.
The game policy only picks each robot's command, 10 times per second.

Rules, checked after every 0.02 s physics step (the first one that applies wins):
  1. A robot fell over or left the arena: it loses (both on the same step: draw).
     With --standup, a fall only costs time: a frozen stand-up policy (standup.py) gets the
     robot up again, and it loses only if it isn't standing standup_s after its 0.6 s settle.
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

import numpy as np
import torch

from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab.viewer.viewer_config import ViewerConfig
from mjlab_env import MjlabEnv
from ppo import load_actor_critic
from standup import Recovery, configure
from terrain import FIELDS, ROCKY, TerrainArena

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
  # g1_turn/model_4349: runs 5 m/s and turns (slowly); g1_sprint2 can't turn. See EXPERIMENTS.md.
  p.add_argument("--legs", default="runs/g1_turn/model_4349.pt", help="frozen locomotion policy")
  p.add_argument("--arena", type=float, default=20.0, help="side of the square arena (m)")
  p.add_argument("--catch_radius", type=float, default=0.8, help="pelvis distance that counts as a catch (m)")
  p.add_argument("--max_speed", type=float, default=CMD_HIGH[0],
                 help="game rule: cap on every robot's forward speed command (m/s); the policy still outputs the full range")
  p.add_argument("--game_s", type=float, default=20.0, help="the runner wins after this many seconds")
  p.add_argument("--shaping", type=float, default=1.0, help="chaser reward per metre closed (runner: the opposite)")
  p.add_argument("--terrain", action="store_true",
                 help="patchwork arena: normal, grass, mud and rocky cells, a new layout every game (terrain.py)")
  p.add_argument("--standup", default=None,
                 help="stand-up checkpoint (e.g. runs/standup/model_11000.pt): fallen robots get up again")
  p.add_argument("--standup_s", type=float, default=5.0,
                 help="with --standup: a robot not standing this long after its 0.6 s settle loses")


def game_kwargs(args):
  return dict(legs_ckpt=args.legs, arena=args.arena, catch_radius=args.catch_radius,
              game_s=args.game_s, shaping=args.shaping, terrain=args.terrain,
              standup_ckpt=args.standup, standup_s=args.standup_s, max_speed=args.max_speed)


def rotate(v, heading):
  """World-frame xy vectors -> the frame of a robot facing `heading` (x forward, y left)."""
  c, s = torch.cos(heading), torch.sin(heading)
  return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], dim=1)


class ChaseEnv:
  def __init__(
    self, num_games, device, legs_ckpt, arena=20.0, catch_radius=0.8, game_s=20.0,
    decision_s=0.1, shaping=1.0, start_dist=(3.0, 8.0), video_dir=None, camera="chaser", terrain=False,
    standup_ckpt=None, standup_s=5.0, max_speed=CMD_HIGH[0],
  ):
    self.G = num_games
    self.num_envs = 2 * num_games  # agents, as ppo.py sees them
    self.num_actions = 3  # forward speed, sideways speed, turning speed
    self.device = device
    self.half = arena / 2
    self.catch_radius = catch_radius
    self.max_speed = max_speed
    self.shaping = shaping
    self.start_dist = start_dist
    self.camera = camera

    # Video of game 0 (use num_games=1). camera="chaser": a free camera that debug_vis()
    # moves every frame to sit behind the chaser, looking at the runner.
    # camera="overhead": fixed, above the arena. The chaser's world is the "primary" one,
    # so mjlab draws its command arrows; the runner's world is drawn as an extra.
    viewer = None
    if video_dir:
      viewer = dict(origin_type=ViewerConfig.OriginType.WORLD, env_idx=num_games, max_extra_envs=1,
                    width=960, height=720, distance=arena, lookat=(0.0, 0.0, 0.0))
      if camera == "overhead":
        viewer.update(distance=arena * 1.45, elevation=-70.0)
    # terrain=True: grass / mud / rocky patches (terrain.py). Their per-world model fields are
    # expanded so every game can have its own layout.
    self.terrain = TerrainArena(arena, device) if terrain else None
    # standup_ckpt: the task gets the switchable action (standup.configure), and the joint
    # damping becomes per-world, so it can change while a robot gets up.
    expand = (FIELDS if terrain else ()) + (("actuator_biasprm",) if standup_ckpt else ())
    self.mj = MjlabEnv(TASK, self.num_envs, device, play=True, external_commands=True,
                       video_dir=video_dir, video_steps=int(game_s / 0.02), viewer=viewer,
                       spec_fn=self.terrain.spec_fn if terrain else None, expand_fields=expand,
                       cfg_fn=configure if standup_ckpt else None)
    if terrain:
      self.terrain.bind(self.mj)
    self.recovery = Recovery(self.mj, standup_ckpt, device, standup_s) if standup_ckpt else None
    if video_dir:
      # mjlab calls debug_vis() of everything in this dict when it renders a frame.
      self.mj.env.unwrapped.manager_visualizers["chase"] = self
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
    self.falls = torch.zeros(num_games, 2, dtype=torch.long, device=device)  # falls (runner, chaser) of the games that just ended
    self.terrain_under = None  # terrain type under every robot at the last decision (terrain only)

  # ------------------------------------------------------------------------- #
  # Game state
  # ------------------------------------------------------------------------- #
  def _pos(self):
    return self.mj.robot.data.root_link_pos_w[:, :2]

  def _new_games(self, games):
    """Start new games: runner somewhere inside the arena, chaser 3-8 m away, random headings."""
    n, G, dev = len(games), self.G, self.device
    worlds = torch.cat([games, games + G])
    # With terrain: a new layout, the same in both worlds of a game, and 16 candidate spots
    # per robot, of which the first one outside the rocky cells is used.
    tries = 16 if self.terrain else 1
    runner_xy = (torch.rand(n, tries, 2, device=dev) * 2 - 1) * (self.half - 3.0)
    angle = torch.rand(n, tries, device=dev) * 2 * math.pi
    dist = torch.empty(n, tries, device=dev).uniform_(*self.start_dist)
    offset = dist[..., None] * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    # Keep the chaser at least 1 m from the walls (so it may start a bit closer than 3 m).
    chaser_xy = (runner_xy + offset).clamp(-(self.half - 1.0), self.half - 1.0)
    pick = torch.zeros(n, dtype=torch.long, device=dev)
    if self.terrain:
      cells, rock_xy, rock_yaw = self.terrain.random_layouts(n)
      self.terrain.set_cells(worlds, cells.repeat(2, 1))
      self.terrain.place_rocks(worlds, rock_xy.repeat(2, 1, 1), rock_yaw.repeat(2, 1))
      ok = torch.ones(n, tries, dtype=torch.bool, device=dev)
      for xy in (runner_xy, chaser_xy):
        types = self.terrain.type_at(games.repeat_interleave(tries), xy.reshape(-1, 2)).reshape(n, tries)
        ok &= types != ROCKY
      pick = ok.int().argmax(dim=1)  # first spot where neither robot is on rocks (0 if none)
    rows = torch.arange(n, device=dev)
    runner_xy, chaser_xy = runner_xy[rows, pick], chaser_xy[rows, pick]
    yaw = torch.rand(2 * n, device=dev) * 2 * math.pi - math.pi
    self.mj.place_robots(worlds, torch.cat([runner_xy, chaser_xy]), yaw)
    if self.recovery:
      self.recovery.reset(worlds)
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
    self.command[:, 0] = self.command[:, 0].clamp(max=self.max_speed)  # game rule: speed cap
    if self.terrain:
      # Game rule: the terrain under a robot caps its forward speed (e.g. 2 m/s in mud).
      self.terrain_under = self.terrain.under_robots()
      self.command[:, 0] = torch.minimum(self.command[:, 0], self.terrain.speed_cap(self.terrain_under))
    if self.recovery:  # a robot that is down, or just got up, is told to stand still
      self.command[self.recovery.down] = 0.0
    legs_obs = self.mj.set_command(self.command)

    outcome = torch.full((G,), -1, dtype=torch.long, device=self.device)
    for _ in range(self.substeps):
      if self.terrain:
        self.terrain.step_effects()
      with torch.no_grad():
        actions = self.legs.act_deterministic(legs_obs)
        if self.recovery:
          actions = self.recovery.act(actions)  # the stand-up policy drives the robots that are down
        legs_obs, _, _, fell, _, _ = self.mj.step(actions)
      if self.recovery:
        fell = self.recovery.update()  # a fall only counts once the robot has given up getting up
      self.steps += 1
      pos = self._pos()  # without --standup, a robot that fell has already been respawned by mjlab
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
      if self.recovery:
        self.falls = self.recovery.robot_falls.reshape(2, G).T.clone()  # read before the robots respawn
        info["Game/falls"] = self.falls[ended].float().mean().item()  # per robot and game
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
    # Distance to the arena edge straight ahead. At most 0.7 rad/s of turning means a turning
    # circle of radius speed / 0.7, so slow down to speed <= 0.7 * (distance - 1.5 m).
    pos, heading_vec = self._pos()[ids], torch.stack([torch.cos(heading), torch.sin(heading)], dim=1)
    ahead = ((self.half - pos * torch.sign(heading_vec)) / heading_vec.abs().clamp(min=1e-6)).min(dim=1).values
    forward = torch.minimum(forward, (CMD_HIGH[2] * (ahead - 1.5)).clamp(min=0.0))
    cmd = torch.stack([forward, torch.zeros_like(forward), turn], dim=1)
    return 2 * (cmd - self.cmd_low) / (self.cmd_high - self.cmd_low) - 1

  def pursue(self):
    """Chasers: run at full speed to where the runner will be (up to 1 s ahead)."""
    G, pos = self.G, self._pos()
    runner_vel = self.mj.robot.data.root_link_lin_vel_w[:G, :2]
    lead = ((pos[:G] - pos[G:]).norm(dim=1, keepdim=True) / self.max_speed).clamp(max=1.0)
    target = pos[:G] + runner_vel * lead
    return self._steer(torch.arange(G, 2 * G, device=self.device), target - pos[G:], self.max_speed)

  def flee(self):
    """Runners: run at full speed away from the chaser, pushed back from walls closer than 4 m."""
    G, pos = self.G, self._pos()
    away = pos[:G] - pos[G:]
    away = away / away.norm(dim=1, keepdim=True).clamp(min=1e-6)
    near_wall = (pos[:G].abs() - (self.half - 4.0)).clamp(min=0.0) / 4.0  # 0 far away, 1 at the wall
    direction = away - 2.0 * torch.sign(pos[:G]) * near_wall
    return self._steer(torch.arange(G, device=self.device), direction, self.max_speed)

  def debug_vis(self, vis):
    """Video overlay: yellow arena edges, a blue disc under the runner and a red disc
    (the catch radius) under the chaser of game 0."""
    h = self.half
    edges = (((h, 0), (0.05, h)), ((-h, 0), (0.05, h)), ((0, h), (h, 0.05)), ((0, -h), (h, 0.05)))
    for (x, y), (sx, sy) in edges:
      vis.add_box(np.array([x, y, 0.01]), np.array([sx, sy, 0.01]), np.eye(3), (1.0, 0.8, 0.0, 1.0))
    pos = self._pos().cpu().numpy()
    for xy, radius, color in ((pos[0], 0.4, (0.1, 0.4, 1.0, 0.8)), (pos[self.G], self.catch_radius, (1.0, 0.1, 0.1, 0.6))):
      vis.add_cylinder(np.array([xy[0], xy[1], 0.0]), np.array([xy[0], xy[1], 0.02]), radius, color)
    if self.camera == "chaser":
      self._follow_chaser(pos[self.G], pos[0])

  def _follow_chaser(self, chaser, runner, elevation=-35.0):
    """Put the video camera behind the chaser, looking toward the runner, and zoom out with the
    gap so the runner stays in view. It takes effect from the next frame (mjlab draws the
    overlay after placing the camera), and moves smoothly, except at the very first frame."""
    # mjlab has no public way to move the offscreen camera, so we set it directly.
    cam = self.mj.env.unwrapped._offline_renderer._cam
    gap = runner - chaser
    dist = float(np.hypot(gap[0], gap[1]))
    # Look at a point 40% of the way to the runner, from 6 m behind the chaser (more when far apart).
    want_distance = (0.6 * dist + 6.0) / math.cos(math.radians(elevation))
    want_lookat = np.array([chaser[0] + 0.4 * gap[0], chaser[1] + 0.4 * gap[1], 0.6])
    # Face the runner; when they nearly touch, the direction jumps around, so keep the old one.
    want_azimuth = math.degrees(math.atan2(gap[1], gap[0])) if dist > 1.0 else cam.azimuth
    k = 1.0 if not getattr(self, "_camera_placed", False) else 0.15  # snap the first time
    self._camera_placed = True
    cam.elevation = elevation
    cam.azimuth += k * ((want_azimuth - cam.azimuth + 180.0) % 360.0 - 180.0)  # shortest way round
    cam.distance += k * (want_distance - cam.distance)
    cam.lookat[:] = cam.lookat + k * (want_lookat - cam.lookat)

  def close(self):
    self.mj.close()
