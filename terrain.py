"""A patchwork arena for the runner vs chaser game: normal, grass, mud and rocky ground.

The arena is a 5 x 5 grid of square cells (4 m for the 20 m arena). Every game gets a new
random layout with 8 normal, 6 grass, 6 mud and 5 rocky cells (32/24/24/20%), and no two
side-by-side cells of the same type. Each terrain type does two things:
  - physics: grip, softness, drag in mud, solid rocks (table below);
  - a game rule: the forward command is capped while the robot is on it (SPEED_CAP).

How it's built on mjlab / MuJoCo Warp:
  - The robots still walk on mjlab's ground plane. Each world holds one robot, so instead
    of separate ground pieces (whose seams trip the feet), every world's plane takes the
    grip and softness of the terrain under its robot, updated every 0.02 s (step_effects).
  - The G1's foot geoms have contact priority 1: their own friction (randomized 0.3-1.2 per
    robot by mjlab) beats the ground's. The plane gets priority 2, so its values win. On
    normal and rocky ground it copies the robot's own foot friction and softness, which is
    exactly the flat arena.
  - The colored tiles are visual only (no collisions). The rocks are solid boxes, each on
    its own mocap body: MuJoCo Warp computes the position of geoms fixed to the world only
    once, but moves mocap bodies every step (data.mocap_pos), per world.
  - MuJoCo Warp shares one model between all worlds, except for fields that are "expanded"
    to one copy per world (FIELDS). Plane grip/softness, tile colors and the rocks' default
    poses (body_pos/quat, which a world reset puts the mocap bodies back to) are expanded, so
    every world can have its own layout without rebuilding anything. The caller writes the
    same layout into the two worlds of a game.
"""

import numpy as np
import torch

import mujoco

NORMAL, GRASS, MUD, ROCKY = range(4)
TERRAINS = ("normal", "grass", "mud", "rocky")
CELLS = 5  # cells per side
CELL_COUNTS = (8, 6, 6, 5)  # cells of each type in a layout (sums to CELLS²)

# Per terrain type: friction (None = the robot's own foot friction), contact softness as
# MuJoCo's solref (time constant s, damping ratio) and solimp (impedance from dmin to dmax
# over `width` m of penetration; lower impedance and a longer time constant = the feet sink
# in deeper), the game's forward speed cap (m/s), and color.
# Measured with the g1_turn legs (EXPERIMENTS.md): standing, the feet sink about 0.4 cm into
# grass and 1 cm into mud. At mud's 2 m/s cap nobody falls; at 5 m/s most robots do, and
# grip below 0.4 makes that worse.
FRICTION = (None, 0.5, 0.4, None)
SOLREF = ((0.02, 1.0), (0.06, 1.0), (0.10, 1.0), (0.02, 1.0))
SOLIMP = ((0.9, 0.95, 0.001, 0.5, 2.0), (0.5, 0.6, 0.02, 0.5, 2.0),
          (0.3, 0.4, 0.05, 0.5, 2.0), (0.9, 0.95, 0.001, 0.5, 2.0))
SPEED_CAP = (5.0, 4.0, 2.0, 3.0)
COLOR = ((0.36, 0.38, 0.43, 1.0), (0.20, 0.42, 0.14, 1.0), (0.30, 0.18, 0.09, 1.0), (0.55, 0.50, 0.41, 1.0))
MUD_DRAG = 0.5  # 1/s: in mud, the pelvis is pulled back by MUD_DRAG × robot mass × velocity

ROCKS_PER_CELL = 5
ROCK_COLOR = (0.10, 0.09, 0.09, 1.0)
ROCK_MARGIN = 0.6  # rock centres stay this far inside their cell (m), so no rock pokes out of it
TILE_HALF_THICKNESS = 0.001  # visual tiles: 2 mm thick, resting on the plane
PARKED = -5.0  # z of rocks that aren't used (buried under the plane, never touched)
FIELDS = ("geom_friction", "geom_solref", "geom_solimp", "geom_rgba", "body_pos", "body_quat")


def make_layouts(n, rng):
  """n random layouts (n, CELLS²) of terrain types, with no two side-by-side cells alike."""
  layouts = np.empty((n, CELLS * CELLS), dtype=np.int64)
  for k in range(n):
    grid, left = [-1] * (CELLS * CELLS), list(CELL_COUNTS)

    def fill(i):  # randomized backtracking, cell by cell in reading order
      if i == len(grid):
        return True
      row, col = divmod(i, CELLS)
      for t in rng.permutation(len(TERRAINS)):
        if left[t] and (col == 0 or grid[i - 1] != t) and (row == 0 or grid[i - CELLS] != t):
          grid[i], left[t] = t, left[t] - 1
          if fill(i + 1):
            return True
          left[t] += 1
      grid[i] = -1
      return False

    assert fill(0)
    layouts[k] = grid
  return layouts


class TerrainArena:
  """Build with spec_fn (before mjlab compiles the scene), then bind() to the built env."""

  def __init__(self, arena, device, num_layouts=4096, seed=0):
    self.half, self.cell, self.device = arena / 2, arena / CELLS, device
    rng = np.random.default_rng(seed)
    self.layouts = torch.as_tensor(make_layouts(num_layouts, rng), device=device)
    # Rock half-sizes: footprint 0.3-0.8 m, 0.1-0.4 m tall. Fixed per rock; positions change.
    self.num_rocks = CELL_COUNTS[ROCKY] * ROCKS_PER_CELL
    self.rock_size = np.stack([rng.uniform(0.15, 0.4, self.num_rocks), rng.uniform(0.15, 0.4, self.num_rocks),
                               rng.uniform(0.05, 0.2, self.num_rocks)], axis=1)

  # ------------------------------------------------------------------------- #
  # Geometry (runs once, inside mjlab's scene build)
  # ------------------------------------------------------------------------- #
  def spec_fn(self, spec):
    box = mujoco.mjtGeom.mjGEOM_BOX
    body = spec.body("terrain")
    spec.geom("terrain").priority = 2  # its grip/softness beat the feet's (see the docstring)
    for i in range(CELLS * CELLS):  # visual only: contype = conaffinity = 0
      row, col = divmod(i, CELLS)
      body.add_geom(
        name=f"tile_{i}", type=box, size=[self.cell / 2, self.cell / 2, TILE_HALF_THICKNESS],
        pos=[-self.half + (col + 0.5) * self.cell, -self.half + (row + 0.5) * self.cell, TILE_HALF_THICKNESS],
        contype=0, conaffinity=0, rgba=COLOR[NORMAL],
      )
    for k, size in enumerate(self.rock_size):  # mocap bodies must hang off the world body
      rock = spec.worldbody.add_body(name=f"rock_{k}", mocap=True, pos=[0.0, 0.0, PARKED])
      rock.add_geom(name=f"rock_{k}", type=box, size=size.tolist(),
                    priority=2, condim=3, friction=[0.8, 0.005, 0.0001], rgba=ROCK_COLOR)

  def bind(self, mj):
    """Look up the geoms and bodies once the env exists (MjlabEnv built with expand_fields=FIELDS)."""
    sim, dev = mj.env.unwrapped.sim, self.device
    m = sim.mj_model
    self.model, self.data = sim.model, sim.data
    self.robot = mj.robot
    self.tile_ids = torch.tensor([m.geom(f"tile_{i}").id for i in range(CELLS * CELLS)], device=dev)
    self.rock_ids = torch.tensor([m.geom(f"rock_{k}").id for k in range(self.num_rocks)], device=dev)
    rock_bodies = [m.body(f"rock_{k}").id for k in range(self.num_rocks)]
    self.rock_bodies = torch.tensor(rock_bodies, device=dev)
    self.rock_mocap = torch.tensor(m.body_mocapid[rock_bodies], device=dev)
    self.plane = m.geom("terrain").id
    self.foot_geom = m.geom("robot/left_foot1_collision").id  # all foot geoms share one friction
    self.rock_half_height = torch.as_tensor(self.rock_size[:, 2], dtype=torch.float, device=dev)
    self.pelvis = self.robot.find_bodies("pelvis")[0]
    self.mass = float(m.body_subtreemass[m.body("robot/pelvis").id])
    self.cells = torch.full((mj.num_envs, CELLS * CELLS), NORMAL, dtype=torch.long, device=dev)
    self.friction = torch.tensor([f if f is not None else 0.0 for f in FRICTION], device=dev)
    self.own_grip = torch.tensor([f is None for f in FRICTION], device=dev)
    self.solref = torch.tensor(SOLREF, device=dev)
    self.solimp = torch.tensor(SOLIMP, device=dev)
    self.color = torch.tensor(COLOR, device=dev)
    self.speed_caps = torch.tensor(SPEED_CAP, device=dev)

  # ------------------------------------------------------------------------- #
  # Layouts
  # ------------------------------------------------------------------------- #
  def random_layouts(self, n):
    """n layouts from the bank, plus random rock positions (n, rocks, 2) and yaws inside their rocky cells."""
    cells = self.layouts[torch.randint(len(self.layouts), (n,), device=self.device)]
    # Indices of the rocky cells of each layout (every layout has the same number of them).
    rocky = torch.argsort((cells != ROCKY).int(), dim=1, stable=True)[:, :CELL_COUNTS[ROCKY]]
    centre = self.cell_centre(rocky.repeat_interleave(ROCKS_PER_CELL, dim=1))
    spread = self.cell / 2 - ROCK_MARGIN
    xy = centre + (torch.rand(n, self.num_rocks, 2, device=self.device) * 2 - 1) * spread
    return cells, xy, torch.rand(n, self.num_rocks, device=self.device) * 2 * torch.pi

  def cell_centre(self, idx):
    row, col = idx // CELLS, idx % CELLS
    return torch.stack([-self.half + (col + 0.5) * self.cell, -self.half + (row + 0.5) * self.cell], dim=-1)

  def set_cells(self, worlds, cells):
    """Give worlds (n,) the terrain types cells (n, CELLS²): the layout and the tile colors."""
    self.model.geom_rgba[worlds[:, None], self.tile_ids[None, :]] = self.color[cells]
    self.cells[worlds] = cells

  def place_rocks(self, worlds, xy, yaw):
    """Put the rocks of worlds (n,) at xy (n, rocks, 2) with yaw (n, rocks); NaN xy = not used."""
    w = worlds[:, None]
    used = ~xy[..., 0].isnan()
    z = torch.where(used, 0.5 * self.rock_half_height, torch.full_like(xy[..., 0], PARKED))
    pos = torch.cat([xy.nan_to_num(0.0), z[..., None]], dim=-1)
    quat = torch.zeros(*yaw.shape, 4, device=self.device)
    quat[..., 0], quat[..., 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    self.data.mocap_pos[w, self.rock_mocap[None, :]] = pos  # where they are now
    self.data.mocap_quat[w, self.rock_mocap[None, :]] = quat
    self.model.body_pos[w, self.rock_bodies[None, :]] = pos  # where a world reset puts them
    self.model.body_quat[w, self.rock_bodies[None, :]] = quat

  # ------------------------------------------------------------------------- #
  # Queries and effects
  # ------------------------------------------------------------------------- #
  def type_at(self, worlds, xy):
    """Terrain type under points xy (n, 2) of worlds (n,); outside the arena counts as normal."""
    col = ((xy[:, 0] + self.half) / self.cell).floor().long()
    row = ((xy[:, 1] + self.half) / self.cell).floor().long()
    inside = (col >= 0) & (col < CELLS) & (row >= 0) & (row < CELLS)
    idx = (row.clamp(0, CELLS - 1) * CELLS + col.clamp(0, CELLS - 1))
    return torch.where(inside, self.cells[worlds, idx], NORMAL)

  def under_robots(self):
    """Terrain type under every robot's pelvis."""
    worlds = torch.arange(self.cells.shape[0], device=self.device)
    return self.type_at(worlds, self.robot.data.root_link_pos_w[:, :2])

  def speed_cap(self, types):
    return self.speed_caps[types]

  def step_effects(self):
    """Call before every 0.02 s step: the ground under each robot sets its world's plane grip
    and softness, and robots in mud get pulled back by the drag."""
    types = self.under_robots()
    own = self.model.geom_friction[:, self.foot_geom, 0]  # each robot's own grip
    self.model.geom_friction[:, self.plane, 0] = torch.where(self.own_grip[types], own, self.friction[types])
    self.model.geom_solref[:, self.plane] = self.solref[types]
    self.model.geom_solimp[:, self.plane] = self.solimp[types]
    vel = self.robot.data.root_link_lin_vel_w.clone()
    vel[:, 2] = 0.0
    force = (-MUD_DRAG * self.mass * (types == MUD).float()[:, None] * vel)[:, None, :]
    self.robot.write_external_wrench_to_sim(force, torch.zeros_like(force), body_ids=self.pelvis)
    return types
