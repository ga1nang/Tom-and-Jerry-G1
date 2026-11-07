"""PPO written from scratch, for any vectorized environment.

The environment only has to provide:
  reset() -> (actor_obs, critic_obs)
  step(actions) -> (actor_obs, critic_obs, reward, terminated, truncated, info)
All values are torch tensors with a leading "num_envs" dimension, and the
environment resets finished episodes by itself (like mjlab and Isaac Lab do).

Read it top to bottom:
  1. RunningNorm      - normalizes observations to mean 0 / std 1
  2. ActorCritic      - the two neural networks
  3. collect_rollout  - let the robots act for T steps and record everything
  4. compute_gae      - turn rewards into "advantages" (was this action better than expected?)
  5. ppo_update       - improve the networks with the clipped PPO loss
  6. train            - the outer loop, logging and checkpoints
"""

import collections
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal


# --------------------------------------------------------------------------- #
# Hyperparameters (defaults = the values mjlab uses for the G1 velocity task)
# --------------------------------------------------------------------------- #
@dataclass
class PPOConfig:
  num_steps: int = 24  # T: steps each robot takes before every update
  num_epochs: int = 5  # passes over the collected data per update
  num_minibatches: int = 4
  gamma: float = 0.99  # discount: how much future reward counts
  lam: float = 0.95  # GAE lambda: bias/variance trade-off of the advantage
  clip: float = 0.2  # PPO clip range for the probability ratio
  value_coef: float = 1.0
  entropy_coef: float = 0.01  # bonus for keeping some randomness (exploration)
  lr: float = 1e-3
  desired_kl: float = 0.01  # adaptive learning rate keeps the policy change near this size
  max_grad_norm: float = 1.0
  hidden: tuple = (512, 256, 128)
  init_std: float = 1.0  # initial exploration noise of the actions
  save_every: int = 50


# --------------------------------------------------------------------------- #
# 1. Observation normalization
# --------------------------------------------------------------------------- #
class RunningNorm(nn.Module):
  """Keeps a running mean/variance of everything it has seen and normalizes with it.

  Joint velocities can be ~10, gravity components ~1, contact forces ~500.
  Networks learn much better when every input has a similar scale.
  """

  def __init__(self, dim):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(1e-4))

  @torch.no_grad()
  def update(self, x):
    batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    delta = batch_mean - self.mean
    total = self.count + n
    self.mean += delta * n / total
    self.var = (self.var * self.count + batch_var * n + delta**2 * self.count * n / total) / total
    self.count = total

  def forward(self, x):
    return (x - self.mean) / (self.var.sqrt() + 1e-2)


# --------------------------------------------------------------------------- #
# 2. The networks
# --------------------------------------------------------------------------- #
def mlp(inp, out, hidden):
  layers, d = [], inp
  for h in hidden:
    layers += [nn.Linear(d, h), nn.ELU()]
    d = h
  layers.append(nn.Linear(d, out))
  return nn.Sequential(*layers)


class ActorCritic(nn.Module):
  """Actor: observation -> Gaussian over actions.  Critic: observation -> expected future reward.

  The actor only sees what a real robot could measure. The critic is only used
  during training, so it may also see privileged simulator information
  ("asymmetric actor-critic").
  """

  def __init__(self, actor_dim, critic_dim, act_dim, hidden=(512, 256, 128), init_std=1.0):
    super().__init__()
    self.actor_norm = RunningNorm(actor_dim)
    self.critic_norm = RunningNorm(critic_dim)
    self.actor = mlp(actor_dim, act_dim, hidden)
    self.critic = mlp(critic_dim, 1, hidden)
    # One learnable noise level per joint. Starts high (explore), PPO shrinks it.
    self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))

  def dist(self, actor_obs):
    mean = self.actor(self.actor_norm(actor_obs))
    return Normal(mean, self.log_std.exp().expand_as(mean))

  def value(self, critic_obs):
    return self.critic(self.critic_norm(critic_obs)).squeeze(-1)

  @torch.no_grad()
  def act_deterministic(self, actor_obs):
    """For evaluation: use the mean action, no exploration noise."""
    return self.actor(self.actor_norm(actor_obs))


# --------------------------------------------------------------------------- #
# 3. Collect experience
# --------------------------------------------------------------------------- #
class EpisodeTracker:
  """Sums rewards per robot so we can report the return of finished episodes."""

  def __init__(self, n, device):
    self.ret = torch.zeros(n, device=device)
    self.len = torch.zeros(n, device=device)
    self.returns = collections.deque(maxlen=200)
    self.lengths = collections.deque(maxlen=200)

  def add(self, reward, done):
    self.ret += reward
    self.len += 1
    if done.any():
      self.returns.extend(self.ret[done].tolist())
      self.lengths.extend(self.len[done].tolist())
      self.ret[done] = 0
      self.len[done] = 0


@torch.no_grad()
def collect_rollout(env, ac, obs, cfg, tracker):
  actor_obs, critic_obs = obs
  T = cfg.num_steps
  buf = collections.defaultdict(list)
  infos = []
  for _ in range(T):
    ac.actor_norm.update(actor_obs)
    ac.critic_norm.update(critic_obs)
    dist = ac.dist(actor_obs)
    action = dist.sample()
    value = ac.value(critic_obs)

    buf["actor_obs"].append(actor_obs)
    buf["critic_obs"].append(critic_obs)
    buf["actions"].append(action)
    buf["log_probs"].append(dist.log_prob(action).sum(-1))
    buf["values"].append(value)
    buf["mu"].append(dist.mean)
    buf["sigma"].append(dist.stddev)

    actor_obs, critic_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated | truncated
    tracker.add(reward, done)
    # A timeout is not a failure: the robot would have kept collecting reward.
    # Add the critic's estimate of that missing future so the timeout is not punished.
    reward = reward + cfg.gamma * value * truncated.float()

    buf["rewards"].append(reward)
    buf["dones"].append(done.float())
    infos.append(info)

  batch = {k: torch.stack(v) for k, v in buf.items()}  # each: (T, num_envs, ...)
  batch["last_value"] = ac.value(critic_obs)
  return batch, (actor_obs, critic_obs), infos


# --------------------------------------------------------------------------- #
# 4. Generalized Advantage Estimation
# --------------------------------------------------------------------------- #
def compute_gae(rewards, values, dones, last_value, gamma, lam):
  """advantage_t = how much better things went than the critic predicted.

  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)       (one-step surprise)
  A_t     = delta_t + gamma * lam * A_{t+1}          (blend of future surprises)
  When an episode ends at t, nothing from t+1 is carried back (not_done = 0).
  """
  T = rewards.shape[0]
  advantages = torch.zeros_like(rewards)
  gae = torch.zeros_like(last_value)
  for t in reversed(range(T)):
    next_value = last_value if t == T - 1 else values[t + 1]
    not_done = 1.0 - dones[t]
    delta = rewards[t] + gamma * next_value * not_done - values[t]
    gae = delta + gamma * lam * not_done * gae
    advantages[t] = gae
  returns = advantages + values  # targets for the critic
  return advantages, returns


# --------------------------------------------------------------------------- #
# 5. PPO update
# --------------------------------------------------------------------------- #
def ppo_update(ac, optimizer, batch, cfg, state):
  advantages, returns = compute_gae(
    batch["rewards"], batch["values"], batch["dones"], batch["last_value"], cfg.gamma, cfg.lam
  )
  # Flatten (T, num_envs, ...) -> (T * num_envs, ...): every step is one training sample.
  flat = lambda x: x.reshape(-1, *x.shape[2:])
  data = {k: flat(v) for k, v in batch.items() if k not in ("rewards", "dones", "last_value")}
  data["advantages"] = flat(advantages)
  data["returns"] = flat(returns)
  adv = data["advantages"]
  data["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

  n = data["actions"].shape[0]
  mb_size = n // cfg.num_minibatches
  stats = collections.defaultdict(list)
  for _ in range(cfg.num_epochs):
    perm = torch.randperm(n, device=adv.device)
    for i in range(cfg.num_minibatches):
      idx = perm[i * mb_size : (i + 1) * mb_size]
      mb = {k: v[idx] for k, v in data.items()}

      dist = ac.dist(mb["actor_obs"])
      log_prob = dist.log_prob(mb["actions"]).sum(-1)
      entropy = dist.entropy().sum(-1).mean()
      value = ac.value(mb["critic_obs"])

      # Adaptive learning rate: if the policy moved too much (KL large), slow down.
      with torch.no_grad():
        mu, sigma = dist.mean, dist.stddev
        kl = (
          torch.log(sigma / mb["sigma"])
          + (mb["sigma"] ** 2 + (mb["mu"] - mu) ** 2) / (2 * sigma**2)
          - 0.5
        ).sum(-1).mean()
        if kl > 2 * cfg.desired_kl:
          state["lr"] = max(1e-5, state["lr"] / 1.5)
        elif 0 < kl < cfg.desired_kl / 2:
          state["lr"] = min(1e-2, state["lr"] * 1.5)
        for g in optimizer.param_groups:
          g["lr"] = state["lr"]

      # Clipped surrogate: increase probability of good actions, but never by more than +-20%.
      ratio = torch.exp(log_prob - mb["log_probs"])
      a = mb["advantages"]
      policy_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()

      # Critic regression toward the returns (also clipped, for stability).
      v_clipped = mb["values"] + (value - mb["values"]).clamp(-cfg.clip, cfg.clip)
      value_loss = torch.max((value - mb["returns"]) ** 2, (v_clipped - mb["returns"]) ** 2).mean()

      loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
      optimizer.step()

      stats["policy_loss"].append(policy_loss.item())
      stats["value_loss"].append(value_loss.item())
      stats["entropy"].append(entropy.item())
      stats["kl"].append(kl.item())
  return {k: sum(v) / len(v) for k, v in stats.items()}


# --------------------------------------------------------------------------- #
# 6. Training loop
# --------------------------------------------------------------------------- #
def save_checkpoint(path, ac, optimizer, iteration, cfg, dims, state):
  torch.save(
    {
      "model": ac.state_dict(),
      "optimizer": optimizer.state_dict(),
      "iteration": iteration,
      "cfg": asdict(cfg),
      "dims": dims,
      "lr": state["lr"],
    },
    path,
  )


def load_actor_critic(path, device):
  ckpt = torch.load(path, map_location=device)
  cfg = PPOConfig(**{**ckpt["cfg"], "hidden": tuple(ckpt["cfg"]["hidden"])})
  ac = ActorCritic(*ckpt["dims"], hidden=cfg.hidden, init_std=cfg.init_std).to(device)
  ac.load_state_dict(ckpt["model"])
  return ac, ckpt


def train(env, iterations, cfg=PPOConfig(), device="cpu", log_dir="runs/ppo", resume=None, log_keys=(), seed=0):
  torch.manual_seed(seed)
  os.makedirs(log_dir, exist_ok=True)
  obs = env.reset()
  dims = (obs[0].shape[-1], obs[1].shape[-1], env.num_actions)
  num_envs = obs[0].shape[0]

  ac = ActorCritic(*dims, hidden=cfg.hidden, init_std=cfg.init_std).to(device)
  optimizer = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
  state = {"lr": cfg.lr}
  start = 0
  if resume:
    ckpt = torch.load(resume, map_location=device)
    ac.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    state["lr"] = ckpt["lr"]
    start = ckpt["iteration"] + 1
    print(f"resumed from {resume} at iteration {start}")

  try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
  except Exception:
    writer = None

  tracker = EpisodeTracker(num_envs, device)
  history = []
  t_start = time.time()
  for it in range(start, start + iterations):
    t0 = time.time()
    batch, obs, infos = collect_rollout(env, ac, obs, cfg, tracker)
    t_collect = time.time() - t0
    losses = ppo_update(ac, optimizer, batch, cfg, state)
    t_iter = time.time() - t0

    row = {
      "iteration": it,
      "mean_return": sum(tracker.returns) / max(len(tracker.returns), 1),
      "mean_episode_length": sum(tracker.lengths) / max(len(tracker.lengths), 1),
      "action_std": ac.log_std.exp().mean().item(),
      "lr": state["lr"],
      "steps_per_s": cfg.num_steps * num_envs / t_iter,
      **losses,
    }
    # Environment-specific metrics (e.g. how often robots fell), averaged over the rollout.
    for key in log_keys:
      vals = [float(i[key]) for i in infos if key in i]
      if vals:
        row[key] = sum(vals) / len(vals)
    history.append(row)

    if writer:
      for k, v in row.items():
        if k != "iteration":
          writer.add_scalar(k, v, it)
    done_its = it - start + 1
    eta = (time.time() - t_start) / done_its * (iterations - done_its)
    extra = " | ".join(f"{k.split('/')[-1]} {row[k]:.3f}" for k in log_keys if k in row)
    print(
      f"it {it:5d} | return {row['mean_return']:8.2f} | ep len {row['mean_episode_length']:6.1f} | "
      f"std {row['action_std']:.2f} | lr {state['lr']:.1e} | {extra} | "
      f"{t_iter:.2f}s/it (collect {t_collect:.2f}s) | ETA {eta / 60:.1f} min",
      flush=True,
    )
    if (it + 1) % cfg.save_every == 0 or it == start + iterations - 1:
      save_checkpoint(os.path.join(log_dir, f"model_{it}.pt"), ac, optimizer, it, cfg, dims, state)

  if writer:
    writer.close()
  return ac, history
