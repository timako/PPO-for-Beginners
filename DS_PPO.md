# Dissipative Structure PPO (DS-PPO)

This repository now includes an experimental `DSPPO` agent that mirrors the
theoretical ideas you shared:

1. **Dynamic growth (涨落 → 增殖)**: The trainer monitors batch TD-style errors.
   When a state slice exhibits persistent large errors, DS-PPO spawns a new
   specialist module to focus on that region.
2. **Modularization & specialization (技能分化)**: A learnable gating network
  now produces a softmax mixture over the base policy and every specialist
  module, letting the agent route individual observations to the most relevant
  expert instead of averaging them together.
3. **Pruning & metabolism (新陈代谢)**: Specialists accumulate an inactivity
   counter and are pruned after sustained disuse, keeping the model lean.

## How it works

- The class lives in [`ds_ppo.py`](ds_ppo.py) and subclasses the baseline
  [`PPO`](ppo.py) implementation.
- Growth triggers inside `_monitor_td_error`, called once per PPO iteration
  (see the hook in `PPO.learn`). The trainer accumulates recent TD-error maxima
  and only grows when the rolling average stays above
  `td_error_growth_threshold` after a short warmup and cooldown window. Late
  training plateaus are detected through the moving average episodic returns;
  when the reward window stabilizes within `return_plateau_tolerance`, growth is
  automatically paused. A hard `max_specialists` limit prevents unbounded
  structures. When a trigger fires, a new `nn.Sequential` specialist is appended
  along with its own gating head and the actor optimizer refreshes so all
  parameters keep receiving gradients.
- Gating: `_policy_mean` mixes the base actor output with a softmax-weighted sum
  of the specialist outputs. Each specialist owns a tiny gating head so activity
  can be tracked per expert instead of collectively.
- Pruning: `_metabolize_structures` runs once per iteration. After
  `min_iterations_before_prune` iterations, any specialist whose inactivity
  counter exceeds `prune_patience` is removed, together with its gating head.
  The pruning logic includes a `prune_cooldown` so freshly added experts are not
  immediately reaped.

## Key hyperparameters

- `td_error_growth_threshold` (default `1.25`): Rolling-average TD-error needed
  to spawn a specialist.
- `growth_window` (default `5`): Number of recent iterations used to smooth the
  TD-error signal.
- `min_iterations_before_growth` (default `2`): Warmup iterations before any
  growth is allowed.
- `growth_cooldown` (default `5`): Minimum iterations to wait between growth
  events to reduce early runaway spawning.
- `specialist_width` (default `64`): Hidden size of each specialist module.
- `max_specialists` (default `32`): Hard cap on simultaneously active
  specialists.
- `return_plateau_window` (default `5`), `return_plateau_tolerance` (default
  `5.0`), `return_plateau_min_return` (default `-350`): define the reward window
  that, once stable, pauses further growth so late-stage training does not keep
  spawning experts.
- `gating_hidden_dim` (default `64`): Width of the gating network.
- `specialist_activation_threshold` (default `0.35`): Activity signal required
  to reset a specialist's inactivity counter.
- `prune_patience` (default `10`), `min_iterations_before_prune` (default `3`),
  `prune_cooldown` (default `3`): Control how long a specialist can remain idle
  before removal and how frequently pruning may occur.
- `verbose_growth` (default `True`): Emit console logs on growth and pruning
  events.

## Usage

```python
from ds_ppo import DSPPO
from network import FeedForwardNN
import gymnasium as gym

env = gym.make("LunarLanderContinuous-v2")
agent = DSPPO(FeedForwardNN, env, td_error_growth_threshold=0.8)
agent.learn(total_timesteps=200_000)
```

You can tune the above hyperparameters when instantiating `DSPPO` to emphasize
faster growth, more aggressive pruning, or quieter logging.
