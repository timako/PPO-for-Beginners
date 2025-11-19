# Dissipative Structure PPO (DS-PPO)

This repository now includes an experimental `DSPPO` agent that mirrors the
theoretical ideas you shared:

1. **Dynamic growth (涨落 → 增殖)**: The trainer monitors batch TD-style errors.
   When a state slice exhibits persistent large errors, DS-PPO spawns a new
   specialist module to focus on that region.
2. **Modularization & specialization (技能分化)**: A learnable gating network
   blends the base policy with specialists, enabling skill selection instead of
   overwriting older knowledge.
3. **Pruning & metabolism (新陈代谢)**: Specialists accumulate an inactivity
   counter and are pruned after sustained disuse, keeping the model lean.

## How it works

- The class lives in [`ds_ppo.py`](ds_ppo.py) and subclasses the baseline
  [`PPO`](ppo.py) implementation.
- Growth triggers inside `_monitor_td_error`, called once per PPO iteration
  (see the hook in `PPO.learn`). The trainer accumulates recent TD-error maxima
  and only grows when the rolling average stays above
  `td_error_growth_threshold` after a short warmup and cooldown window. When a
  trigger fires, a new `nn.Sequential` specialist is appended and the actor
  optimizer refreshes so its parameters are trained.
- Gating: `_policy_mean` mixes the base actor output with the mean output of all
  specialists using a sigmoid gating network. The gating parameters train along
  with the actor because the optimizer includes them.
- Pruning: `_metabolize_structures` runs once per iteration. After
  `min_iterations_before_prune` iterations, any specialist whose inactivity
  counter exceeds `prune_patience` is removed and the optimizer is refreshed.

## Key hyperparameters

- `td_error_growth_threshold` (default `1.0`): Rolling-average TD-error needed
  to spawn a specialist.
- `growth_window` (default `5`): Number of recent iterations used to smooth the
  TD-error signal.
- `min_iterations_before_growth` (default `2`): Warmup iterations before any
  growth is allowed.
- `growth_cooldown` (default `5`): Minimum iterations to wait between growth
  events to reduce early runaway spawning.
- `specialist_width` (default `64`): Hidden size of each specialist module.
- `gating_hidden_dim` (default `64`): Width of the gating network.
- `specialist_activation_threshold` (default `0.35`): Activity signal required
  to reset a specialist's inactivity counter.
- `prune_patience` (default `10`) & `min_iterations_before_prune` (default `3`):
  Control how long a specialist can remain idle before removal.
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
