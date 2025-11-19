"""
DSPPO extends the baseline PPO implementation with the structural concepts
outlined in the "Dissipative Structure PPO" design. The class adds hooks for
monitoring temporal-difference style errors, spawning specialist modules when
those errors persist, gating between base and specialist policies, and pruning
modules that remain inactive.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam

from ppo import PPO


class DSPPO(PPO):
        """
                PPO variant that supports adaptive structure changes inspired by
                dissipative structures.
        """

        def _init_hyperparameters(self, hyperparameters):
                super()._init_hyperparameters(hyperparameters)

                # Growth controls
                self.td_error_growth_threshold = getattr(self, "td_error_growth_threshold", 1.0)
                self.min_iterations_before_growth = getattr(self, "min_iterations_before_growth", 2)
                self.growth_window = getattr(self, "growth_window", 5)
                self.growth_cooldown = getattr(self, "growth_cooldown", 5)
                self.specialist_width = getattr(self, "specialist_width", 64)
                self.verbose_growth = getattr(self, "verbose_growth", True)

                # Growth state trackers
                self._recent_td_errors = deque(maxlen=self.growth_window)
                self._last_growth_iter = -float("inf")

                # Gating controls
                self.gating_hidden_dim = getattr(self, "gating_hidden_dim", 64)
                self.specialist_activation_threshold = getattr(self, "specialist_activation_threshold", 0.35)

                # Pruning controls
                self.prune_patience = getattr(self, "prune_patience", 10)
                self.min_iterations_before_prune = getattr(self, "min_iterations_before_prune", 3)
                self.activity_decay = getattr(self, "activity_decay", 0.9)

        def _init_actor_optimizer(self):
                if not hasattr(self, "specialist_modules"):
                        self.specialist_modules = nn.ModuleList()
                if not hasattr(self, "specialist_activity"):
                        self.specialist_activity = []
                if not hasattr(self, "gating_network"):
                        self.gating_network = nn.Sequential(
                                nn.Linear(self.obs_dim, self.gating_hidden_dim),
                                nn.ReLU(),
                                nn.Linear(self.gating_hidden_dim, 1),
                        )

                params = list(self.actor.parameters())
                params += list(self.gating_network.parameters())
                params += list(self.specialist_modules.parameters())
                return Adam(params, lr=self.lr)

        def _policy_mean(self, obs_tensor):
                base_mean = self.actor(obs_tensor)
                if len(self.specialist_modules) == 0:
                        return base_mean, None

                gating = torch.sigmoid(self.gating_network(obs_tensor))
                specialist_outputs = torch.stack([module(obs_tensor) for module in self.specialist_modules])
                specialist_mean = specialist_outputs.mean(dim=0)

                combined_mean = gating * base_mean + (1 - gating) * specialist_mean
                return combined_mean, gating

        def _update_specialist_activity(self, gating_values):
                if gating_values is None or len(self.specialist_modules) == 0:
                        return

                usage_signal = (1 - gating_values.detach()).mean().item()
                for idx in range(len(self.specialist_activity)):
                        if usage_signal > self.specialist_activation_threshold:
                                self.specialist_activity[idx] = 0
                        else:
                                self.specialist_activity[idx] = self.activity_decay * self.specialist_activity[idx] + 1

        def _monitor_td_error(self, batch_obs, td_errors):
                if td_errors.numel() == 0:
                        return None

                max_error = torch.max(torch.abs(td_errors)).item()
                self._recent_td_errors.append(max_error)

                # Require a few iterations of signal accumulation before reacting
                if len(self._recent_td_errors) < self.growth_window:
                        return None

                # Do not grow too early or too frequently
                if self.logger.get("i_so_far", 0) < self.min_iterations_before_growth:
                        return None
                if self.logger.get("i_so_far", 0) - self._last_growth_iter < self.growth_cooldown:
                        return None

                avg_error = np.mean(self._recent_td_errors)
                if avg_error < self.td_error_growth_threshold:
                        return None

                error_index = torch.argmax(torch.abs(td_errors)).item()
                state_of_interest = batch_obs[error_index].detach()
                self._trigger_growth(state_of_interest, max_error)

        def _trigger_growth(self, state_reference, magnitude):
                new_specialist = nn.Sequential(
                        nn.Linear(self.obs_dim, self.specialist_width),
                        nn.ReLU(),
                        nn.Linear(self.specialist_width, self.act_dim),
                )
                self.specialist_modules.append(new_specialist)
                self.specialist_activity.append(0)
                self._last_growth_iter = self.logger.get("i_so_far", 0)

                # refresh optimizer so new parameters receive gradients
                self.actor_optim = self._init_actor_optimizer()

                if self.verbose_growth:
                        print(f"[DSPPO] Spawned specialist #{len(self.specialist_modules)} for TD-error={magnitude:.3f}.")

        def _metabolize_structures(self):
                if len(self.specialist_modules) == 0:
                        return

                if self.logger.get('i_so_far', 0) < self.min_iterations_before_prune:
                        return

                pruned = False
                for idx in reversed(range(len(self.specialist_activity))):
                        if self.specialist_activity[idx] >= self.prune_patience:
                                del self.specialist_modules[idx]
                                del self.specialist_activity[idx]
                                pruned = True

                if pruned:
                        self.actor_optim = self._init_actor_optimizer()
                        if self.verbose_growth:
                                print(f"[DSPPO] Pruned inactive specialists. Remaining: {len(self.specialist_modules)}")

        def _prepare_obs(self, obs):
                if isinstance(obs, np.ndarray):
                        return torch.tensor(obs, dtype=torch.float)
                return obs

        def get_action(self, obs):
                obs_tensor = self._prepare_obs(obs)
                mean, gating_values = self._policy_mean(obs_tensor)
                dist = MultivariateNormal(mean, self.cov_mat)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                self._update_specialist_activity(gating_values)
                return action.detach().numpy(), log_prob.detach()

        def evaluate(self, batch_obs, batch_acts):
                V = self.critic(batch_obs).squeeze()
                mean, gating_values = self._policy_mean(batch_obs)
                dist = MultivariateNormal(mean, self.cov_mat)
                log_probs = dist.log_prob(batch_acts)
                self._update_specialist_activity(gating_values)
                return V, log_probs
