"""
TD3: Twin Delayed Deep Deterministic Policy Gradient
Fujimoto, van Hoof, Meger — "Addressing Function Approximation Error in Actor-Critic Methods" (ICML 2018)

TD3 improves on DDPG by addressing the overestimation bias in Q-learning.
Three key tricks:
  1. Twin critics (clipped double Q-learning): take the min of two Q-networks for targets
  2. Delayed policy updates: update the actor less frequently than the critics
  3. Target policy smoothing: add noise to target actions to prevent overfitting to peaks

This is a from-scratch PyTorch implementation. I wrote it by going through
the TD3 paper section by section.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from src.common.networks import DeterministicActor, QNetwork
from src.common.replay_buffer import ReplayBuffer


class TD3:
    """
    TD3 agent.

    Parameters are set from a config dict so we can sweep them easily.
    """

    def __init__(self, state_dim, action_dim, max_action, config, device):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device     = device

        # ── Hyperparameters ──────────────────────────────────────────────────
        hidden_dims         = config["hidden_dims"]
        lr                  = config["learning_rate"]
        self.gamma          = config["gamma"]
        self.tau            = config["tau"]
        self.batch_size     = config["batch_size"]
        self.explore_noise  = config["exploration_noise"]
        self.target_noise   = config["target_noise"]
        self.noise_clip     = config["target_noise_clip"]
        self.policy_delay   = config["policy_delay"]
        self.learn_starts   = config["learning_starts"]

        # ── Actor: one network + one target ──────────────────────────────────
        self.actor        = DeterministicActor(state_dim, action_dim, hidden_dims, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt    = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # ── Critics: twin Q-networks, each with its own target ───────────────
        # Using two completely separate networks (not one network with two heads)
        # as in the original paper
        self.critic1        = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic2        = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        # Both critics share one optimizer — convenient and equivalent to two separate ones
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

        # ── Replay buffer ────────────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, config["buffer_size"], device)

        # Counter for delayed policy updates
        self.total_updates = 0

        # For logging — we accumulate losses and return them to the training loop
        self._last_critic_loss = 0.0
        self._last_actor_loss  = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Action selection
    # ─────────────────────────────────────────────────────────────────────────

    def select_action(self, state, evaluate=False):
        """
        Select an action for a given state.

        During training (evaluate=False): add Gaussian exploration noise.
        During evaluation (evaluate=True): pure deterministic action, no noise.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()

        if not evaluate:
            # Add Gaussian noise and clip to valid action range
            noise = np.random.normal(0, self.max_action * self.explore_noise, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)

        return action

    # ─────────────────────────────────────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────────────────────────────────────

    def train_step(self):
        """
        One gradient update step. Called once per environment step after
        learning_starts transitions have been collected.

        Returns (critic_loss, actor_loss) — actor_loss is None if we skipped
        the actor update due to policy delay.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        batch = self.replay_buffer.sample(self.batch_size)
        s  = batch["states"]
        a  = batch["actions"]
        r  = batch["rewards"]
        s_ = batch["next_states"]
        d  = batch["dones"]

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to the target action
            # This prevents the policy from exploiting sharp Q-function peaks
            noise = (
                torch.randn_like(a) * self.target_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            # Next action from TARGET actor (smoothed)
            a_next = (self.actor_target(s_) + noise).clamp(-self.max_action, self.max_action)

            # Clipped double Q: take the min of the two target critics
            # This is the key trick that prevents overestimation
            q1_target = self.critic1_target(s_, a_next)
            q2_target = self.critic2_target(s_, a_next)
            q_target  = torch.min(q1_target, q2_target)

            # Bellman backup
            y = r + (1.0 - d) * self.gamma * q_target

        # Current Q-values from both critics
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)

        # MSE loss for both critics together
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.total_updates += 1
        self._last_critic_loss = critic_loss.item()
        actor_loss_val = None

        # ── Actor update (delayed) ────────────────────────────────────────────
        # Only update the actor every policy_delay critic updates.
        # This gives the critics time to stabilize before the actor responds.
        if self.total_updates % self.policy_delay == 0:
            # Actor loss: maximize Q1(s, pi(s))
            # We only use Q1 here (not the min) — the original paper does this
            actor_loss = -self.critic1(s, self.actor(s)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            actor_loss_val = actor_loss.item()
            self._last_actor_loss = actor_loss_val

            # ── Soft update of target networks (Polyak averaging) ─────────────
            # theta_target = tau * theta + (1 - tau) * theta_target
            # Small tau means target networks change slowly — stabilizes training
            self._polyak_update(self.actor, self.actor_target)
            self._polyak_update(self.critic1, self.critic1_target)
            self._polyak_update(self.critic2, self.critic2_target)

        return self._last_critic_loss, actor_loss_val

    def _polyak_update(self, source, target):
        """Soft-update target network parameters towards source network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path):
        """Save all network weights to a .pt file."""
        torch.save({
            "actor":          self.actor.state_dict(),
            "actor_target":   self.actor_target.state_dict(),
            "critic1":        self.critic1.state_dict(),
            "critic2":        self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
        }, path)

    def load(self, path):
        """Load network weights from a saved .pt file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
