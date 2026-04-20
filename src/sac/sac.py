"""
SAC: Soft Actor-Critic
Haarnoja, Zhou, Abbeel, Levine — "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor" (ICML 2018)
+ auto-tuning from: "Soft Actor-Critic Algorithms and Applications" (2018)

SAC optimizes a maximum entropy objective:
    J(pi) = E[sum_t (r_t + alpha * H(pi(.|s_t)))]

The entropy term alpha * H(pi) encourages the policy to be as random as possible
while still getting high reward. This gives better exploration and robustness.

Three key components different from TD3:
  1. Stochastic policy with reparameterization + tanh squashing (not deterministic)
  2. Entropy term in the Bellman backup (soft Bellman equation)
  3. Auto-tuned temperature alpha — learned alongside the policy

I also use twin critics (like TD3) to avoid overestimation.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from src.common.networks import SquashedGaussianActor, QNetwork
from src.common.replay_buffer import ReplayBuffer


class SAC:
    """
    SAC agent with automatic entropy tuning.
    """

    def __init__(self, state_dim, action_dim, max_action, config, device):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device     = device

        # ── Hyperparameters ──────────────────────────────────────────────────
        hidden_dims       = config["hidden_dims"]
        lr                = config["learning_rate"]
        self.gamma        = config["gamma"]
        self.tau          = config["tau"]
        self.batch_size   = config["batch_size"]
        self.learn_starts = config["learning_starts"]
        self.auto_tune    = config.get("auto_tune_alpha", True)

        # ── Actor ─────────────────────────────────────────────────────────────
        self.actor     = SquashedGaussianActor(state_dim, action_dim, hidden_dims, max_action).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # ── Twin critics + their targets ──────────────────────────────────────
        # SAC only has TARGET critics (no target actor — we use the current policy for next actions)
        self.critic1        = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic2        = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

        # ── Entropy temperature alpha ──────────────────────────────────────────
        # The target entropy is -dim(A) — this is a common heuristic that works well.
        # Intuitively, we want the policy to have at least 1 nat of entropy per action dimension.
        self.target_entropy = -float(action_dim)

        if self.auto_tune:
            # log_alpha is learned; alpha = exp(log_alpha) (keeps alpha > 0)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            # Fixed alpha — use whatever initial_alpha says
            self.alpha = float(config.get("initial_alpha", 0.2))
            self.log_alpha = None
            self.alpha_opt = None

        # ── Replay buffer ────────────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, config["buffer_size"], device)

        # For logging
        self._last_critic_loss = 0.0
        self._last_actor_loss  = 0.0
        self._last_alpha_loss  = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Action selection
    # ─────────────────────────────────────────────────────────────────────────

    def select_action(self, state, evaluate=False):
        """
        Sample or compute an action for the given state.

        During training: sample stochastically from the policy (exploration is
        built-in via the stochastic policy — no separate noise needed like TD3).
        During eval: use the deterministic mean action (no sampling).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                action = self.actor.get_action(state_tensor)
            else:
                action, _ = self.actor(state_tensor)

        return action.cpu().numpy().flatten()

    # ─────────────────────────────────────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────────────────────────────────────

    def train_step(self):
        """
        One full SAC gradient update (critic + actor + alpha).

        Returns (critic_loss, actor_loss, alpha_loss, alpha)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None, None

        batch = self.replay_buffer.sample(self.batch_size)
        s  = batch["states"]
        a  = batch["actions"]
        r  = batch["rewards"]
        s_ = batch["next_states"]
        d  = batch["dones"]

        # ── Critic update (soft Bellman backup) ───────────────────────────────
        with torch.no_grad():
            # Sample next action from CURRENT policy (not a target actor)
            a_next, log_prob_next = self.actor(s_)

            # Soft Bellman target: r + gamma * (min_Q(s', a') - alpha * log pi(a'|s'))
            # The entropy term alpha * log pi appears because we're maximizing
            # the entropy-augmented return
            q1_next = self.critic1_target(s_, a_next)
            q2_next = self.critic2_target(s_, a_next)
            min_q_next = torch.min(q1_next, q2_next)

            y = r + (1.0 - d) * self.gamma * (min_q_next - self.alpha * log_prob_next)

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor update ─────────────────────────────────────────────────────
        # Sample fresh actions from current policy (can't reuse batch actions —
        # those came from a possibly different policy version)
        a_new, log_prob_new = self.actor(s)

        # Actor loss: maximize E[Q - alpha * log pi]
        # Equivalently: minimize E[alpha * log pi - Q]
        q1_new = self.critic1(s, a_new)
        q2_new = self.critic2(s, a_new)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob_new - min_q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Alpha (temperature) update ────────────────────────────────────────
        alpha_loss_val = 0.0
        if self.auto_tune:
            # We want: H(pi) >= target_entropy
            # alpha loss: minimize -alpha * (log pi + target_entropy)
            # When H(pi) < target_entropy, log pi is too large (not enough entropy),
            # so alpha increases to push the policy toward more exploration.
            # When H(pi) > target_entropy, alpha decreases to focus on reward.
            alpha_loss = -(
                self.log_alpha.exp() * (log_prob_new + self.target_entropy).detach()
            ).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # Update the cached alpha value
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()

        # ── Soft update target critics ────────────────────────────────────────
        self._polyak_update(self.critic1, self.critic1_target)
        self._polyak_update(self.critic2, self.critic2_target)

        self._last_critic_loss = critic_loss.item()
        self._last_actor_loss  = actor_loss.item()
        self._last_alpha_loss  = alpha_loss_val

        return (
            self._last_critic_loss,
            self._last_actor_loss,
            self._last_alpha_loss,
            self.alpha,
        )

    def _polyak_update(self, source, target):
        """Soft-update: target = tau * source + (1-tau) * target"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path):
        checkpoint = {
            "actor":          self.actor.state_dict(),
            "critic1":        self.critic1.state_dict(),
            "critic2":        self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
        }
        if self.auto_tune:
            checkpoint["log_alpha"] = self.log_alpha.data
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        if self.auto_tune and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            self.alpha = self.log_alpha.exp().item()
