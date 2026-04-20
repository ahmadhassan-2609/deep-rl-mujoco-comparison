"""
PPO: Proximal Policy Optimization for continuous action spaces.
Reference: Schulman et al. 2017  "Proximal Policy Optimization Algorithms"
           CleanRL ppo_continuous_action.py (correctness reference)

On-policy: collect a rollout → compute GAE → run K epochs of updates → discard data.

The clipped surrogate objective prevents large policy updates:
    L_CLIP = E[ min(r·Â,  clip(r, 1-ε, 1+ε)·Â) ]
where r = π_new(a|s) / π_old(a|s).

The value function loss is also clipped (CleanRL style) to limit how far the
critic can move from its old predictions in a single update step.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.common.networks import GaussianActor, ValueNetwork
from src.common.rollout_buffer import RolloutBuffer


class PPO:
    """PPO agent for continuous action spaces."""

    def __init__(self, state_dim, action_dim, config, device):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = device

        # ── Hyperparameters ───────────────────────────────────────────────────
        hidden_dims         = config["hidden_dims"]
        lr                  = config["learning_rate"]
        self.gamma          = config["gamma"]
        self.gae_lambda     = config["gae_lambda"]
        self.clip_range     = config["clip_range"]
        self.n_epochs       = config["n_epochs"]
        self.batch_size     = config["batch_size"]
        self.rollout_length = config["rollout_length"]
        self.vf_coef        = config["vf_coef"]
        self.ent_coef       = config["ent_coef"]
        self.max_grad_norm  = config["max_grad_norm"]

        # Optional features (safe defaults if not in config)
        self.clip_vloss    = config.get("clip_vloss", True)   # clipped value loss (CleanRL default)
        self.norm_adv      = config.get("norm_adv", True)     # per-mini-batch advantage normalisation
        self.anneal_lr     = config.get("anneal_lr", False)   # linear LR decay to 0
        self.target_kl     = config.get("target_kl", None)    # early stopping threshold (None = off)
        self.total_timesteps = config.get("total_timesteps", None)  # needed for LR annealing

        self._base_lr      = lr
        self._global_steps = 0   # total env steps collected (updated in update())

        # ── Networks ──────────────────────────────────────────────────────────
        # Separate actor and critic — no shared weights (simpler, standard for MuJoCo PPO).
        # GaussianActor: Tanh activations + orthogonal init (hidden gain √2, output gain 0.01).
        # ValueNetwork:  Tanh activations + orthogonal init (hidden gain √2, output gain 1.0).
        self.actor  = GaussianActor(state_dim, action_dim, hidden_dims).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dims).to(device)

        # Single combined optimizer — actor and critic gradients are the same order
        # of magnitude with NormalizeReward, so one Adam accumulator is fine and
        # avoids momentum drift from two independent ones.  eps=1e-5 matches CleanRL.
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, eps=1e-5,
        )

        # ── Rollout buffer ─────────────────────────────────────────────────────
        self.buffer = RolloutBuffer(
            rollout_length=self.rollout_length,
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=device,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(self, state, evaluate=False):
        """
        Training mode  → stochastic sample; returns (action, log_prob, value).
        Evaluation mode → deterministic mean; returns action only.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                dist   = self.actor(state_t)
                action = dist.mean
                return action.cpu().numpy().flatten()
            else:
                action, log_prob = self.actor.get_action_and_log_prob(state_t)
                value            = self.critic(state_t)
                return (
                    action.cpu().numpy().flatten(),
                    log_prob.cpu().item(),
                    value.cpu().item(),
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Rollout collection
    # ──────────────────────────────────────────────────────────────────────────

    def store_transition(self, state, action, reward, done, log_prob, value):
        """Add one step to the rollout buffer."""
        self.buffer.add(state, action, reward, done, log_prob, value)

    def rollout_complete(self):
        """True once rollout_length steps have been stored."""
        return self.buffer.full

    def finish_rollout(self, last_state, last_done):
        """
        Bootstrap V(s_{T+1}) and run the GAE backwards pass.

        last_state : observation after the final rollout step (may be from
                     a new episode if the last step was a truncation).
        last_done  : True only for genuine terminations; False for truncations
                     (truncated episodes should still be bootstrapped).
        """
        last_state_t = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            last_value = self.critic(last_state_t).item()
        self.buffer.compute_returns_and_advantages(last_value, last_done)

    # ──────────────────────────────────────────────────────────────────────────
    # Policy update  (the PPO core)
    # ──────────────────────────────────────────────────────────────────────────

    def update(self):
        """
        Run n_epochs of PPO updates on the collected rollout.

        Mirrors CleanRL's update loop exactly:
          • Per-mini-batch advantage normalisation
          • Clipped surrogate policy loss
          • Clipped value loss  (prevents critic from moving too far per step)
          • Entropy bonus
          • Single backward pass, gradient clipping
          • Optional KL early-stopping

        Returns (policy_loss, value_loss, entropy, approx_kl) — averaged
        across all mini-batches / epochs for logging.
        """
        assert self.buffer.full, "Call finish_rollout() before update()"

        # ── LR annealing (linear decay to 0) ─────────────────────────────────
        self._global_steps += self.rollout_length
        if self.anneal_lr and self.total_timesteps:
            frac = max(0.0, 1.0 - self._global_steps / self.total_timesteps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = frac * self._base_lr

        # ── Accumulate metrics for logging ────────────────────────────────────
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_approx_kl   = 0.0
        n_batches         = 0

        for _epoch in range(self.n_epochs):
            for batch in self.buffer.get_mini_batches(self.batch_size):
                states     = batch["states"]
                actions    = batch["actions"]
                old_lp     = batch["log_probs"]
                returns    = batch["returns"]
                advantages = batch["advantages"]
                old_values = batch["old_values"]

                # ── Advantage normalisation (per mini-batch, CleanRL style) ──
                if self.norm_adv:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ── Policy loss ───────────────────────────────────────────────
                new_lp, entropy = self.actor.evaluate_actions(states, actions)

                log_ratio = new_lp - old_lp
                ratio     = log_ratio.exp()

                # Approximate KL for monitoring (Joschu's unbiased estimator)
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

                # Clipped surrogate: pessimistic min of unclipped and clipped terms
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # ── Value loss ────────────────────────────────────────────────
                new_values = self.critic(states).view(-1)

                if self.clip_vloss:
                    # Clip the critic update — prevent the value function from
                    # changing too much in one step (mirrors policy clipping).
                    v_unclipped = (new_values - returns) ** 2
                    v_clipped   = old_values + (new_values - old_values).clamp(
                        -self.clip_range, self.clip_range
                    )
                    v_loss      = 0.5 * torch.max(v_unclipped, (v_clipped - returns) ** 2).mean()
                else:
                    v_loss = 0.5 * F.mse_loss(new_values, returns)

                # ── Combined loss + backward ──────────────────────────────────
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += pg_loss.item()
                total_value_loss  += v_loss.item()
                total_entropy     += entropy_loss.item()
                total_approx_kl   += approx_kl
                n_batches         += 1

            # ── Optional KL early stopping ────────────────────────────────────
            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        # Clear buffer — on-policy data is now stale
        self.buffer.reset()

        return (
            total_policy_loss / n_batches,
            total_value_loss  / n_batches,
            total_entropy     / n_batches,
            total_approx_kl   / n_batches,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
