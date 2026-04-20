"""
Rollout buffer for on-policy PPO.

Stores a fixed-length rollout of experience, computes GAE advantages and
λ-returns, then yields shuffled mini-batches for PPO's update epochs.
After the update the buffer is reset — on-policy means old data is stale.

Storage convention (differs from CleanRL by one index):
    CleanRL stores dones[t] = next_done BEFORE the action at step t, so
    their "nextnonterminal" uses dones[t+1].
    We store dones[t] = done flag AFTER the action at step t, so our
    "nextnonterminal" uses dones[t] directly.
    Both represent the same physical quantity: "did the transition at step t
    result in an episode boundary?"
"""

import numpy as np
import torch


class RolloutBuffer:
    """
    Parameters
    ----------
    rollout_length : steps to collect before each PPO update
    state_dim      : observation dimension
    action_dim     : action dimension
    gamma          : discount factor γ
    gae_lambda     : GAE λ  (0 = pure TD, 1 = pure MC)
    device         : torch device for mini-batch tensors
    """

    def __init__(self, rollout_length, state_dim, action_dim, gamma, gae_lambda, device="cpu"):
        self.rollout_length = rollout_length
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.device         = device

        # Pre-allocated storage (numpy for fast sequential writes during rollout)
        self.states    = np.zeros((rollout_length, state_dim),  dtype=np.float32)
        self.actions   = np.zeros((rollout_length, action_dim), dtype=np.float32)
        self.rewards   = np.zeros((rollout_length,),            dtype=np.float32)
        self.dones     = np.zeros((rollout_length,),            dtype=np.float32)
        self.log_probs = np.zeros((rollout_length,),            dtype=np.float32)
        self.values    = np.zeros((rollout_length,),            dtype=np.float32)

        # Filled by compute_returns_and_advantages()
        self.advantages = np.zeros((rollout_length,), dtype=np.float32)
        self.returns    = np.zeros((rollout_length,), dtype=np.float32)

        self.ptr  = 0
        self.full = False

    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Clear the buffer at the start of each rollout."""
        self.ptr  = 0
        self.full = False

    def add(self, state, action, reward, done, log_prob, value):
        """Store a single transition."""
        assert self.ptr < self.rollout_length, "Buffer full — call reset() first"
        self.states[self.ptr]    = state
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = float(done)   # 1.0 if episode ended after this step
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.ptr += 1
        if self.ptr == self.rollout_length:
            self.full = True

    # ──────────────────────────────────────────────────────────────────────────

    def compute_returns_and_advantages(self, last_value: float, last_done: bool):
        """
        GAE backwards pass (Schulman et al. 2016).

        δ_t  = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
        Â_t  = δ_t + γλ · (1 − done_t) · Â_{t+1}

        done_t = 1 means the transition at t ended the episode, so V(s_{t+1})
        must not be bootstrapped (it belongs to a new episode).

        Parameters
        ----------
        last_value : V(s_{T+1}), used to bootstrap a non-terminal rollout end.
        last_done  : True only for actual terminations (not time-limit truncations).
                     Truncated episodes are bootstrapped (last_done=False) because
                     the agent hasn't truly failed — it just hit the step budget.
        """
        last_gae = 0.0
        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                # Final step: bootstrap with the externally provided last_value/last_done
                non_terminal = 1.0 - float(last_done)
                next_val     = last_value
            else:
                # General case: use dones[t] — "did transition t end the episode?"
                # Equivalent to CleanRL's 1 - dones[t+1] (their convention stores
                # the done one step later).
                non_terminal = 1.0 - self.dones[t]
                next_val     = self.values[t + 1]

            delta    = self.rewards[t] + self.gamma * next_val * non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae

        # λ-returns = advantages + baseline values (target for the value network)
        self.returns = self.advantages + self.values

    # ──────────────────────────────────────────────────────────────────────────

    def get_mini_batches(self, batch_size: int):
        """
        Yield shuffled mini-batches as torch tensors.

        Advantages are returned *unnormalized*. PPO's update() normalizes them
        per mini-batch, which keeps gradient magnitudes stable across epochs.

        Each dict contains:
            states, actions, log_probs, returns, advantages, old_values
        """
        assert self.full, "Rollout incomplete — collect rollout_length steps first"
        indices = np.random.permutation(self.rollout_length)
        for start in range(0, self.rollout_length, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "states":     torch.FloatTensor(self.states[idx]).to(self.device),
                "actions":    torch.FloatTensor(self.actions[idx]).to(self.device),
                "log_probs":  torch.FloatTensor(self.log_probs[idx]).to(self.device),
                "returns":    torch.FloatTensor(self.returns[idx]).to(self.device),
                "advantages": torch.FloatTensor(self.advantages[idx]).to(self.device),
                # old_values: V(s) from collection time, used for value clipping
                "old_values": torch.FloatTensor(self.values[idx]).to(self.device),
            }
