"""
Neural network architectures used across all three algorithms.

I'm keeping things simple here — all actors and critics use the same
2-hidden-layer MLP backbone. The differences come in the output heads
and how actions are sampled (deterministic vs stochastic).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# Numerical stability constant — used in log-prob calculations
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def build_mlp(input_dim, output_dim, hidden_dims, activation=nn.ReLU):
    """
    Helper to build a simple feedforward network.

    input_dim  -> hidden_dims[0] -> hidden_dims[1] -> ... -> output_dim
    with `activation` between each layer (but not after the last one).
    """
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# TD3 Networks
# ─────────────────────────────────────────────────────────────────────────────

class DeterministicActor(nn.Module):
    """
    Actor for TD3. Outputs a single deterministic action.

    The output is passed through tanh so it's in (-1, 1), then scaled
    by max_action so it fits the actual action space bounds.
    """

    def __init__(self, state_dim, action_dim, hidden_dims, max_action):
        super().__init__()
        self.net = build_mlp(state_dim, action_dim, hidden_dims)
        self.max_action = max_action

    def forward(self, state):
        # tanh squashes to (-1, 1), then scale to action range
        return self.max_action * torch.tanh(self.net(state))


# ─────────────────────────────────────────────────────────────────────────────
# SAC Networks
# ─────────────────────────────────────────────────────────────────────────────

class SquashedGaussianActor(nn.Module):
    """
    Actor for SAC. Outputs a stochastic action via the reparameterization trick.

    The key idea is:
      1. Network outputs mean and log_std
      2. Sample: u = mean + std * epsilon,  epsilon ~ N(0,1)
      3. Squash: a = tanh(u), so a is in (-1, 1)
      4. Correct the log-probability for the tanh transformation

    The log-prob correction (step 4) matters a lot. Without it the policy
    gradient is wrong. The correction is:
        log pi(a|s) = log N(u|mean,std) - sum(log(1 - tanh(u)^2 + eps))
    """

    def __init__(self, state_dim, action_dim, hidden_dims, max_action):
        super().__init__()
        # Single network outputs both mean and log_std
        self.net = build_mlp(state_dim, action_dim * 2, hidden_dims)
        self.max_action = max_action

    def forward(self, state):
        """Returns (action, log_prob) — used during training."""
        out = self.net(state)
        mean, log_std = out.chunk(2, dim=-1)

        # Clamp log_std to a reasonable range for numerical stability
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        # Reparameterization trick: sample from N(mean, std)
        dist = Normal(mean, std)
        u = dist.rsample()  # u = mean + std * epsilon (differentiable)

        # Squash to (-1, 1) with tanh, then scale to action bounds
        a = torch.tanh(u)
        action = self.max_action * a

        # Log-prob with tanh correction
        # log pi(a|s) = log p(u) - log |da/du|
        # da/du = diag(1 - tanh(u)^2), so log|da/du| = sum(log(1 - a^2))
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(1.0 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state):
        """Deterministic action for evaluation (use mean, no sampling)."""
        out = self.net(state)
        mean, _ = out.chunk(2, dim=-1)
        return self.max_action * torch.tanh(mean)


# ─────────────────────────────────────────────────────────────────────────────
# PPO Networks
# ─────────────────────────────────────────────────────────────────────────────

def _ortho_init(sequential, hidden_gain: float, output_gain: float):
    """
    Apply orthogonal initialisation to every Linear layer in a Sequential.

    Hidden layers get `hidden_gain` (sqrt(2) is optimal for Tanh).
    The final Linear layer gets `output_gain`:
      - 0.01 for actor output  → near-zero initial actions, low control cost
      - 1.0  for critic output → standard scale for value predictions
    Biases are zeroed throughout.
    """
    linear_layers = [m for m in sequential if isinstance(m, nn.Linear)]
    for layer in linear_layers[:-1]:
        nn.init.orthogonal_(layer.weight, gain=hidden_gain)
        nn.init.constant_(layer.bias, 0.0)
    nn.init.orthogonal_(linear_layers[-1].weight, gain=output_gain)
    nn.init.constant_(linear_layers[-1].bias, 0.0)


class GaussianActor(nn.Module):
    """
    Actor for PPO. Stochastic policy with a Gaussian distribution.

    Unlike SAC, there's no tanh squashing here — PPO environments usually
    have bounded action spaces handled by the env itself or via clipping.
    The log_std is a learnable parameter (not state-dependent), which is
    the standard approach for continuous-action PPO.

    Two design choices that match CleanRL's reference implementation:
      - Tanh activations: bounded outputs prevent exploding pre-activations
        with large observations; no dead-neuron problem unlike ReLU.
      - Orthogonal init with output gain=0.01: makes initial action means
        near zero, so the policy starts with low control cost and learns
        coordinated movements from scratch rather than fighting random noise.
    """

    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        # Tanh matches CleanRL / most reference PPO implementations for MuJoCo
        self.net = build_mlp(state_dim, action_dim, hidden_dims, activation=nn.Tanh)
        # log_std as a separate learnable parameter, initialized to 0 (std=1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Orthogonal init: hidden gain = sqrt(2) for Tanh, output gain = 0.01
        _ortho_init(self.net, hidden_gain=2 ** 0.5, output_gain=0.01)

    def forward(self, state):
        """Returns the action distribution."""
        mean = self.net(state)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
        return Normal(mean, std)

    def get_action_and_log_prob(self, state):
        """Sample an action and compute its log-probability. Used during rollout collection."""
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dims
        return action, log_prob

    def evaluate_actions(self, state, action):
        """
        Given states and actions (from an old rollout), compute log-probs and entropy
        under the CURRENT policy. Used during PPO update epochs.
        """
        dist = self.forward(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """
    Value function V(s) for PPO. Just predicts a scalar given the state.

    Uses Tanh activations and orthogonal initialisation (output gain=1.0)
    to match the CleanRL reference architecture for MuJoCo PPO.
    """

    def __init__(self, state_dim, hidden_dims):
        super().__init__()
        self.net = build_mlp(state_dim, 1, hidden_dims, activation=nn.Tanh)
        _ortho_init(self.net, hidden_gain=2 ** 0.5, output_gain=1.0)

    def forward(self, state):
        return self.net(state)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: Q-Network (TD3 and SAC both use this)
# ─────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Action-value function Q(s, a). Takes the concatenated (state, action) as input.

    Both TD3 and SAC instantiate two of these ("twin critics") and use
    the minimum of their outputs as the target to avoid overestimation.
    """

    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        self.net = build_mlp(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
