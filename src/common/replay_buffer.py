"""
Replay buffer for off-policy algorithms (TD3 and SAC).

The idea is simple: store transitions (s, a, r, s', done) in a circular
buffer and sample random mini-batches for training. The buffer has a fixed
capacity — once full, oldest transitions get overwritten.

Using a numpy array under the hood (not a Python list) for speed.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Simple uniform experience replay buffer.

    Parameters
    ----------
    state_dim   : dimensionality of observation space
    action_dim  : dimensionality of action space
    max_size    : maximum number of transitions to store (default 1M)
    device      : torch device to move samples to
    """

    def __init__(self, state_dim, action_dim, max_size=1_000_000, device="cpu"):
        self.max_size = max_size
        self.ptr = 0          # points to where we'll write next
        self.size = 0         # current number of stored transitions
        self.device = device

        # Pre-allocate all arrays upfront — much faster than appending
        self.states      = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((max_size, 1),          dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((max_size, 1),          dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Add a single transition to the buffer."""
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)

        # Circular: wrap around when we hit the end
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample a random mini-batch of transitions.

        Returns a dict of tensors, all on self.device.
        """
        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            "states":      torch.FloatTensor(self.states[idx]).to(self.device),
            "actions":     torch.FloatTensor(self.actions[idx]).to(self.device),
            "rewards":     torch.FloatTensor(self.rewards[idx]).to(self.device),
            "next_states": torch.FloatTensor(self.next_states[idx]).to(self.device),
            "dones":       torch.FloatTensor(self.dones[idx]).to(self.device),
        }

    def __len__(self):
        return self.size
