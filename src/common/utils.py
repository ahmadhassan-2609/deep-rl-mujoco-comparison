"""
Shared utility functions used across all algorithms and scripts.
"""

import random
import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int):
    """
    Set random seeds everywhere for reproducibility.
    Gymnasium environments get seeded separately during reset().
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # Deterministic ops can be slower but are needed for exact reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_env(env_name: str, seed: int):
    """
    Create and seed a Gymnasium environment.

    Returns the environment. Seeding happens via reset(seed=...) which is
    the Gymnasium v26+ API (not env.seed() which is deprecated).
    """
    env = gym.make(env_name)
    # Do an initial reset to apply the seed
    env.reset(seed=seed)
    return env


def make_ppo_env(env_name: str, seed: int, gamma: float = 0.99):
    """
    Create a PPO training environment with reward normalisation.

    PPO is acutely sensitive to reward scale.  Without normalisation,
    HalfCheetah returns span roughly [-700, +10000], which makes the
    value-function loss enormous early in training and the GAE advantage
    estimates very noisy.  NormalizeReward tracks a running discounted
    variance of rewards and rescales them to ~unit variance, matching
    what CleanRL does for all MuJoCo benchmarks.

    Observation normalisation is deliberately omitted so the policy can
    be evaluated on the raw, unmodified environment without any
    train/eval distribution mismatch.

    ClipAction ensures sampled actions never exceed the environment's
    declared bounds (safe for all MuJoCo tasks).
    """
    env = gym.make(env_name)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10.0, 10.0))
    env.reset(seed=seed)
    return env


def evaluate_policy(env_name: str, agent, n_episodes: int = 10, seed_offset: int = 1000):
    """
    Run the agent deterministically for n_episodes and return mean/std reward.

    We use a separate evaluation environment (not the training one) and
    deterministic actions (no exploration noise). The seed_offset ensures
    eval episodes don't overlap with training seeds.

    Parameters
    ----------
    env_name   : Gymnasium environment id
    agent      : agent object with a select_action(state, evaluate=True) method
    n_episodes : number of evaluation episodes
    seed_offset: added to episode index for eval seeds (separate from training)

    Returns
    -------
    mean_reward, std_reward : floats
    """
    eval_env = gym.make(env_name)
    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = eval_env.reset(seed=seed_offset + ep)
        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)

    eval_env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward


def get_device():
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
