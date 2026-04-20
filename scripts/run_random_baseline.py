"""
Evaluate a random policy on all three environments.

The random policy is our lower-bound baseline — it takes uniformly random actions
from the action space on every step. We run it for several episodes and record the
average return, which serves as the zero-performance reference on learning curves.

Usage:
    python scripts/run_random_baseline.py
    python scripts/run_random_baseline.py --episodes 20
"""

import argparse
import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from src.common.utils import set_seed


ENVS  = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
SEEDS = [0, 1, 2, 3, 4]


def evaluate_random_policy(env_name: str, n_episodes: int, seed: int):
    """Run a random policy and return per-episode rewards."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep + 1000)
        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            action = env.action_space.sample()
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)

    env.close()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate random policy baseline")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--envs",  nargs="+", default=ENVS)
    args = parser.parse_args()

    print("Evaluating random policy baseline...\n")

    for env_name in args.envs:
        all_rewards = []

        for seed in args.seeds:
            set_seed(seed)
            rewards = evaluate_random_policy(env_name, args.episodes, seed)
            all_rewards.extend(rewards)

            mean_r = np.mean(rewards)
            std_r  = np.std(rewards)
            print(f"  {env_name} | seed={seed} | mean={mean_r:.1f} ± {std_r:.1f}")

            # Save per-seed results
            result_dir = os.path.join("results", "main", "random", env_name, f"seed_{seed}")
            os.makedirs(result_dir, exist_ok=True)

            csv_path = os.path.join(result_dir, "eval_log.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestep", "mean_reward", "std_reward", "min_reward", "max_reward"])
                # Random baseline is constant — report at t=0
                writer.writerow([0, round(mean_r, 4), round(std_r, 4),
                                  round(min(rewards), 4), round(max(rewards), 4)])

        overall_mean = np.mean(all_rewards)
        overall_std  = np.std(all_rewards)
        print(f"  {env_name} | OVERALL | mean={overall_mean:.1f} ± {overall_std:.1f}\n")

    print("Random baseline evaluation complete.")
    print("Results saved to: results/main/random/")


if __name__ == "__main__":
    main()
