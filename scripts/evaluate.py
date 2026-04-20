"""
Load a saved model and run evaluation episodes.

Useful for:
  - Sanity-checking a trained model
  - Recording videos of trained agents
  - Comparing specific models after training is done

Usage:
    python scripts/evaluate.py --model results/main/td3/HalfCheetah-v5/seed_0/final_model.pt \
                               --algo td3 --env HalfCheetah-v5

    python scripts/evaluate.py --model results/main/sac/Hopper-v5/seed_2/final_model.pt \
                               --algo sac --env Hopper-v5 --render --episodes 5

    # Record video:
    python scripts/evaluate.py --model results/main/ppo/Walker2d-v5/seed_1/final_model.pt \
                               --algo ppo --env Walker2d-v5 --record-video results/videos/
"""

import argparse
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from src.common.utils import get_device, set_seed


def load_config_from_results(model_path: str) -> dict:
    """Try to load the config.yaml saved alongside the model."""
    config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def build_agent(algo, state_dim, action_dim, max_action, config, device):
    if algo == "td3":
        from src.td3.td3 import TD3
        return TD3(state_dim, action_dim, max_action, config, device)
    elif algo == "sac":
        from src.sac.sac import SAC
        return SAC(state_dim, action_dim, max_action, config, device)
    elif algo == "ppo":
        from src.ppo.ppo import PPO
        return PPO(state_dim, action_dim, config, device)
    else:
        raise ValueError(f"Unknown algo: {algo}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model")
    parser.add_argument("--model",       type=str, required=True,  help="Path to .pt model file")
    parser.add_argument("--algo",        type=str, required=True,  choices=["td3", "sac", "ppo"])
    parser.add_argument("--env",         type=str, required=True,  help="Gymnasium environment id")
    parser.add_argument("--episodes",    type=int, default=10,     help="Number of eval episodes")
    parser.add_argument("--seed",        type=int, default=42,     help="Random seed")
    parser.add_argument("--render",      action="store_true",      help="Render environment (human mode)")
    parser.add_argument("--record-video",type=str, default=None,   help="Directory to save video to")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Load config (to know hidden dims etc.)
    config = load_config_from_results(args.model)
    if not config:
        # Use defaults if no config found
        print("Warning: no config.yaml found alongside model, using default hyperparams")
        config = {
            "hidden_dims": [256, 256],
            "gamma": 0.99, "tau": 0.005, "batch_size": 256,
            "buffer_size": 1000000, "learning_starts": 25000,
            "exploration_noise": 0.1, "target_noise": 0.2,
            "target_noise_clip": 0.5, "policy_delay": 2,
            "auto_tune_alpha": True, "initial_alpha": 1.0,
            "gae_lambda": 0.95, "clip_range": 0.2, "n_epochs": 10,
            "rollout_length": 2048, "vf_coef": 0.5, "ent_coef": 0.0,
            "max_grad_norm": 0.5, "learning_rate": 3e-4,
        }

    # Setup environment
    render_mode = "human" if args.render else None
    if args.record_video:
        env = gym.make(args.env, render_mode="rgb_array")
        os.makedirs(args.record_video, exist_ok=True)
        env = RecordVideo(env, video_folder=args.record_video,
                          name_prefix=f"{args.algo}_{args.env.replace('-', '_')}")
    else:
        env = gym.make(args.env, render_mode=render_mode)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Build and load agent
    agent = build_agent(args.algo, state_dim, action_dim, max_action, config, device)
    agent.load(args.model)
    print(f"Loaded model from: {args.model}")
    print(f"Running {args.episodes} evaluation episodes on {args.env}...\n")

    episode_rewards = []

    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not done and not truncated:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        print(f"  Episode {ep+1:>2d}: reward = {total_reward:>8.1f}  (steps: {steps})")

    env.close()

    mean_r = np.mean(episode_rewards)
    std_r  = np.std(episode_rewards)
    min_r  = np.min(episode_rewards)
    max_r  = np.max(episode_rewards)

    print(f"\nResults over {args.episodes} episodes:")
    print(f"  Mean:  {mean_r:.1f}")
    print(f"  Std:   {std_r:.1f}")
    print(f"  Min:   {min_r:.1f}")
    print(f"  Max:   {max_r:.1f}")

    if args.record_video:
        print(f"\nVideo saved to: {args.record_video}")


if __name__ == "__main__":
    main()
