"""
Record 10-episode videos for the best seed of each (algo, env) combination.

Reads eval_log.csv for every seed, picks the one with the highest mean_reward
at the final logged timestep (~1 M steps), loads final_model.pt, and records
10 deterministic episodes with gymnasium RecordVideo.

Output: results/videos/{algo}_{env_short}_best_seed{N}.mp4
"""

import os
import sys
import glob
import shutil
import csv
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from src.common.utils import get_device, set_seed


# ── configuration ────────────────────────────────────────────────────────────

ALGOS = ["td3", "sac", "ppo"]
ENVS  = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]

ENV_SHORT = {
    "HalfCheetah-v5": "halfcheetah",
    "Hopper-v5":      "hopper",
    "Walker2d-v5":    "walker2d",
}

RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "main",
)
VIDEO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "videos",
)

N_EPISODES = 10


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(model_path: str) -> dict:
    config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {
        "hidden_dims": [256, 256],
        "gamma": 0.99, "tau": 0.005, "batch_size": 256,
        "buffer_size": 1_000_000, "learning_starts": 25_000,
        "exploration_noise": 0.1, "target_noise": 0.2,
        "target_noise_clip": 0.5, "policy_delay": 2,
        "auto_tune_alpha": True, "initial_alpha": 1.0,
        "gae_lambda": 0.95, "clip_range": 0.2, "n_epochs": 10,
        "rollout_length": 2048, "vf_coef": 0.5, "ent_coef": 0.0,
        "max_grad_norm": 0.5, "learning_rate": 3e-4,
    }


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


def find_best_seed(algo: str, env: str) -> tuple[int, float]:
    """Return (best_seed, final_mean_reward) for the given (algo, env) pair."""
    base = os.path.join(RESULTS_ROOT, algo, env)
    seed_dirs = sorted(glob.glob(os.path.join(base, "seed_*")))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found under {base}")

    best_seed   = None
    best_reward = -np.inf

    for seed_dir in seed_dirs:
        csv_path = os.path.join(seed_dir, "eval_log.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        # Use the last logged row (closest to 1 M steps)
        final_reward = float(rows[-1]["mean_reward"])
        seed_n = int(os.path.basename(seed_dir).split("_")[1])
        if final_reward > best_reward:
            best_reward = final_reward
            best_seed   = seed_n

    if best_seed is None:
        raise RuntimeError(f"Could not determine best seed for {algo}/{env}")
    return best_seed, best_reward


def record(algo: str, env: str, seed: int, out_path: str) -> None:
    """Load final_model.pt for (algo, env, seed) and record N_EPISODES."""
    model_path = os.path.join(RESULTS_ROOT, algo, env, f"seed_{seed}", "final_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    config = load_config(model_path)
    device = get_device()
    set_seed(42)

    # Temporary subfolder so RecordVideo names don't clash across runs
    tmp_dir = os.path.join(VIDEO_DIR, f"_tmp_{algo}_{ENV_SHORT[env]}")
    os.makedirs(tmp_dir, exist_ok=True)

    env_obj = gym.make(env, render_mode="rgb_array")
    env_obj = RecordVideo(
        env_obj,
        video_folder=tmp_dir,
        name_prefix=f"{algo}_{ENV_SHORT[env]}_seed{seed}",
        episode_trigger=lambda ep: True,   # record every episode
    )

    state_dim  = env_obj.observation_space.shape[0]
    action_dim = env_obj.action_space.shape[0]
    max_action = float(env_obj.action_space.high[0])

    agent = build_agent(algo, state_dim, action_dim, max_action, config, device)
    agent.load(model_path)

    for ep in range(N_EPISODES):
        state, _ = env_obj.reset(seed=42 + ep)
        done = truncated = False
        total_reward = 0.0
        steps = 0
        while not done and not truncated:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env_obj.step(action)
            total_reward += reward
            steps += 1
        print(f"    Episode {ep+1:>2d}: reward = {total_reward:>8.1f}  (steps: {steps})")

    env_obj.close()

    # RecordVideo produces one mp4 per episode; concatenate with moviepy.
    mp4_files = sorted(glob.glob(os.path.join(tmp_dir, "*.mp4")))
    if not mp4_files:
        raise RuntimeError(f"No mp4 files produced in {tmp_dir}")

    from moviepy import VideoFileClip, concatenate_videoclips
    clips = [VideoFileClip(p) for p in mp4_files]
    final = concatenate_videoclips(clips)
    final.write_videofile(out_path, logger=None)
    for c in clips:
        c.close()
    final.close()
    print(f"    All {len(mp4_files)} episodes concatenated -> {out_path}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    tasks = [(algo, env) for algo in ALGOS for env in ENVS]

    for algo, env in tasks:
        env_short = ENV_SHORT[env]
        print(f"\n{'='*60}")
        print(f"  {algo.upper()} — {env}")
        print(f"{'='*60}")

        try:
            best_seed, best_reward = find_best_seed(algo, env)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  ERROR finding best seed: {e}")
            print("  Stopping.")
            sys.exit(1)

        print(f"  Best seed: {best_seed}  (final mean reward: {best_reward:.1f})")

        out_path = os.path.join(VIDEO_DIR, f"{algo}_{env_short}_best_seed{best_seed}.mp4")
        print(f"  Recording {N_EPISODES} episodes -> {out_path}")

        try:
            record(algo, env, best_seed, out_path)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  ERROR during recording: {e}")
            print("  Stopping.")
            sys.exit(1)

        print(f"  Done: {out_path}")

    print(f"\n{'='*60}")
    print("All 9 videos recorded successfully.")
    print(f"Output directory: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
