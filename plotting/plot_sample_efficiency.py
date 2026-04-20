"""
Sample efficiency analysis: steps-to-threshold plots.

For each environment, computes how many timesteps each algorithm needs to
reach X% of the best final performance (where best = max across all algorithms).

Thresholds: 25%, 50%, 75% of the best final reward.

Usage:
    python plotting/plot_sample_efficiency.py
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ALGOS  = ["td3", "sac", "ppo"]
ENVS   = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
THRESHOLDS = [0.25, 0.50, 0.75]

ALGO_COLORS = {
    "td3": "#e41a1c",
    "sac": "#377eb8",
    "ppo": "#4daf4a",
}


def load_mean_curve(results_dir, algo, env, seeds):
    """Load and average eval curves across seeds. Returns (timesteps, mean_rewards)."""
    dfs = []
    for seed in seeds:
        path = os.path.join(results_dir, algo, env, f"seed_{seed}", "eval_log.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)

    if not dfs:
        return None, None

    # Get common timesteps (use first seed's as reference)
    ref_timesteps = dfs[0]["timestep"].values
    rewards_grid  = []

    for df in dfs:
        r = np.interp(ref_timesteps, df["timestep"].values, df["mean_reward"].values)
        rewards_grid.append(r)

    mean_rewards = np.mean(rewards_grid, axis=0)
    return ref_timesteps, mean_rewards


def steps_to_threshold(timesteps, rewards, threshold_value):
    """Find the first timestep where reward exceeds threshold_value."""
    for t, r in zip(timesteps, rewards):
        if r >= threshold_value:
            return t
    return None  # Never reached


def plot_sample_efficiency(results_dir, output_dir, seeds):
    """Create one figure with subplots for each environment."""
    n_envs = len(ENVS)
    n_thresholds = len(THRESHOLDS)

    fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 5), sharey=False)
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, ENVS):
        # First pass: get best final performance across all algorithms
        best_final = -np.inf
        all_curves = {}

        for algo in ALGOS:
            timesteps, mean_rewards = load_mean_curve(results_dir, algo, env, seeds)
            if timesteps is None:
                continue
            all_curves[algo] = (timesteps, mean_rewards)
            best_final = max(best_final, mean_rewards[-1])

        if best_final <= 0:
            print(f"Warning: best final reward is non-positive for {env}, skipping")
            continue

        # Second pass: compute steps to each threshold
        x_positions = np.arange(n_thresholds)
        bar_width   = 0.25

        for i, algo in enumerate(ALGOS):
            if algo not in all_curves:
                continue

            timesteps, mean_rewards = all_curves[algo]
            steps = []

            for thresh_frac in THRESHOLDS:
                threshold_value = thresh_frac * best_final
                s = steps_to_threshold(timesteps, mean_rewards, threshold_value)
                if s is None:
                    s = max(timesteps) * 1.1  # Show as "beyond" the x-axis
                steps.append(s / 1e6)  # Convert to millions

            offset = (i - 1) * bar_width
            bars = ax.bar(x_positions + offset, steps, bar_width,
                          label=algo.upper(), color=ALGO_COLORS[algo], alpha=0.8,
                          edgecolor="white", linewidth=0.5)

            # Add value labels on top of bars
            for bar, s in zip(bars, steps):
                if s < max(timesteps) / 1e6 * 1.05:  # only label if reached
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{s:.2f}M", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{int(t*100)}% of best" for t in THRESHOLDS])
        ax.set_ylabel("Timesteps to Threshold (millions)")
        ax.set_title(env.replace("-v5", ""))
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Sample Efficiency: Steps to Performance Threshold", fontsize=13, y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "sample_efficiency.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot sample efficiency analysis")
    parser.add_argument("--results-dir", type=str, default="results/main")
    parser.add_argument("--output",      type=str, default="results/figures")
    parser.add_argument("--seeds",  nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    print("Computing sample efficiency...")
    plot_sample_efficiency(args.results_dir, args.output, args.seeds)
    print("Done.")


if __name__ == "__main__":
    main()
