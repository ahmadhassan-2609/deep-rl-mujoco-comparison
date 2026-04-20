"""
Plot learning curves (mean ± std across seeds) for each environment.

Produces one figure per environment showing how each algorithm's evaluation
reward evolves over training timesteps. The shaded region is ±1 std across seeds.

Usage:
    python plotting/plot_learning_curves.py
    python plotting/plot_learning_curves.py --results-dir results/main --output results/figures
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (for running on servers)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ALGOS = ["td3", "sac", "ppo", "random"]
ENVS  = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]

# Colors and labels for consistent style across all figures
STYLE = {
    "td3":    {"color": "#e41a1c", "label": "TD3",    "linestyle": "-"},
    "sac":    {"color": "#377eb8", "label": "SAC",    "linestyle": "-"},
    "ppo":    {"color": "#4daf4a", "label": "PPO",    "linestyle": "-"},
    "random": {"color": "#999999", "label": "Random", "linestyle": "--"},
}


def load_eval_logs(results_dir: str, algo: str, env: str, seeds: list) -> pd.DataFrame:
    """
    Load eval_log.csv files for all seeds and return a combined DataFrame
    with columns [timestep, mean_reward] indexed by seed.

    We interpolate to a common timestep grid since seeds might have
    slightly different evaluation points.
    """
    dfs = []
    for seed in seeds:
        path = os.path.join(results_dir, algo, env, f"seed_{seed}", "eval_log.csv")
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def compute_mean_std(df: pd.DataFrame, timestep_col="timestep", reward_col="mean_reward"):
    """
    Given data from multiple seeds, compute mean ± std at each timestep.

    We use the union of all timestep values and interpolate missing ones.
    """
    seeds = df["seed"].unique()
    all_timesteps = sorted(df[timestep_col].unique())

    # Create a grid where each row is a seed's rewards at each timestep
    grid = []
    for seed in seeds:
        seed_df = df[df["seed"] == seed].sort_values(timestep_col)
        # Interpolate onto the common timestep grid
        interp_rewards = np.interp(all_timesteps, seed_df[timestep_col], seed_df[reward_col])
        grid.append(interp_rewards)

    grid = np.array(grid)  # shape: (n_seeds, n_timesteps)
    mean = grid.mean(axis=0)
    std  = grid.std(axis=0)

    return np.array(all_timesteps), mean, std


def plot_env(env: str, results_dir: str, output_dir: str, seeds: list, smooth_window: int = 5):
    """Create a learning curve figure for one environment."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo in ALGOS:
        style = STYLE[algo]
        df = load_eval_logs(results_dir, algo, env, seeds)

        if df is None:
            print(f"  No data for {algo} on {env}")
            continue

        if algo == "random":
            # Random baseline is a constant — show as horizontal dashed line
            random_mean = df["mean_reward"].mean()
            ax.axhline(random_mean, color=style["color"], linestyle=style["linestyle"],
                       linewidth=1.5, label=f"{style['label']} ({random_mean:.0f})", alpha=0.8)
            continue

        timesteps, mean, std = compute_mean_std(df)

        # Optional smoothing to reduce noise in the curves
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            mean = np.convolve(mean, kernel, mode="same")
            # Don't smooth std — the variability is the point

        ax.plot(timesteps, mean, color=style["color"], linestyle=style["linestyle"],
                linewidth=2, label=style["label"])
        ax.fill_between(timesteps, mean - std, mean + std,
                         color=style["color"], alpha=0.15)

    # Format axes
    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Evaluation Return", fontsize=12)
    ax.set_title(f"Learning Curves — {env}", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis with K/M suffixes
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    )

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    env_clean = env.replace("-", "_").replace(".", "").lower()
    save_path = os.path.join(output_dir, f"learning_curves_{env_clean}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves")
    parser.add_argument("--results-dir", type=str, default="results/main")
    parser.add_argument("--output",      type=str, default="results/figures")
    parser.add_argument("--seeds",  nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--envs",   nargs="+", default=ENVS)
    parser.add_argument("--smooth", type=int,  default=5,
                        help="Smoothing window size (1=no smoothing)")
    args = parser.parse_args()

    print(f"Plotting learning curves from: {args.results_dir}")
    for env in args.envs:
        print(f"\n{env}:")
        plot_env(env, args.results_dir, args.output, args.seeds, args.smooth)

    print("\nDone.")


if __name__ == "__main__":
    main()
