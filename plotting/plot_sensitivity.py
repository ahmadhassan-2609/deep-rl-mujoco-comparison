"""
Plot hyperparameter sensitivity results.

For each (algorithm, hyperparameter) sweep, shows how final performance
varies with the hyperparameter value across environments.

Usage:
    python plotting/plot_sensitivity.py
    python plotting/plot_sensitivity.py --sweep td3_lr td3_noise
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


ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]

# Same info as run_sensitivity.py — what values were swept
SWEEPS = {
    "td3_lr": {
        "algo":    "td3",
        "param":   "learning_rate",
        "values":  [1e-4, 3e-4, 1e-3],
        "default": 3e-4,
        "x_label": "Learning Rate",
        "x_format": "sci",
    },
    "td3_noise": {
        "algo":    "td3",
        "param":   "exploration_noise",
        "values":  [0.05, 0.1, 0.2, 0.3],
        "default": 0.1,
        "x_label": "Exploration Noise σ",
        "x_format": "float",
    },
    "sac_lr": {
        "algo":    "sac",
        "param":   "learning_rate",
        "values":  [1e-4, 3e-4, 1e-3],
        "default": 3e-4,
        "x_label": "Learning Rate",
        "x_format": "sci",
    },
    "sac_entropy": {
        "algo":    "sac",
        "param":   "alpha_mode",
        "values":  ["auto", 0.1, 0.5, 1.0],
        "default": "auto",
        "x_label": "Alpha (entropy coeff)",
        "x_format": "str",
    },
    "ppo_clip": {
        "algo":    "ppo",
        "param":   "clip_range",
        "values":  [0.1, 0.2, 0.3, 0.4],
        "default": 0.2,
        "x_label": "Clip Range ε",
        "x_format": "float",
    },
}

ENV_COLORS = {
    "HalfCheetah-v5": "#e41a1c",
    "Hopper-v5":      "#377eb8",
    "Walker2d-v5":    "#4daf4a",
}

ENV_MARKERS = {
    "HalfCheetah-v5": "o",
    "Hopper-v5":      "s",
    "Walker2d-v5":    "^",
}


def format_value(v) -> str:
    """Format a value the same way run_sensitivity.py does for directory names."""
    if isinstance(v, float):
        if v < 0.01 or v >= 100:
            return f"{v:.0e}"
        return str(v)
    return str(v)


def load_final_performance(results_base: str, sweep_name: str, algo: str, env: str,
                            value, seeds: list):
    """
    Load the last eval reward (at 1M steps) for a given (sweep_name, value, env).
    Also handles the "default" values which are stored under results/main/.
    """
    value_str = format_value(value)
    result_dir = os.path.join(results_base, f"sensitivity/{sweep_name}/{value_str}", algo, env)

    rewards = []
    for seed in seeds:
        path = os.path.join(result_dir, f"seed_{seed}", "eval_log.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        # Take the last evaluation point as final performance
        final_reward = df["mean_reward"].iloc[-1]
        rewards.append(final_reward)

    if not rewards:
        return None, None
    return np.mean(rewards), np.std(rewards)


def load_final_from_main(results_base: str, algo: str, env: str, seeds: list):
    """Load final performance from the main experiments (for default hyperparameter values)."""
    rewards = []
    for seed in seeds:
        path = os.path.join(results_base, "main", algo, env, f"seed_{seed}", "eval_log.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        rewards.append(df["mean_reward"].iloc[-1])

    if not rewards:
        return None, None
    return np.mean(rewards), np.std(rewards)


def plot_sweep(sweep_name: str, results_base: str, output_dir: str, seeds: list):
    """Create a sensitivity plot for one sweep."""
    sweep = SWEEPS[sweep_name]
    algo  = sweep["algo"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for env in ENVS:
        means = []
        stds  = []
        x_vals = []
        has_non_default = False

        for value in sweep["values"]:
            # Default values are in main experiments directory
            if value == sweep["default"]:
                mean, std = load_final_from_main(results_base, algo, env, seeds)
            else:
                mean, std = load_final_performance(results_base, sweep_name, algo, env, value, seeds)
                if mean is not None:
                    has_non_default = True

            if mean is None:
                continue

            means.append(mean)
            stds.append(std)
            x_vals.append(value)

        # Only plot environments that have actual sweep data (not just the default point)
        if not has_non_default:
            continue

        # For string x-values (like "auto"), use integer positions
        x_positions = list(range(len(x_vals)))

        ax.plot(x_positions, means,
                color=ENV_COLORS[env], marker=ENV_MARKERS[env],
                linewidth=2, markersize=7, label=env.replace("-v5", ""))
        ax.errorbar(x_positions, means, yerr=stds,
                    color=ENV_COLORS[env], fmt="none", capsize=4, linewidth=1.5)

        # Mark default value
        if sweep["default"] in x_vals:
            default_idx = x_vals.index(sweep["default"])
            ax.axvline(default_idx, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    # Set x-axis tick labels
    all_x = sweep["values"]
    x_positions = list(range(len(all_x)))

    if sweep["x_format"] == "sci":
        tick_labels = [f"{v:.0e}" for v in all_x]
    elif sweep["x_format"] == "float":
        tick_labels = [str(v) for v in all_x]
    else:
        tick_labels = [str(v) for v in all_x]

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel(sweep["x_label"], fontsize=11)
    ax.set_ylabel("Final Evaluation Return (mean ± std)", fontsize=11)
    ax.set_title(f"{algo.upper()} Sensitivity — {sweep['param']}", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"sensitivity_{sweep_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot sensitivity experiment results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output",      type=str, default="results/figures")
    parser.add_argument("--seeds",  nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sweep",  nargs="+", default=list(SWEEPS.keys()),
                        choices=list(SWEEPS.keys()))
    args = parser.parse_args()

    for sweep_name in args.sweep:
        print(f"Plotting sensitivity: {sweep_name}")
        plot_sweep(sweep_name, args.results_dir, args.output, args.seeds)

    print("\nDone.")


if __name__ == "__main__":
    main()
