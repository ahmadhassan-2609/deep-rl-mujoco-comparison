"""
Generate a summary comparison table of final performance for all algorithms and environments.

Outputs:
  - results/figures/summary_table.png  : formatted table image (for reports)
  - results/figures/summary_table.csv  : raw numbers (for LaTeX tables)

Usage:
    python plotting/plot_comparison_table.py
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


ALGOS = ["td3", "sac", "ppo", "random"]
ENVS  = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]

ALGO_LABELS = {
    "td3":    "TD3",
    "sac":    "SAC",
    "ppo":    "PPO",
    "random": "Random",
}


def load_final_performance(results_dir, algo, env, seeds):
    """Return mean ± std of final evaluation reward across seeds."""
    rewards = []
    for seed in seeds:
        path = os.path.join(results_dir, algo, env, f"seed_{seed}", "eval_log.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        rewards.append(df["mean_reward"].iloc[-1])

    if not rewards:
        return None, None
    return np.mean(rewards), np.std(rewards)


def generate_table(results_dir, output_dir, seeds):
    """Build summary table and save as PNG and CSV."""
    data = {}  # data[env][algo] = (mean, std)

    for env in ENVS:
        data[env] = {}
        for algo in ALGOS:
            mean, std = load_final_performance(results_dir, algo, env, seeds)
            data[env][algo] = (mean, std)

    # ── Save as CSV for LaTeX ─────────────────────────────────────────────────
    rows = []
    for env in ENVS:
        row = {"Environment": env}
        for algo in ALGOS:
            mean, std = data[env][algo]
            if mean is not None:
                row[ALGO_LABELS[algo]] = f"{mean:.0f} ± {std:.0f}"
            else:
                row[ALGO_LABELS[algo]] = "N/A"
        rows.append(row)

    df_table = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "summary_table.csv")
    df_table.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    print("\nSummary Table:")
    print(df_table.to_string(index=False))

    # ── Save as formatted image ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    col_labels = ["Environment"] + [ALGO_LABELS[a] for a in ALGOS]
    cell_text  = []

    for env in ENVS:
        row = [env.replace("-v5", "")]
        best_mean = -np.inf

        # Find the best performing algorithm
        for algo in ["td3", "sac", "ppo"]:  # exclude random from best
            mean, _ = data[env][algo]
            if mean is not None and mean > best_mean:
                best_mean = mean

        for algo in ALGOS:
            mean, std = data[env][algo]
            if mean is None:
                row.append("N/A")
            else:
                row.append(f"{mean:.0f} ± {std:.0f}")
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style the header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(ENVS) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

    plt.title("Final Performance at 1M Steps (mean ± std across 5 seeds)",
              fontsize=12, pad=15)
    plt.tight_layout()

    img_path = os.path.join(output_dir, "summary_table.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved image: {img_path}")

    # Also print LaTeX table format
    print("\nLaTeX table:")
    print("\\begin{tabular}{lrrrr}")
    print("\\hline")
    print("Environment & TD3 & SAC & PPO & Random \\\\")
    print("\\hline")
    for env in ENVS:
        row_vals = []
        for algo in ALGOS:
            mean, std = data[env][algo]
            row_vals.append(f"{mean:.0f} $\\pm$ {std:.0f}" if mean is not None else "N/A")
        env_short = env.replace("-v5", "")
        print(f"{env_short} & {' & '.join(row_vals)} \\\\")
    print("\\hline")
    print("\\end{tabular}")


def main():
    parser = argparse.ArgumentParser(description="Generate summary comparison table")
    parser.add_argument("--results-dir", type=str, default="results/main")
    parser.add_argument("--output",      type=str, default="results/figures")
    parser.add_argument("--seeds",  nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    generate_table(args.results_dir, args.output, args.seeds)


if __name__ == "__main__":
    main()
