"""
Grouped bar chart of final-episode mean return (± std across 5 seeds)
for each (algorithm, environment) pair, including the random-policy baseline.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENVS  = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
ALGOS = ["random", "ppo", "td3", "sac"]

ALGO_LABEL = {
    "random": "Random",
    "ppo":    "PPO",
    "td3":    "TD3",
    "sac":    "SAC",
}

ALGO_COLOR = {
    "random": "#999999",
    "ppo":    "#e41a1c",
    "td3":    "#377eb8",
    "sac":    "#4daf4a",
}

SEEDS = [0, 1, 2, 3, 4]


def final_reward(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df["mean_reward"].iloc[-1])


def collect(results_base: str, algo: str, env: str):
    rewards = []
    for s in SEEDS:
        path = os.path.join(results_base, "main", algo, env, f"seed_{s}", "eval_log.csv")
        r = final_reward(path)
        if r is not None:
            rewards.append(r)
    if not rewards:
        return None, None
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    results_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
    )
    out_path = os.path.join(results_base, "figures", "final_performance_bars.png")

    means = {algo: [] for algo in ALGOS}
    stds  = {algo: [] for algo in ALGOS}

    for env in ENVS:
        for algo in ALGOS:
            m, s = collect(results_base, algo, env)
            means[algo].append(m if m is not None else 0.0)
            stds[algo].append(s if s is not None else 0.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    n_groups = len(ENVS)
    n_algos  = len(ALGOS)
    bar_w    = 0.2
    x        = np.arange(n_groups)

    for i, algo in enumerate(ALGOS):
        offset = (i - (n_algos - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            means[algo],
            width=bar_w,
            yerr=stds[algo],
            capsize=3,
            color=ALGO_COLOR[algo],
            label=ALGO_LABEL[algo],
            edgecolor="black",
            linewidth=0.5,
            error_kw=dict(ecolor="black", linewidth=1.0),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("-v5", "") for e in ENVS], fontsize=11)
    ax.set_ylabel("Final Evaluation Return (mean ± std, 5 seeds)", fontsize=11)
    ax.set_title("Final Performance After 1 M Environment Steps", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left", frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    print("\nNumeric summary:")
    for env_i, env in enumerate(ENVS):
        print(f"  {env}")
        for algo in ALGOS:
            print(f"    {ALGO_LABEL[algo]:>7s}: {means[algo][env_i]:>9.1f} ± {stds[algo][env_i]:>7.1f}")


if __name__ == "__main__":
    main()
