"""
Run all main experiments: 3 algorithms × 3 environments × 5 seeds.

This launches training runs sequentially (one at a time) to avoid memory issues.
Each run takes ~30-45 min on a GPU, so the full suite takes a while.

If you want to run in parallel (multiple GPUs or CPU cores), you can run
individual train.py commands manually in separate terminals.

Usage:
    python scripts/run_main_experiments.py
    python scripts/run_main_experiments.py --dry-run   # just print commands, don't run
    python scripts/run_main_experiments.py --seeds 0 1 2  # only run these seeds
    python scripts/run_main_experiments.py --algos td3 sac  # only these algos
    python scripts/run_main_experiments.py --envs HalfCheetah-v5  # only this env
"""

import argparse
import subprocess
import sys
import os
import time


# The three main environments, from easiest to hardest
ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]

# All three algorithms
ALGOS = ["td3", "sac", "ppo"]

# 5 seeds for statistical significance
SEEDS = [0, 1, 2, 3, 4]


def run_command(cmd: list, dry_run: bool = False) -> bool:
    """Run a command and return True if it succeeded."""
    cmd_str = " ".join(cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {cmd_str}")

    if dry_run:
        return True

    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR: command failed (return code {result.returncode})")
        return False

    print(f"  Completed in {elapsed/60:.1f} min")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all main comparison experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--algos", nargs="+", default=ALGOS,
                        choices=ALGOS, help="Algorithms to run")
    parser.add_argument("--envs", nargs="+", default=ENVS,
                        help="Environments to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Seeds to run")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs where eval_log.csv already exists")
    args = parser.parse_args()

    total_runs = len(args.algos) * len(args.envs) * len(args.seeds)
    print(f"Main experiments: {len(args.algos)} algos × {len(args.envs)} envs × {len(args.seeds)} seeds = {total_runs} runs")
    print(f"Algorithms:   {args.algos}")
    print(f"Environments: {args.envs}")
    print(f"Seeds:        {args.seeds}")

    failed = []
    completed = 0

    for algo in args.algos:
        for env in args.envs:
            for seed in args.seeds:
                # Check if already done
                result_path = os.path.join("results", "main", algo, env, f"seed_{seed}", "eval_log.csv")
                if args.skip_existing and os.path.exists(result_path):
                    print(f"  Skipping {algo}/{env}/seed_{seed} (already exists)")
                    completed += 1
                    continue

                cmd = [
                    sys.executable, "scripts/train.py",
                    "--algo", algo,
                    "--env", env,
                    "--seed", str(seed),
                    "--exp-group", "main",
                ]

                success = run_command(cmd, dry_run=args.dry_run)
                if not success:
                    failed.append(f"{algo}/{env}/seed_{seed}")
                else:
                    completed += 1

                print(f"Progress: {completed}/{total_runs} completed, {len(failed)} failed")

    print(f"\n{'='*50}")
    print(f"Main experiments complete!")
    print(f"  Completed: {completed}/{total_runs}")
    if failed:
        print(f"  Failed runs:")
        for f in failed:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
