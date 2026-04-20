"""
Run hyperparameter sensitivity experiments.

For each algorithm we sweep two hyperparameters:
  - Learning rate (all three algorithms)
  - Algorithm-specific: exploration noise (TD3), entropy coefficient (SAC), clip range (PPO)

We use 3 seeds (less than the main 5) and run on all 3 environments.
The default value for each hyperparameter is already covered by the main experiments,
so this only adds the non-default configurations.

Usage:
    python scripts/run_sensitivity.py
    python scripts/run_sensitivity.py --dry-run
    python scripts/run_sensitivity.py --sweep td3_lr         # only TD3 LR sweep
    python scripts/run_sensitivity.py --envs HalfCheetah-v5  # one env only (faster)
    python scripts/run_sensitivity.py --seeds 0 1 2
"""

import argparse
import subprocess
import sys
import os
import time


ENVS  = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
SEEDS = [0, 1, 2]  # 3 seeds for sensitivity (5 is used in main experiments)

# Default values — these runs already exist from the main experiments, so we skip them
TD3_LR_DEFAULT    = 3e-4
SAC_LR_DEFAULT    = 3e-4
PPO_LR_DEFAULT    = 1e-4
TD3_NOISE_DEFAULT = 0.1
PPO_CLIP_DEFAULT  = 0.2

# All sweeps: (sweep_name, algo, param_name, param_values, skip_defaults)
SWEEPS = {
    "td3_lr": {
        "algo":   "td3",
        "param":  "learning_rate",
        "values": [1e-4, 3e-4, 1e-3],
        "default": TD3_LR_DEFAULT,
    },
    "td3_noise": {
        "algo":   "td3",
        "param":  "exploration_noise",
        "values": [0.05, 0.1, 0.2, 0.3],
        "default": TD3_NOISE_DEFAULT,
    },
    "sac_lr": {
        "algo":   "sac",
        "param":  "learning_rate",
        "values": [1e-4, 3e-4, 1e-3],
        "default": SAC_LR_DEFAULT,
    },
    "sac_entropy": {
        "algo":   "sac",
        "param":  "alpha_mode",  # special handling below
        "values": ["auto", 0.1, 0.5, 1.0],
        "default": "auto",
    },
    "ppo_lr": {
        "algo":   "ppo",
        "param":  "learning_rate",
        "values": [1e-4, 3e-4, 1e-3],
        "default": PPO_LR_DEFAULT,
    },
    "ppo_clip": {
        "algo":   "ppo",
        "param":  "clip_range",
        "values": [0.1, 0.2, 0.3, 0.4],
        "default": PPO_CLIP_DEFAULT,
    },
}


def format_value(v):
    """Format a value for use in a directory name and as a CLI arg."""
    if isinstance(v, float):
        # Use scientific notation for very small/large floats, e.g. 1e-4 -> 1e-4
        if v < 0.01 or v >= 100:
            return f"{v:.0e}"
        return str(v)
    return str(v)


def run_command(cmd: list, dry_run: bool = False) -> bool:
    cmd_str = " ".join(cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {cmd_str}")

    if dry_run:
        return True

    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR (return code {result.returncode})")
        return False

    print(f"  Done in {elapsed/60:.1f} min")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sensitivity experiments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sweep", nargs="+", default=list(SWEEPS.keys()),
                        choices=list(SWEEPS.keys()), help="Which sweeps to run")
    parser.add_argument("--envs",  nargs="+", default=ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-defaults", action="store_true",
                        help="Also run default values (normally skipped since main exps cover them)")
    args = parser.parse_args()

    failed = []
    total = 0
    completed = 0

    for sweep_name in args.sweep:
        sweep = SWEEPS[sweep_name]
        algo  = sweep["algo"]

        print(f"\n{'='*60}")
        print(f"Sweep: {sweep_name} | algo={algo} | param={sweep['param']}")
        print(f"Values: {sweep['values']}")

        for value in sweep["values"]:
            # Skip default unless asked
            if not args.include_defaults and value == sweep["default"]:
                print(f"  Skipping default value: {value}")
                continue

            value_str = format_value(value)
            exp_group = f"sensitivity/{sweep_name}/{value_str}"

            for env in args.envs:
                for seed in args.seeds:
                    total += 1
                    result_path = os.path.join("results", exp_group, algo, env,
                                               f"seed_{seed}", "eval_log.csv")
                    if args.skip_existing and os.path.exists(result_path):
                        print(f"  Skipping {sweep_name}/{value_str}/{env}/seed_{seed}")
                        completed += 1
                        continue

                    # Build the training command with config override
                    cmd = [
                        sys.executable, "scripts/train.py",
                        "--algo",      algo,
                        "--env",       env,
                        "--seed",      str(seed),
                        "--exp-group", exp_group,
                    ]

                    # Handle special SAC entropy sweep
                    if sweep_name == "sac_entropy":
                        if value == "auto":
                            cmd += ["--override", "auto_tune_alpha=1"]
                        else:
                            cmd += ["--override", f"auto_tune_alpha=0",
                                    "--override", f"initial_alpha={value}"]
                    else:
                        cmd += ["--override", f"{sweep['param']}={value}"]

                    success = run_command(cmd, dry_run=args.dry_run)
                    if not success:
                        failed.append(f"{sweep_name}/{value_str}/{env}/seed_{seed}")
                    else:
                        completed += 1

    print(f"\n{'='*60}")
    print(f"Sensitivity experiments complete: {completed}/{total}")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
