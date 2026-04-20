"""
Run all plotting scripts in sequence to generate every figure for the report.

Usage:
    python plotting/plot_all.py
    python plotting/plot_all.py --output results/figures
"""

import argparse
import subprocess
import sys
import os


def run(script, extra_args=None, dry_run=False):
    cmd = [sys.executable, script]
    if extra_args:
        cmd += extra_args
    cmd_str = " ".join(cmd)
    print(f"\n{'[DRY]' if dry_run else '>>>'}  {cmd_str}")
    if not dry_run:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"  WARNING: {script} exited with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Generate all report figures")
    parser.add_argument("--output",      type=str, default="results/figures")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seeds-main",  nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds-sens",  nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds_main_str = [str(s) for s in args.seeds_main]
    seeds_sens_str = [str(s) for s in args.seeds_sens]

    base = "plotting"
    out  = args.output
    main_dir = os.path.join(args.results_dir, "main")

    print("=" * 60)
    print("Generating all figures for the report...")
    print(f"Output directory: {out}")
    print("=" * 60)

    # 1. Learning curves (one per environment)
    run(f"{base}/plot_learning_curves.py",
        ["--results-dir", main_dir, "--output", out, "--seeds"] + seeds_main_str,
        args.dry_run)

    # 2. Sample efficiency
    run(f"{base}/plot_sample_efficiency.py",
        ["--results-dir", main_dir, "--output", out, "--seeds"] + seeds_main_str,
        args.dry_run)

    # 3. Summary comparison table
    run(f"{base}/plot_comparison_table.py",
        ["--results-dir", main_dir, "--output", out, "--seeds"] + seeds_main_str,
        args.dry_run)

    # 4. Sensitivity plots (one per sweep)
    run(f"{base}/plot_sensitivity.py",
        ["--results-dir", args.results_dir, "--output", out, "--seeds"] + seeds_sens_str,
        args.dry_run)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
