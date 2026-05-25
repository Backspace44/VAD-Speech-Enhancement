"""
Unified evaluation script for all enhancement methods.
Compares classical methods (Spectral Subtraction, Wiener Filter) and MaskNet against noisy baseline.
Outputs JSON results and PNG plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.eval_core import (
    METRIC_NAMES,
    compute_improvements,
    compute_statistics,
    evaluate_dataset,
    evaluate_single_file,
)
from src.eval.eval_plots import plot_results
from src.eval.eval_reporting import (
    improvements_to_dataframe,
    print_summary_clean,
    save_results,
    stats_to_dataframe,
)


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for all enhancement methods"
    )
    parser.add_argument(
        "--clean-dir",
        type=str,
        required=True,
        help="Directory with clean audio files",
    )
    parser.add_argument(
        "--noisy-dir",
        type=str,
        required=True,
        help="Directory with noisy audio files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/masknet_best.pth",
        help="Path to MaskNet checkpoint (optional for classical methods only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to evaluate",
    )

    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    model_path = Path(args.model) if args.model else None
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print(" Speech Enhancement Evaluation")
    print("=" * 70)
    print(f"\nClean dir:  {clean_dir}")
    print(f"Noisy dir:  {noisy_dir}")
    print(f"Model:      {model_path}")
    print(f"Output dir: {output_dir}")

    results = evaluate_dataset(clean_dir, noisy_dir, model_path, args.max_files)
    stats = compute_statistics(results)
    improvements = compute_improvements(stats)

    save_results(stats, improvements, output_dir)
    plot_results(stats, improvements, output_dir)
    print_summary_clean(stats, improvements)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
