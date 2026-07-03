"""Compare MaskNet checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.experiments import save_json
from src.eval.evaluate_all_methods import (
    compute_improvements,
    compute_statistics,
    evaluate_dataset,
    improvements_to_dataframe,
    print_summary_clean,
    save_results,
    stats_to_dataframe,
)


def parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    """Parse LABEL=PATH specs from CLI."""
    if "=" not in spec:
        raise ValueError(f"Checkpoint spec must be LABEL=PATH, got: {spec}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    checkpoint_path = Path(raw_path.strip())
    if not label:
        raise ValueError(f"Checkpoint label cannot be empty: {spec}")
    return label, checkpoint_path


def build_comparison_rows(label: str, stats: dict, improvements: dict) -> list[dict]:
    """Create flattened comparison rows for all methods and metrics."""
    stat_rows = stats_to_dataframe(stats)
    improvement_rows = improvements_to_dataframe(improvements)

    merged = stat_rows.merge(
        improvement_rows,
        on=["method", "metric"],
        how="left",
    )
    merged.insert(0, "run_label", label)
    return merged.to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple checkpoints on the same evaluation set")
    parser.add_argument("--clean-dir", type=str, required=True, help="Directory with clean audio files")
    parser.add_argument("--noisy-dir", type=str, required=True, help="Directory with noisy audio files")
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        required=True,
        help="Checkpoint in LABEL=PATH format. Repeat this flag for multiple runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/checkpoint_comparison",
        help="Directory where comparison reports will be written",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of file pairs to evaluate")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    summary = {"runs": {}}

    print("=" * 70)
    print(" CHECKPOINT COMPARISON")
    print("=" * 70)
    print(f"Clean dir:  {clean_dir}")
    print(f"Noisy dir:  {noisy_dir}")
    print(f"Output dir: {output_dir}")

    for checkpoint_spec in args.checkpoints:
        label, checkpoint_path = parse_checkpoint_spec(checkpoint_spec)
        run_output_dir = output_dir / label

        print("\n" + "-" * 70)
        print(f"Run: {label}")
        print(f"Checkpoint: {checkpoint_path}")

        results = evaluate_dataset(clean_dir, noisy_dir, checkpoint_path, args.max_files)
        stats = compute_statistics(results)
        improvements = compute_improvements(stats)
        save_results(stats, improvements, run_output_dir)
        print_summary_clean(stats, improvements)

        comparison_rows.extend(build_comparison_rows(label, stats, improvements))
        summary["runs"][label] = {
            "checkpoint": str(checkpoint_path),
            "stats": stats,
            "improvements": improvements,
        }

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "checkpoint_comparison.csv", index=False)

    masknet_summary = comparison_df[comparison_df["method"] == "MaskNet"].copy()
    if not masknet_summary.empty:
        pivot = masknet_summary.pivot_table(
            index="run_label",
            columns="metric",
            values="improvement",
            aggfunc="first",
        ).reset_index()
        pivot.to_csv(output_dir / "masknet_improvement_summary.csv", index=False)

    save_json(output_dir / "checkpoint_comparison.json", summary)

    print("\nSaved:")
    print(f"  {output_dir / 'checkpoint_comparison.csv'}")
    if not masknet_summary.empty:
        print(f"  {output_dir / 'masknet_improvement_summary.csv'}")
    print(f"  {output_dir / 'checkpoint_comparison.json'}")


if __name__ == "__main__":
    main()
