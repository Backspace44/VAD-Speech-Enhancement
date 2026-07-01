"""
Aggregate experiment summary.json files into a final benchmark report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.experiments import (
    build_benchmark_row,
    load_experiment_summary,
    pick_best_rows_by_key,
    save_csv_rows,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark experiment summaries into CSV/JSON")
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Experiment directory name under checkpoints/ to include. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/final_benchmark_report",
        help="Directory for aggregated benchmark outputs",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        load_experiment_summary(config.CHECKPOINTS_DIR, experiment_name)
        for experiment_name in args.experiment
    ]
    rows = [build_benchmark_row(summary) for summary in summaries]
    best_by_dataset = pick_best_rows_by_key(rows, key_field="dataset", metric_field="best_val_loss")

    save_csv_rows(output_dir / "benchmark_summary.csv", rows)
    save_csv_rows(output_dir / "benchmark_best_by_dataset.csv", list(best_by_dataset.values()))
    save_json(
        output_dir / "benchmark_summary.json",
        {
            "experiments": rows,
            "best_by_dataset": best_by_dataset,
        },
    )

    print(f"Saved benchmark report to: {output_dir}")
    for dataset, row in best_by_dataset.items():
        print(
            f"{dataset}: best={row['experiment_name']} "
            f"(model={row['model_variant']}, val_loss={row['best_val_loss']:.6f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
