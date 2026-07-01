"""Generate a delivery-focused inventory of checkpoints and results artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.experiments import (
    build_delivery_recommendations,
    build_inventory_rows,
    save_csv_rows,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a delivery-ready inventory for checkpoints and results artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.RESULTS_DIR / "project_inventory",
        help="Directory where the inventory files will be written",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rows = build_inventory_rows(config.ROOT, config.CHECKPOINTS_DIR, kind="checkpoint")
    result_rows = build_inventory_rows(config.ROOT, config.RESULTS_DIR, kind="result")
    all_rows = checkpoint_rows + result_rows
    recommendations = build_delivery_recommendations(config.ROOT)

    save_csv_rows(output_dir / "artifact_inventory.csv", all_rows)
    save_json(
        output_dir / "artifact_inventory.json",
        {
            "checkpoints": checkpoint_rows,
            "results": result_rows,
            "recommended_artifacts": recommendations,
        },
    )
    save_csv_rows(output_dir / "recommended_artifacts.csv", recommendations)

    print(f"Saved project inventory to: {output_dir}")
    print(f"Checkpoint artifacts: {len(checkpoint_rows)}")
    print(f"Result artifacts: {len(result_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
