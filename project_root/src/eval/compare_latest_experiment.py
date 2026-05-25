"""
Compare the latest experiment checkpoint against a reference checkpoint.
Useful right after training a new model run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.eval.compare_checkpoints import parse_checkpoint_spec


def find_latest_experiment_checkpoint(checkpoints_dir: Path) -> tuple[Path, Path]:
    """Return latest experiment dir plus its best checkpoint path."""
    candidates = []
    for exp_dir in checkpoints_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        checkpoint_path = exp_dir / "masknet_best.pth"
        if checkpoint_path.exists():
            candidates.append((exp_dir, checkpoint_path))

    if not candidates:
        raise FileNotFoundError(f"No experiment directories with masknet_best.pth found under {checkpoints_dir}")

    latest_exp_dir, latest_checkpoint = max(candidates, key=lambda item: item[0].stat().st_mtime)
    return latest_exp_dir, latest_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Compare latest experiment checkpoint with a reference checkpoint")
    parser.add_argument("--clean-dir", type=str, required=True, help="Directory with clean audio files")
    parser.add_argument("--noisy-dir", type=str, required=True, help="Directory with noisy audio files")
    parser.add_argument(
        "--reference",
        type=str,
        default="baseline=checkpoints/masknet_best.pth",
        help="Reference checkpoint in LABEL=PATH format",
    )
    parser.add_argument(
        "--latest-label",
        type=str,
        default="latest_experiment",
        help="Label used for the latest experiment checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/latest_experiment_comparison",
        help="Directory where comparison reports will be written",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of file pairs to evaluate")
    args = parser.parse_args()

    _, reference_path = parse_checkpoint_spec(args.reference)
    reference_label = args.reference.split("=", 1)[0].strip()
    latest_exp_dir, latest_checkpoint = find_latest_experiment_checkpoint(config.CHECKPOINTS_DIR)

    from src.eval.compare_checkpoints import main as compare_main
    import sys

    original_argv = sys.argv[:]
    sys.argv = [
        "compare_checkpoints.py",
        "--clean-dir", args.clean_dir,
        "--noisy-dir", args.noisy_dir,
        "--checkpoint", f"{reference_label}={reference_path}",
        "--checkpoint", f"{args.latest_label}={latest_checkpoint}",
        "--output-dir", args.output_dir,
    ]
    if args.max_files is not None:
        sys.argv.extend(["--max-files", str(args.max_files)])

    print(f"Latest experiment directory: {latest_exp_dir}")
    print(f"Latest experiment checkpoint: {latest_checkpoint}")
    try:
        compare_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
