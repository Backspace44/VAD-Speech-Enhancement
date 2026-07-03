"""Compare LibriSpeech VAD label sets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src import config
from src.data_prep.dataset import SpeechEnhancementDataset
from src.experiments import save_csv_rows, save_json
from src.utils.logger import setup_script_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare no-VAD, hard VAD, and soft VAD target-mask behavior on LibriSpeech"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="LibriSpeech split to analyze",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum number of matched files to analyze per setting",
    )
    parser.add_argument(
        "--hard-label-set",
        type=str,
        default="adaptive_v1",
        help="Binary VAD label set to compare",
    )
    parser.add_argument(
        "--soft-label-set",
        type=str,
        default="adaptive_soft_v1",
        help="Soft VAD label set to compare",
    )
    parser.add_argument(
        "--soft-mask-floor",
        type=float,
        default=0.15,
        help="Minimum scaling used for soft VAD gating",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/vad_label_set_comparison",
        help="Directory for CSV/JSON reports",
    )
    return parser.parse_args()


def resolve_split_dirs(split: str) -> tuple[Path, Path]:
    if split == "train":
        return (
            config.LIBRISPEECH_ROOT / "clean" / "train" / "train-clean-100",
            config.LIBRISPEECH_ROOT / "noise" / "train",
        )
    if split == "val":
        return (
            config.LIBRISPEECH_ROOT / "clean" / "val" / "dev-clean",
            config.LIBRISPEECH_ROOT / "noise" / "val",
        )
    return (
        config.LIBRISPEECH_ROOT / "clean" / "test" / "test-clean",
        config.LIBRISPEECH_ROOT / "noise" / "test",
    )


def build_dataset(
    split: str,
    max_files: int,
    vad_dir: Path | None,
    use_vad_labels: bool,
    vad_soft_mask_floor: float,
) -> SpeechEnhancementDataset:
    clean_dir, noise_dir = resolve_split_dirs(split)
    return SpeechEnhancementDataset(
        clean_dir=clean_dir,
        noise_dir=noise_dir,
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LEN,
        win_length=config.FRAME_LEN,
        max_length=config.DATASET_CONFIG["max_length_samples"],
        augmentation=None,
        preprocessing=None,
        cache_in_memory=False,
        return_audio=False,
        max_samples=max_files,
        vad_dir=vad_dir,
        use_vad_labels=use_vad_labels,
        vad_soft_mask_floor=vad_soft_mask_floor,
        deterministic_mixing=True,
        random_seed=config.RANDOM_SEED,
    )


def summarize_mask(mask: torch.Tensor) -> dict[str, float]:
    return {
        "mask_mean": float(mask.mean()),
        "mask_std": float(mask.std()),
        "mask_min": float(mask.min()),
        "mask_max": float(mask.max()),
        "zero_ratio": float((mask <= 1e-7).float().mean()),
        "low_ratio_below_0_1": float((mask < 0.1).float().mean()),
        "high_ratio_above_0_5": float((mask > 0.5).float().mean()),
    }


def summarize_labels(vad_path: Path | None, clean_stem: str) -> dict[str, float]:
    if vad_path is None:
        return {
            "label_mean": 1.0,
            "label_std": 0.0,
            "label_min": 1.0,
            "label_max": 1.0,
        }

    label = np.load(vad_path / f"{clean_stem}.npy")
    label = label.astype(np.float32)
    return {
        "label_mean": float(label.mean()),
        "label_std": float(label.std()),
        "label_min": float(label.min()),
        "label_max": float(label.max()),
    }


def analyze_dataset(
    name: str,
    dataset: SpeechEnhancementDataset,
    vad_path: Path | None,
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    rows: list[dict[str, float | str]] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        clean_stem = dataset.clean_files[idx].stem
        row: dict[str, float | str] = {
            "setting": name,
            "filename": item["filename"],
            "clean_stem": clean_stem,
        }
        row.update(summarize_mask(item["ideal_mask"]))
        row.update(summarize_labels(vad_path, clean_stem))
        rows.append(row)

    numeric_keys = [key for key in rows[0].keys() if key not in {"setting", "filename", "clean_stem"}]
    summary = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in numeric_keys
    }
    summary["num_files"] = float(len(rows))
    return rows, summary

def main() -> int:
    args = parse_args()
    logger = logging.getLogger("compare_vad_label_sets")
    if not logger.handlers:
        logger = setup_script_logger("compare_vad_label_sets")

    hard_vad_dir = config.LIBRISPEECH_VAD_ROOT / args.hard_label_set / args.split
    soft_vad_dir = config.LIBRISPEECH_VAD_ROOT / args.soft_label_set / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = [
        ("no_vad", None, False, 0.0),
        (args.hard_label_set, hard_vad_dir, True, 0.0),
        (args.soft_label_set, soft_vad_dir, True, args.soft_mask_floor),
    ]

    logger.info("Comparing VAD label sets on LibriSpeech target masks")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max files: {args.max_files}")
    logger.info(f"Hard labels: {hard_vad_dir}")
    logger.info(f"Soft labels: {soft_vad_dir}")

    all_rows: list[dict[str, float | str]] = []
    summaries: dict[str, dict[str, float]] = {}

    for name, vad_dir, use_vad_labels, soft_floor in settings:
        logger.info(f"Analyzing setting: {name}")
        dataset = build_dataset(
            split=args.split,
            max_files=args.max_files,
            vad_dir=vad_dir,
            use_vad_labels=use_vad_labels,
            vad_soft_mask_floor=soft_floor,
        )
        rows, summary = analyze_dataset(name, dataset, vad_dir)
        all_rows.extend(rows)
        summaries[name] = summary
        logger.info(
            f"{name}: mask_mean={summary['mask_mean']:.4f}, "
            f"zero_ratio={summary['zero_ratio']:.4f}, "
            f"high_ratio_above_0_5={summary['high_ratio_above_0_5']:.4f}"
        )

    baseline = summaries["no_vad"]
    delta_summary: dict[str, dict[str, float]] = {}
    for name, summary in summaries.items():
        if name == "no_vad":
            continue
        delta_summary[name] = {
            f"delta_{key}": float(summary[key] - baseline[key])
            for key in baseline.keys()
            if key != "num_files"
        }

    save_csv_rows(output_dir / "vad_label_set_rows.csv", all_rows)
    save_csv_rows(
        output_dir / "vad_label_set_summary.csv",
        [
            {"setting": name, **summary}
            for name, summary in summaries.items()
        ],
    )
    save_json(
        output_dir / "vad_label_set_summary.json",
        {
            "split": args.split,
            "max_files": args.max_files,
            "hard_label_set": args.hard_label_set,
            "soft_label_set": args.soft_label_set,
            "soft_mask_floor": args.soft_mask_floor,
            "summaries": summaries,
            "deltas_vs_no_vad": delta_summary,
        },
    )

    logger.info(f"Saved reports to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
