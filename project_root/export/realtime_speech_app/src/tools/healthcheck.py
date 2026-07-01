"""
Project health check for the stage-0 upgrade.

Usage:
    python -m src.tools.healthcheck
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from src import config


REQUIRED_PACKAGES = (
    "numpy",
    "scipy",
    "soundfile",
    "matplotlib",
    "torch",
    "torchaudio",
    "pandas",
    "tqdm",
    "colorama",
)


OPTIONAL_PACKAGES = (
    "pesq",
    "pystoi",
    "seaborn",
    "librosa",
    "tensorboard",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate project environment and local assets.")
    parser.add_argument(
        "--dataset",
        choices=["voicebank", "librispeech", "none"],
        default="librispeech",
        help="Dataset layout to validate. LibriSpeech is used for training; VoiceBank is benchmark-only.",
    )
    return parser.parse_args()


def import_status(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, "ok"
    except Exception as exc:  # pragma: no cover - defensive CLI reporting
        return False, f"{type(exc).__name__}: {exc}"


def print_section(title: str):
    print(f"\n[ {title} ]")


def validate_paths(dataset_name: str) -> int:
    checks_failed = 0

    print_section("Paths")
    required_dirs = [
        config.ROOT,
        config.CHECKPOINTS_DIR,
        config.LOGS_DIR,
        config.RESULTS_DIR,
    ]

    if dataset_name == "voicebank":
        required_dirs.extend([
            config.VOICEBANK_ROOT / "clean" / "test",
            config.VOICEBANK_ROOT / "noisy" / "test",
        ])
    elif dataset_name == "librispeech":
        required_dirs.extend([
            config.LIBRISPEECH_ROOT / "clean" / "train" / "train-clean-100",
            config.LIBRISPEECH_ROOT / "noise" / "train",
            config.LIBRISPEECH_ROOT / "clean" / "val" / "dev-clean",
            config.LIBRISPEECH_ROOT / "noise" / "val",
        ])

    for path in required_dirs:
        status = "OK" if Path(path).exists() else "MISSING"
        print(f"{status:8} {path}")
        if status != "OK":
            checks_failed += 1

    return checks_failed


def validate_packages() -> int:
    checks_failed = 0

    print_section("Required Packages")
    for module_name in REQUIRED_PACKAGES:
        ok, message = import_status(module_name)
        print(f"{'OK' if ok else 'MISSING':8} {module_name} - {message}")
        if not ok:
            checks_failed += 1

    print_section("Optional Packages")
    for module_name in OPTIONAL_PACKAGES:
        ok, message = import_status(module_name)
        print(f"{'OK' if ok else 'WARN':8} {module_name} - {message}")

    return checks_failed


def validate_runtime() -> int:
    checks_failed = 0

    print_section("Runtime")
    print(f"Python    {sys.version.split()[0]}")
    print(f"Device    {config.DEVICE}")
    print(f"GPUs      {config.NUM_GPUS}")
    print(f"Variant   {config.ACTIVE_MODEL_VARIANT}")
    print(f"Workers   {config.DATASET_CONFIG['num_workers']}")

    if config.DATASET_CONFIG["num_workers"] == 0 and config.DATASET_CONFIG["persistent_workers"]:
        print("WARN     persistent_workers should be False when num_workers=0")
        checks_failed += 1

    return checks_failed


def main() -> int:
    args = parse_args()
    failures = 0
    failures += validate_runtime()
    failures += validate_packages()
    if args.dataset != "none":
        failures += validate_paths(args.dataset)

    print_section("Summary")
    if failures == 0:
        print("OK       Project health check passed")
        return 0

    print(f"FAIL     Detected {failures} blocking issue(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
