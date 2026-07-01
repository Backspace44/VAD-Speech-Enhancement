"""
Unified CLI for the speech enhancement project.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from src.tools.logged_runner import forward_args, run_logged_module, run_logged_path

ROOT = Path(__file__).resolve().parents[2]

COMMAND_TARGETS = {
    "train": ("module", "src.train.train_mask_model"),
    "eval-vad": ("logged_module", "src.eval.evaluate_vad_baseline", "evaluate_vad_baseline"),
    "compare-vad": ("logged_module", "src.eval.compare_vad_label_sets", "compare_vad_label_sets"),
    "eval-all": ("logged_module", "src.eval.evaluate_all_methods", "evaluate_all_methods"),
    "compare-checkpoints": ("logged_module", "src.eval.compare_checkpoints", "compare_checkpoints"),
    "compare-latest": ("logged_module", "src.eval.compare_latest_experiment", "compare_latest"),
    "summarize-benchmarks": ("logged_module", "src.eval.summarize_benchmarks", "summarize_benchmarks"),
    "project-inventory": ("logged_module", "src.tools.project_inventory", "project_inventory"),
    "healthcheck": ("module", "src.tools.healthcheck"),
    "demo": ("logged_path", str(ROOT / "demo.py"), "demo"),
    "realtime-demo": ("module", "src.tools.realtime_demo"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CLI for the speech enhancement project")
    parser.add_argument("command", choices=sorted(COMMAND_TARGETS.keys()), help="Command to run")
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected command. Prefix with '--' when needed.",
    )
    return parser.parse_args()


def normalize_forwarded_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main() -> int:
    parsed = parse_args()
    forwarded_args = normalize_forwarded_args(parsed.args)
    target_spec = COMMAND_TARGETS[parsed.command]
    target_type = target_spec[0]

    if target_type == "module":
        _, target = target_spec
        original_argv = sys.argv[:]
        sys.argv = [target, *forwarded_args]
        try:
            runpy.run_module(target, run_name="__main__")
        finally:
            sys.argv = original_argv
    elif target_type == "logged_module":
        _, target, logger_name = target_spec
        original_argv = forward_args(target, forwarded_args)
        try:
            run_logged_module(target, logger_name)
        finally:
            sys.argv = original_argv
    elif target_type == "logged_path":
        _, target, logger_name = target_spec
        original_argv = forward_args(target, forwarded_args)
        try:
            run_logged_path(target, logger_name)
        finally:
            sys.argv = original_argv
    else:
        raise ValueError(f"Unsupported command target type: {target_type}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
