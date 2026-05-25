"""Project inventory helpers for delivery-ready artifact organization."""

from __future__ import annotations

from pathlib import Path


OFFICIAL_CHECKPOINTS = {
    "voicebank_quality_final_checkpoint_v2": "official_voicebank_best",
    "voicebank_quality_final_checkpoint_v1": "official_voicebank_previous",
    "librispeech_fast_benchmark_smoke": "official_librispeech_fast_benchmark",
    "voicebank_baseline_official_smoke": "official_voicebank_baseline",
}

OFFICIAL_RESULTS = {
    "final_benchmark_report_v2": "official_benchmark_report",
    "voicebank_quality_v2_vs_baseline": "official_voicebank_comparison",
}


def classify_checkpoint_artifact(name: str) -> tuple[str, str]:
    """Return class and status for a checkpoint artifact name."""
    if name in OFFICIAL_CHECKPOINTS:
        return "official", OFFICIAL_CHECKPOINTS[name]
    if name in {"masknet_best.pth", "masknet_interrupted.pth"}:
        return "legacy_root", name.replace(".pth", "")
    lowered = name.lower()
    if "smoke" in lowered or "1batch" in lowered or "2s_" in lowered or "1s_" in lowered:
        return "smoke", "short_validation_run"
    if "resume" in lowered:
        return "resume", "continued_training"
    if lowered.startswith("train_"):
        return "legacy", "historical_training_run"
    if "baseline" in lowered or "quality_final" in lowered or "recommended" in lowered:
        return "benchmark", "named_experiment"
    return "other", "unclassified"


def classify_result_artifact(name: str) -> tuple[str, str]:
    """Return class and status for a results artifact name."""
    if name in OFFICIAL_RESULTS:
        return "official", OFFICIAL_RESULTS[name]
    lowered = name.lower()
    if name == "audio_samples":
        return "demo", "recorded_demo_audio"
    if "realtime" in lowered:
        return "demo", "realtime_demo_artifact"
    if "final_benchmark_report" in lowered:
        return "benchmark", "aggregated_benchmark_report"
    if "comparison" in lowered:
        return "benchmark", "comparison_report"
    if "vad" in lowered:
        return "vad", "vad_evaluation"
    if "smoke" in lowered or "test5" in lowered or "test20" in lowered:
        return "smoke", "short_validation_report"
    if name.endswith(".csv") or name.endswith(".json"):
        return "tabular", "top_level_export"
    return "other", "unclassified"


def build_inventory_rows(root: Path, target_dir: Path, kind: str) -> list[dict]:
    """Scan one artifact root and build normalized inventory rows."""
    rows: list[dict] = []
    if not target_dir.exists():
        return rows

    for item in sorted(target_dir.iterdir(), key=lambda path: path.name.lower()):
        item_class, status = (
            classify_checkpoint_artifact(item.name)
            if kind == "checkpoint"
            else classify_result_artifact(item.name)
        )
        rows.append(
            {
                "artifact_kind": kind,
                "name": item.name,
                "path": str(item.resolve()),
                "is_dir": item.is_dir(),
                "classification": item_class,
                "status": status,
            }
        )
    return rows


def build_delivery_recommendations(root: Path) -> list[dict]:
    """List the main artifacts that should be cited in the thesis or demo."""
    recommendations = [
        (
            "checkpoint",
            "voicebank_quality_final_checkpoint_v2/masknet_best.pth",
            "Main VoiceBank checkpoint",
        ),
        (
            "checkpoint",
            "voicebank_baseline_official_smoke/masknet_best.pth",
            "VoiceBank baseline reference checkpoint",
        ),
        (
            "result",
            "final_benchmark_report_v2/benchmark_best_by_dataset.csv",
            "Final best-by-dataset benchmark table",
        ),
        (
            "result",
            "voicebank_quality_v2_vs_baseline/checkpoint_comparison.csv",
            "VoiceBank upgraded-vs-baseline comparison",
        ),
        (
            "result",
            "audio_samples",
            "Realtime demo recordings root",
        ),
    ]

    rows: list[dict] = []
    for artifact_kind, relative_path, purpose in recommendations:
        base_dir = root / ("checkpoints" if artifact_kind == "checkpoint" else "results")
        path = base_dir / relative_path
        rows.append(
            {
                "artifact_kind": artifact_kind,
                "purpose": purpose,
                "path": str(path.resolve()),
                "exists": path.exists(),
            }
        )
    return rows
