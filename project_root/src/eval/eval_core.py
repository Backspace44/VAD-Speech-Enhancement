"""Core evaluation logic for speech enhancement methods."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from src.dsp.metrics import evaluate_speech_enhancement
from src.models import load_masknet_checkpoint
from src.utils.audio_io import align_waveforms, find_matched_file_pairs, load_audio_mono
from src.utils.enhancement_pipeline import build_enhancement_methods, run_enhancement_methods

METRIC_NAMES = ["pesq", "stoi", "snr", "segsnr", "lsd"]


def evaluate_single_file(
    clean_path: Path,
    noisy_path: Path,
    model,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all methods on a single file."""
    clean_audio, sr_clean = load_audio_mono(clean_path)
    noisy_audio, _ = load_audio_mono(noisy_path)
    clean_audio, noisy_audio = align_waveforms(clean_audio, noisy_audio)

    results = {}
    enhanced_outputs = run_enhancement_methods(noisy_audio, model=model, device=device)
    for method_name, enhanced_audio in enhanced_outputs.items():
        aligned_clean, aligned_enhanced = align_waveforms(clean_audio, enhanced_audio)
        results[method_name] = evaluate_speech_enhancement(aligned_clean, aligned_enhanced, sr_clean)
    return results


def evaluate_dataset(
    clean_dir: Path,
    noisy_dir: Path,
    model_path: Path,
    max_files: int = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Evaluate all methods on entire dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    if model_path and model_path.exists():
        print(f"Loading model from: {model_path}")
        model = load_masknet_checkpoint(model_path, device)
        print(f"Model loaded on {device}")
    else:
        print("No model provided, evaluating only classical methods")

    matched_pairs = find_matched_file_pairs(
        clean_dir,
        noisy_dir,
        clean_patterns=("*.wav",),
        noisy_patterns=("*.wav",),
    )
    if max_files:
        matched_pairs = matched_pairs[:max_files]

    print(f"\nEvaluating {len(matched_pairs)} file pairs...")

    methods = list(build_enhancement_methods(model=model, device=device).keys())
    all_results = {method: {metric: [] for metric in METRIC_NAMES} for method in methods}

    for clean_path, noisy_path in tqdm(matched_pairs, desc="Evaluating"):
        try:
            file_results = evaluate_single_file(clean_path, noisy_path, model, device)
            for method, metrics in file_results.items():
                for metric_name in METRIC_NAMES:
                    if metric_name in metrics:
                        all_results[method][metric_name].append(metrics[metric_name])
        except Exception as exc:
            print(f"\nError processing {noisy_path.name}: {exc}")
            continue

    return all_results


def compute_statistics(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute mean, std, median for all metrics."""
    stats = {}
    for method, metrics in results.items():
        stats[method] = {}
        for metric_name, values in metrics.items():
            if len(values) > 0:
                stats[method][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
    return stats


def compute_improvements(stats: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Compute improvement over noisy baseline for each method."""
    improvements = {}
    noisy_baseline = stats.get("Noisy", {})

    for method, metrics in stats.items():
        if method == "Noisy":
            continue
        improvements[method] = {}
        for metric_name, metric_stats in metrics.items():
            if metric_name in noisy_baseline:
                noisy_mean = noisy_baseline[metric_name]["mean"]
                enhanced_mean = metric_stats["mean"]
                improvements[method][metric_name] = enhanced_mean - noisy_mean

    return improvements
