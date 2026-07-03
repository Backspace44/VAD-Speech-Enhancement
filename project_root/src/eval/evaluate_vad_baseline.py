"""Evaluate the adaptive VAD detector."""

from pathlib import Path
import json
import argparse
from typing import Dict, List
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import config
from src.dsp.vad_energy_zcr import EnergyZCRVAD, EnergyZCRVADConfig
from src.dsp.metrics import evaluate_vad as eval_vad_metrics


def evaluate_on_file(
    noisy_path: Path,
    label_path: Path,
    vad_detector: EnergyZCRVAD
) -> Dict[str, float]:
    """Evaluate VAD on a single file."""
    noisy, sr = sf.read(noisy_path)
    
    if sr != config.SAMPLE_RATE:
        raise RuntimeError(
            f"Unexpected sample rate {sr} in {noisy_path}, expected {config.SAMPLE_RATE}"
        )
    
    if noisy.ndim > 1:
        noisy = np.mean(noisy, axis=1)
    
    vad_true = np.load(label_path).astype(np.uint8)
    
    vad_pred = vad_detector.predict(noisy)
    
    L = min(len(vad_pred), len(vad_true))
    if L == 0:
        return None
    
    vad_pred = vad_pred[:L]
    vad_true = vad_true[:L]
    
    metrics = eval_vad_metrics(vad_true, vad_pred)
    return metrics


def evaluate_librispeech_test_for_snr(
    snr_db: int,
    vad_detector: EnergyZCRVAD,
    verbose: bool = True,
    max_files: int | None = None,
    label_set: str = "adaptive_v1",
) -> Dict[str, float]:
    """Evaluate VAD for one SNR level."""
    root = config.LIBRISPEECH_ROOT
    noisy_dir = root / "noisy" / "test"
    labels_dir = config.LIBRISPEECH_VAD_ROOT / label_set / "test"
    
    pattern = f"*_snr{snr_db}dB.wav"
    noisy_files = sorted(noisy_dir.glob(pattern))
    
    if not noisy_files:
        raise RuntimeError(
            f"No noisy files for SNR={snr_db}dB found in {noisy_dir}"
        )
    
    if max_files is not None:
        noisy_files = noisy_files[:max_files]

    if verbose:
        print(f"\nEvaluating SNR={snr_db}dB: {len(noisy_files)} files")
    
    metrics_list: List[Dict[str, float]] = []
    
    iterator = tqdm(noisy_files, desc=f"SNR {snr_db}dB") if verbose else noisy_files
    
    for noisy_path in iterator:
        stem = noisy_path.stem
        label_path = labels_dir / f"{stem}.npy"
        
        if not label_path.exists():
            if verbose:
                print(f" No label file for {stem}, skipping")
            continue
        
        try:
            metrics = evaluate_on_file(noisy_path, label_path, vad_detector)
            if metrics:
                metrics_list.append(metrics)
        except Exception as e:
            if verbose:
                print(f" Error processing {stem}: {e}")
            continue
    
    if not metrics_list:
        raise RuntimeError(f"No files successfully evaluated for SNR={snr_db}dB")
    
    keys = metrics_list[0].keys()
    avg_metrics: Dict[str, float] = {}
    for k in keys:
        avg_metrics[k] = float(np.mean([m[k] for m in metrics_list]))
    
    return avg_metrics


def plot_vad_results(results: Dict[str, Dict[str, float]], output_dir: Path):
    """Generate plots for VAD evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snr_levels = sorted([int(k) for k in results.keys()])
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric in metrics:
        values = [results[str(snr)][metric] for snr in snr_levels]
        ax.plot(snr_levels, values, marker='o', label=metric.capitalize(), linewidth=2)
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Adaptive VAD Performance vs SNR', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vad_performance_vs_snr.png', dpi=150, bbox_inches='tight')
    print(f" Saved: vad_performance_vs_snr.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(snr_levels))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[str(snr)][metric] for snr in snr_levels]
        ax.bar(x + i * width, values, width, label=metric.capitalize(), alpha=0.8)
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Adaptive VAD Metrics by SNR Level', fontsize=14, weight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{snr}dB' for snr in snr_levels])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vad_metrics_bars.png', dpi=150, bbox_inches='tight')
    print(f" Saved: vad_metrics_bars.png")
    plt.close()


def print_results(results: Dict[str, Dict[str, float]]):
    """Print VAD evaluation results to console."""
    print("\n" + "="*70)
    print(" VAD EVALUATION RESULTS (Adaptive Detector)")
    print("="*70)
    
    for snr_db, metrics in sorted(results.items(), key=lambda x: int(x[0])):
        print(f"\n SNR = {snr_db} dB:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize():12s}: {value:.4f}")
    
    print(f"\n{'='*70}")
    print(" Average Across All SNRs:")
    print(f"{'='*70}")
    
    all_metrics = list(list(results.values())[0].keys())
    for metric in all_metrics:
        avg_value = np.mean([results[snr][metric] for snr in results.keys()])
        print(f"  {metric.capitalize():12s}: {avg_value:.4f}")
    
    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the adaptive VAD detector on LibriSpeech test set'
    )
    parser.add_argument(
        '--snr-levels',
        type=int,
        nargs='+',
        default=[0, 5, 10],
        help='SNR levels to evaluate (default: 0 5 10)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/vad_evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--energy-thresh',
        type=float,
        default=0.25,
        help='Energy threshold ratio'
    )
    parser.add_argument(
        '--zcr-max',
        type=float,
        default=0.2,
        help='Maximum ZCR for speech'
    )
    parser.add_argument(
        '--min-speech',
        type=int,
        default=3,
        help='Minimum speech frames'
    )
    parser.add_argument(
        '--min-silence',
        type=int,
        default=4,
        help='Minimum silence frames'
    )
    parser.add_argument(
        '--speech-start-thresh',
        type=float,
        default=0.70,
        help='Speech start score threshold'
    )
    parser.add_argument(
        '--speech-end-thresh',
        type=float,
        default=0.54,
        help='Speech end score threshold'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files per SNR level for quick experiments'
    )
    parser.add_argument(
        '--label-set',
        type=str,
        default='adaptive_v1',
        help='VAD label set to evaluate against'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(" Adaptive VAD Evaluation")
    print("="*70)
    print(f"\nSNR levels: {args.snr_levels}")
    print(f"Output dir: {output_dir}")
    print(f"Label set:  {args.label_set}")
    print(f"\nVAD Config:")
    print(f"  Energy threshold: {args.energy_thresh}")
    print(f"  ZCR max:         {args.zcr_max}")
    print(f"  Speech start:    {args.speech_start_thresh}")
    print(f"  Speech end:      {args.speech_end_thresh}")
    print(f"  Min speech:      {args.min_speech} frames")
    print(f"  Min silence:     {args.min_silence} frames")
    
    vad_config = EnergyZCRVADConfig(
        energy_thresh_ratio=args.energy_thresh,
        zcr_max_speech=args.zcr_max,
        speech_start_threshold=args.speech_start_thresh,
        speech_end_threshold=args.speech_end_thresh,
        min_speech_frames=args.min_speech,
        min_silence_frames=args.min_silence
    )
    vad_detector = EnergyZCRVAD(vad_config)
    
    all_results: Dict[str, Dict[str, float]] = {}
    
    for snr_db in args.snr_levels:
        metrics = evaluate_librispeech_test_for_snr(
            snr_db,
            vad_detector,
            verbose=True,
            max_files=args.max_files,
            label_set=args.label_set,
        )
        all_results[str(snr_db)] = metrics
    
    results_path = output_dir / "vad_energy_zcr_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n Results saved to: {results_path}")
    
    plot_vad_results(all_results, output_dir)
    
    print_results(all_results)
    
    print(f"\n All results saved to: {output_dir}")


if __name__ == "__main__":
    main()




