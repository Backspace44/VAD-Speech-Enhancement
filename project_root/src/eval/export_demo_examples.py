"""Export thesis/demo audio examples and spectrogram figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src import config
from src.dsp.metrics import evaluate_speech_enhancement
from src.models import load_masknet_checkpoint
from src.utils.audio_io import align_waveforms, find_matched_file_pairs, load_audio_mono
from src.utils.enhancement_pipeline import run_enhancement_methods


METHOD_LABELS = {
    "Clean": "Clean",
    "Noisy": "Noisy",
    "Spectral_Subtraction": "Spectral Subtraction",
    "Wiener_Filter": "Wiener Filter",
    "MaskNet": "MaskNet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final audio/spectrogram examples")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/voicebank_demand/clean/test"))
    parser.add_argument("--noisy-dir", type=Path, default=Path("data/voicebank_demand/noisy/test"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/train_balanced_res_20260614_091134/masknet_best.pth"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_demo_examples"))
    parser.add_argument("--num-examples", type=int, default=3)
    return parser.parse_args()


def estimate_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    clean, noisy = align_waveforms(clean, noisy)
    noise = noisy - clean
    clean_power = float(np.mean(clean**2)) + 1e-12
    noise_power = float(np.mean(noise**2)) + 1e-12
    return 10.0 * np.log10(clean_power / noise_power)


def select_examples(pairs: list[tuple[Path, Path]], num_examples: int) -> list[tuple[str, Path, Path, float]]:
    scored: list[tuple[float, Path, Path]] = []
    for clean_path, noisy_path in pairs:
        clean, sr_clean = load_audio_mono(clean_path, target_sample_rate=config.SAMPLE_RATE)
        noisy, _ = load_audio_mono(noisy_path, target_sample_rate=config.SAMPLE_RATE)
        scored.append((estimate_snr(clean, noisy), clean_path, noisy_path))

    scored.sort(key=lambda item: item[0])
    if num_examples <= 1:
        picks = [("low_snr", scored[0])]
    elif num_examples == 2:
        picks = [("low_snr", scored[0]), ("high_snr", scored[-1])]
    else:
        mid = len(scored) // 2
        picks = [("low_snr", scored[0]), ("mid_snr", scored[mid]), ("high_snr", scored[-1])]
        extra_needed = max(0, num_examples - 3)
        if extra_needed:
            step = max(1, len(scored) // (extra_needed + 2))
            picks.extend((f"extra_{i+1}", scored[(i + 1) * step]) for i in range(extra_needed))

    return [(label, clean_path, noisy_path, snr) for label, (snr, clean_path, noisy_path) in picks]


def normalize_for_wav(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.99:
        audio = audio / peak * 0.99
    return audio.astype(np.float32, copy=False)


def save_audio_outputs(example_dir: Path, outputs: dict[str, np.ndarray], sample_rate: int) -> None:
    audio_dir = example_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in outputs.items():
        sf.write(audio_dir / f"{name}.wav", normalize_for_wav(audio), sample_rate)


def plot_example(example_dir: Path, outputs: dict[str, np.ndarray], sample_rate: int, title: str) -> None:
    names = ["Clean", "Noisy", "Spectral_Subtraction", "Wiener_Filter", "MaskNet"]
    names = [name for name in names if name in outputs]
    duration = min(3.0, min(len(outputs[name]) for name in names) / sample_rate)
    samples = int(duration * sample_rate)
    time = np.arange(samples) / sample_rate

    fig, axes = plt.subplots(len(names), 2, figsize=(12, 2.15 * len(names)), constrained_layout=True)
    if len(names) == 1:
        axes = np.array([axes])

    for row, name in enumerate(names):
        audio = outputs[name][:samples]
        label = METHOD_LABELS.get(name, name)

        axes[row, 0].plot(time, audio, linewidth=0.65)
        axes[row, 0].set_title(f"{label} - waveform")
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 0].grid(alpha=0.2)

        axes[row, 1].specgram(audio, NFFT=512, Fs=sample_rate, noverlap=384, cmap="magma")
        axes[row, 1].set_title(f"{label} - spectrogram")
        axes[row, 1].set_xlabel("Time (s)")
        axes[row, 1].set_ylabel("Frequency (Hz)")

    fig.suptitle(title, fontsize=14, weight="bold")
    fig.savefig(example_dir / "waveforms_spectrograms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_example(
    label: str,
    clean_path: Path,
    noisy_path: Path,
    estimated_snr: float,
    model,
    device: torch.device,
    output_dir: Path,
) -> dict:
    clean, sample_rate = load_audio_mono(clean_path, target_sample_rate=config.SAMPLE_RATE)
    noisy, _ = load_audio_mono(noisy_path, target_sample_rate=config.SAMPLE_RATE)
    clean, noisy = align_waveforms(clean, noisy)

    enhanced = run_enhancement_methods(noisy, model=model, device=device)
    outputs = {"Clean": clean, **enhanced}
    outputs = dict(zip(outputs.keys(), align_waveforms(*outputs.values())))

    example_dir = output_dir / label
    example_dir.mkdir(parents=True, exist_ok=True)
    save_audio_outputs(example_dir, outputs, sample_rate)
    plot_example(
        example_dir,
        outputs,
        sample_rate,
        f"{label.replace('_', ' ').title()} | {noisy_path.name} | estimated SNR={estimated_snr:.2f} dB",
    )

    metrics = {}
    for method_name, audio in outputs.items():
        if method_name == "Clean":
            continue
        metrics[method_name] = evaluate_speech_enhancement(clean, audio, sample_rate)

    row = {
        "example": label,
        "clean_file": str(clean_path),
        "noisy_file": str(noisy_path),
        "estimated_input_snr": estimated_snr,
    }
    for method_name, method_metrics in metrics.items():
        for metric_name, value in method_metrics.items():
            row[f"{method_name}_{metric_name}"] = value
    return row


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_matched_file_pairs(args.clean_dir, args.noisy_dir, clean_patterns=("*.wav",), noisy_patterns=("*.wav",))
    if not pairs:
        raise RuntimeError(f"No matched clean/noisy pairs found under {args.clean_dir} and {args.noisy_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_masknet_checkpoint(args.checkpoint, device)
    selected = select_examples(pairs, args.num_examples)

    rows = []
    for label, clean_path, noisy_path, snr in selected:
        print(f"Exporting {label}: {noisy_path.name} (estimated SNR={snr:.2f} dB)")
        rows.append(export_example(label, clean_path, noisy_path, snr, model, device, args.output_dir))

    pd.DataFrame(rows).to_csv(args.output_dir / "example_metrics.csv", index=False)
    print(f"Demo examples saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
