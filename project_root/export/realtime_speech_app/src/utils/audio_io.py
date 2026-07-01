"""Shared audio I/O helpers for dataset matching and waveform loading."""

from __future__ import annotations

from math import gcd
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def normalize_noisy_stem(stem: str) -> str:
    """Normalize a noisy filename stem to its clean reference stem."""
    stem = re.sub(r"_snr-?\d+dB.*", "", stem)
    stem = re.sub(r"_[A-Z]+_16k", "", stem)
    stem = re.sub(r"_[A-Z]+$", "", stem)
    return stem


def find_matched_file_pairs(
    clean_dir: Path,
    noisy_dir: Path,
    clean_patterns: tuple[str, ...] = ("*.wav", "*.flac"),
    noisy_patterns: tuple[str, ...] = ("*.wav",),
) -> list[tuple[Path, Path]]:
    """Match clean/noisy files by normalized stem."""
    clean_files: list[Path] = []
    for pattern in clean_patterns:
        clean_files.extend(clean_dir.rglob(pattern))

    noisy_files: list[Path] = []
    for pattern in noisy_patterns:
        noisy_files.extend(noisy_dir.rglob(pattern))

    clean_files = sorted(clean_files)
    noisy_files = sorted(noisy_files)

    clean_stems = {path.stem: path for path in clean_files}
    noisy_stems = {normalize_noisy_stem(path.stem): path for path in noisy_files}

    common_stems = sorted(set(clean_stems.keys()) & set(noisy_stems.keys()))
    return [(clean_stems[stem], noisy_stems[stem]) for stem in common_stems]


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert an audio array to mono float32."""
    if np.ndim(audio) > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32, copy=False)


def resample_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """Resample audio to the target sample rate using polyphase filtering."""
    audio = ensure_mono(audio)
    if sample_rate == target_sample_rate:
        return audio

    divisor = gcd(int(sample_rate), int(target_sample_rate))
    up = int(target_sample_rate) // divisor
    down = int(sample_rate) // divisor
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def load_audio_mono(path: Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Load an audio file, convert it to mono, and optionally resample it."""
    audio, sample_rate = sf.read(path)
    audio = ensure_mono(audio)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        audio = resample_audio(audio, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return audio, sample_rate


def align_waveforms(*waveforms: np.ndarray) -> tuple[np.ndarray, ...]:
    """Trim all waveforms to the same minimum length."""
    min_len = min(len(waveform) for waveform in waveforms)
    return tuple(waveform[:min_len] for waveform in waveforms)
