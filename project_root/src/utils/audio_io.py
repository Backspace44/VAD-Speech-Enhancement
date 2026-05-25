"""Shared audio I/O helpers for dataset matching and waveform loading."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf


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


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file and convert it to mono."""
    audio, sample_rate = sf.read(path)
    if np.ndim(audio) > 1:
        audio = np.mean(audio, axis=1)
    return audio, sample_rate


def align_waveforms(*waveforms: np.ndarray) -> tuple[np.ndarray, ...]:
    """Trim all waveforms to the same minimum length."""
    min_len = min(len(waveform) for waveform in waveforms)
    return tuple(waveform[:min_len] for waveform in waveforms)
