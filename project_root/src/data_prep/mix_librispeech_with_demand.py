from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import soundfile as sf

from src import config
from src.utils.audio_io import ensure_mono, resample_audio

RNG_SEED = 0
_REPORTED_RESAMPLES: set[Path] = set()

def _list_clean_flacs(split: str) -> list[Path]:
    """
    Return all LibriSpeech clean .flac files for a given split,
    searching recursively under data/librispeech_demand/clean/<split>/.
    Exemplu din structura ta: clean/test/test-clean/61/70968/61-70968-0000.flac
    """
    clean_root = config.LIBRISPEECH_ROOT / "clean" / split
    files = sorted(clean_root.rglob("*.flac"))
    if not files:
        raise RuntimeError(f"no clean .flac files under {clean_root}")
    return files

def _list_noise_wavs(split: str) -> list[Path]:
    """
    Return all DEMAND noise wav files for a given split,
    searching recursively under data/librispeech_demand/noise/<split>/.
    """
    noise_root = config.LIBRISPEECH_ROOT / "noise" / split
    files = sorted(noise_root.rglob("*.wav"))
    if not files:
        raise RuntimeError(f"no noise .wav files under {noise_root}")
    return files

def _ensure_mono(sig: np.ndarray) -> np.ndarray:
    return ensure_mono(sig)

def _resample_if_needed(sig: np.ndarray, sample_rate: int, target_sample_rate: int, path: Path) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return sig.astype(np.float32, copy=False)
    if path not in _REPORTED_RESAMPLES:
        print(
            f"[mix_librispeech_with_demand] Resampling {path.name}: "
            f"{sample_rate} Hz -> {target_sample_rate} Hz"
        )
        _REPORTED_RESAMPLES.add(path)
    return resample_audio(sig, sample_rate, target_sample_rate)

def _match_length(noise: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    """Repeat sau crop astfel încât len(noise) == target_len."""
    if len(noise) == target_len:
        return noise
    if len(noise) > target_len:
        start = int(rng.integers(0, len(noise) - target_len + 1))
        return noise[start : start + target_len]
    reps = int(np.ceil(target_len / len(noise)))
    tiled = np.tile(noise, reps)
    return tiled[:target_len]

def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Adaugă zgomot la SNR dorit (dB)."""
    eps = 1e-12
    clean = clean.astype(np.float32)
    noise = noise.astype(np.float32)
    p_clean = np.mean(clean**2) + eps
    p_noise = np.mean(noise**2) + eps

    target_ratio = 10.0 ** (-snr_db / 10.0)
    scale = np.sqrt(target_ratio * p_clean / p_noise)
    noisy = clean + scale * noise
    peak = np.max(np.abs(noisy))
    if peak > 1.0:
        noisy = noisy / peak
    return noisy

def generate_mixtures_for_split(split: str,
                                snr_list: Sequence[float] = (0.0, 5.0, 10.0)) -> None:
    rng = np.random.default_rng(RNG_SEED)
    clean_files = _list_clean_flacs(split)
    noise_files = _list_noise_wavs(split)
    out_dir = config.LIBRISPEECH_ROOT / "noisy" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mix_librispeech_with_demand] Split={split}, "
          f"{len(clean_files)} clean utts, {len(noise_files)} noise files.")
    sr_target = config.SAMPLE_RATE

    for clean_path in clean_files:
        clean, sr_c = sf.read(clean_path)
        clean = _ensure_mono(clean)
        clean = _resample_if_needed(clean, sr_c, sr_target, clean_path)

        for snr_db in snr_list:
            noise_path = noise_files[int(rng.integers(0, len(noise_files)))]
            noise, sr_n = sf.read(noise_path)
            noise = _ensure_mono(noise)
            noise = _resample_if_needed(noise, sr_n, sr_target, noise_path)

            noise_matched = _match_length(noise, len(clean), rng)
            noisy = _mix_at_snr(clean, noise_matched, snr_db)

            utt_id = clean_path.stem
            out_name = f"{utt_id}_snr{int(snr_db)}dB.wav"
            out_path = out_dir / out_name
            sf.write(out_path, noisy, sr_target)

    print(f"[mix_librispeech_with_demand] Done for split={split}, output in {out_dir}")

def main() -> None:
    for split in ("train", "val", "test"):
        generate_mixtures_for_split(split)

if __name__ == "__main__":
    main()
