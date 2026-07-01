"""Extract noise-only waveforms from aligned clean/noisy pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from src import config
from src.utils.audio_io import align_waveforms, find_matched_file_pairs, load_audio_mono


def extract_noise_split(
    clean_dir: Path,
    noisy_dir: Path,
    output_dir: Path,
    sample_rate: int = config.SAMPLE_RATE,
    overwrite: bool = False,
) -> int:
    """Save residual noise = noisy - clean for each matched pair."""
    pairs = find_matched_file_pairs(clean_dir, noisy_dir)
    if not pairs:
        raise RuntimeError(f"No matched clean/noisy pairs found under {clean_dir} and {noisy_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for clean_path, noisy_path in tqdm(pairs, desc=f"Extracting {output_dir.name} noise"):
        output_path = output_dir / f"{noisy_path.stem}.wav"
        if output_path.exists() and not overwrite:
            continue

        clean, _ = load_audio_mono(clean_path, target_sample_rate=sample_rate)
        noisy, _ = load_audio_mono(noisy_path, target_sample_rate=sample_rate)
        clean, noisy = align_waveforms(clean, noisy)
        noise = noisy - clean
        sf.write(output_path, noise, sample_rate)
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract noise waveforms from clean/noisy dataset pairs")
    parser.add_argument("--dataset", choices=["voicebank"], default="voicebank")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dataset != "voicebank":
        raise ValueError("Only VoiceBank extraction is currently supported")

    splits = ("train", "test") if args.split == "all" else (args.split,)
    total = 0
    for split in splits:
        written = extract_noise_split(
            clean_dir=config.VOICEBANK_ROOT / "clean" / split,
            noisy_dir=config.VOICEBANK_ROOT / "noisy" / split,
            output_dir=config.VOICEBANK_ROOT / "noise" / split,
            overwrite=args.overwrite,
        )
        total += written
        print(f"{split}: wrote {written} noise files")

    print(f"Done. Wrote {total} noise files.")


if __name__ == "__main__":
    main()
