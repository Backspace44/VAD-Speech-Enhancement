"""Generate LibriSpeech VAD label files."""

from pathlib import Path
from typing import Dict
import numpy as np
import soundfile as sf

from src import config
from src.dsp.vad_energy_zcr import EnergyZCRVAD, EnergyZCRVADConfig
from src.utils.audio_io import ensure_mono, resample_audio


_REPORTED_RESAMPLES: set[Path] = set()


def _build_clean_index(split: str) -> Dict[str, Path]:
    """Map utterance IDs to clean files."""
    clean_root = config.LIBRISPEECH_ROOT / "clean" / split
    index: Dict[str, Path] = {}
    for p in clean_root.rglob("*.flac"):
        index[p.stem] = p
    if not index:
        raise RuntimeError(f"No clean .flac files found under {clean_root}")
    return index


def _resample_if_needed(
    sig: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    path: Path,
    verbose: bool,
) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return ensure_mono(sig)
    if verbose and path not in _REPORTED_RESAMPLES:
        print(
            f"Resampling clean speech for VAD: {path.name} "
            f"({sample_rate} Hz -> {target_sample_rate} Hz)"
        )
        _REPORTED_RESAMPLES.add(path)
    return resample_audio(sig, sample_rate, target_sample_rate)


def generate_vad_labels_for_split(
    split: str,
    vad_model: EnergyZCRVAD | None = None,
    verbose: bool = True,
    label_set: str = "adaptive_v1",
    label_mode: str = "binary",
    skip_existing: bool = True,
    max_files: int | None = None,
) -> None:
    """Generate VAD labels for one LibriSpeech split."""
    if vad_model is None:
        vad_model = EnergyZCRVAD()
    
    out_dir = config.LIBRISPEECH_VAD_ROOT / label_set / split
    out_dir.mkdir(parents=True, exist_ok=True)
    
    clean_index = _build_clean_index(split)
    clean_files = sorted(clean_index.values())
    if max_files is not None:
        clean_files = clean_files[:max_files]
    
    if not clean_files:
        raise RuntimeError(f"No clean audio files for split '{split}'")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f" Generating VAD Labels (Adaptive Detector)")
        print(f"{'='*70}")
        print(f"Split:        {split}")
        print(f"Clean files:  {len(clean_files)}")
        print(f"Output dir:   {out_dir}")
        print(f"Label mode:   {label_mode}")
        print(f"VAD config:   energy_thresh={vad_model.cfg.energy_thresh_ratio:.2f}, "
              f"zcr_max={vad_model.cfg.zcr_max_speech:.2f}")
        if skip_existing:
            print("Resume mode:  skip existing labels")
    
    sr_target = config.SAMPLE_RATE
    processed = 0
    
    for i, clean_path in enumerate(clean_files, 1):
        stem = clean_path.stem
        out_path = out_dir / f"{stem}.npy"
        if skip_existing and out_path.exists():
            continue

        clean, sr_c = sf.read(clean_path)
        clean = _resample_if_needed(clean, sr_c, sr_target, clean_path, verbose)
        
        if label_mode == "soft":
            vad = vad_model.predict_soft(clean).astype(np.float32)
        else:
            vad = vad_model.predict(clean).astype(np.uint8)

        np.save(out_path, vad)
        processed += 1
        
        if verbose and i % 100 == 0:
            print(f"  Progress: {i}/{len(clean_files)} files...")
    
    if verbose:
        print(f"\nGenerated {processed} VAD label files")
        print(f"Saved to: {out_dir}")
        print(f"{'='*70}\n")


def generate_vad_labels_all_splits(
    splits: tuple[str, ...] = ("train", "val", "test"),
    vad_config: EnergyZCRVADConfig | None = None,
    verbose: bool = True,
    label_set: str = "adaptive_v1",
    label_mode: str = "binary",
    skip_existing: bool = True,
    max_files: int | None = None,
) -> None:
    """Generate VAD labels for selected splits."""
    vad_model = EnergyZCRVAD(vad_config)
    
    if verbose:
        print("\n" + "="*70)
        print(" VAD Label Generation for LibriSpeech")
        print("="*70)
        print(f"Splits to process: {', '.join(splits)}")
        print(f"VAD method: Adaptive Energy+ZCR+Spectral")
        print("="*70)
    
    for split in splits:
        generate_vad_labels_for_split(
            split,
            vad_model,
            verbose,
            label_set=label_set,
            label_mode=label_mode,
            skip_existing=skip_existing,
            max_files=max_files,
        )
    
    if verbose:
        print("\nAll splits completed!")


def main() -> None:
    """Generate VAD labels."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate VAD labels for LibriSpeech dataset using the adaptive VAD detector"
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'all'],
        default='all',
        help='Dataset split to process (default: all)'
    )
    parser.add_argument(
        '--energy-thresh',
        type=float,
        default=0.25,
        help='Energy threshold ratio 0-1 (default: 0.25)'
    )
    parser.add_argument(
        '--zcr-max',
        type=float,
        default=0.2,
        help='Maximum ZCR for speech 0-1 (default: 0.2)'
    )
    parser.add_argument(
        '--min-speech',
        type=int,
        default=3,
        help='Minimum speech frames to keep (default: 3)'
    )
    parser.add_argument(
        '--min-silence',
        type=int,
        default=4,
        help='Minimum silence frames to keep (default: 4)'
    )
    parser.add_argument(
        '--speech-start-thresh',
        type=float,
        default=0.70,
        help='Speech start score threshold (default: 0.70)'
    )
    parser.add_argument(
        '--speech-end-thresh',
        type=float,
        default=0.54,
        help='Speech end score threshold (default: 0.54)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    parser.add_argument(
        '--label-set',
        type=str,
        default='adaptive_v1',
        help='Name of the VAD label set/version to generate'
    )
    parser.add_argument(
        '--label-mode',
        type=str,
        choices=['binary', 'soft'],
        default='binary',
        help='Whether to save hard binary labels or soft speech confidence scores'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Regenerate labels even if output files already exist'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Limit processed clean files for quick or resumed runs'
    )
    
    args = parser.parse_args()
    
    vad_config = EnergyZCRVADConfig(
        energy_thresh_ratio=args.energy_thresh,
        zcr_max_speech=args.zcr_max,
        speech_start_threshold=args.speech_start_thresh,
        speech_end_threshold=args.speech_end_thresh,
        min_speech_frames=args.min_speech,
        min_silence_frames=args.min_silence
    )
    
    verbose = not args.quiet
    
    if args.split == 'all':
        generate_vad_labels_all_splits(
            vad_config=vad_config,
            verbose=verbose,
            label_set=args.label_set,
            label_mode=args.label_mode,
            skip_existing=not args.no_skip_existing,
            max_files=args.max_files,
        )
    else:
        vad_model = EnergyZCRVAD(vad_config)
        generate_vad_labels_for_split(
            args.split,
            vad_model,
            verbose,
            label_set=args.label_set,
            label_mode=args.label_mode,
            skip_existing=not args.no_skip_existing,
            max_files=args.max_files,
        )


if __name__ == "__main__":
    main()
