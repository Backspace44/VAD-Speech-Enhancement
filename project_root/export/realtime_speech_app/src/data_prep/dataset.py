"""
Optimized Dataset class for speech enhancement with efficient data loading.
Supports parallel loading with num_workers and various preprocessing options.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import random
import logging

from src.dsp.stft_utils import (
    complex_to_channels,
    compute_complex_ratio_mask,
    compute_stft,
)
from src.dsp.preprocessing import AudioPreprocessor
from src.data_prep.augmentation import create_audio_augmenter
from src.utils.audio_io import find_matched_file_pairs

logger = logging.getLogger(__name__)


class SpeechEnhancementDataset(Dataset):
    """
    Optimized dataset for speech enhancement training.
    
    Features:
    - Efficient parallel loading with num_workers
    - On-the-fly STFT computation
    - Optional on-the-fly clean+noise mixing
    - Optional data augmentation
    - Flexible preprocessing
    - Memory-efficient file loading
    - Caching support for small datasets
    
    Args:
        clean_dir: Directory containing clean audio files
        noisy_dir: Directory containing noisy audio files (legacy/evaluation mode)
        noise_dir: Directory containing noise-only audio files (on-the-fly mixing)
        sample_rate: Target sample rate (default: 16000)
        n_fft: FFT size for STFT (default: 512 = config.N_FFT)
        hop_length: Hop length for STFT (default: 160 = config.HOP_LEN)
        win_length: Window length for STFT (default: 400 = config.FRAME_LEN)
        max_length: Maximum audio length in samples (None = no limit)
        augmentation: Data augmentation function
        preprocessing: Audio preprocessing function
        cache_in_memory: Cache all data in memory (for small datasets)
        return_audio: Return audio waveforms (for evaluation)
    """
    
    def __init__(
        self,
        clean_dir: Path,
        noisy_dir: Optional[Path] = None,
        noise_dir: Optional[Path] = None,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        max_length: Optional[int] = None,
        augmentation: Optional[Callable] = None,
        preprocessing: Optional[AudioPreprocessor] = None,
        cache_in_memory: bool = False,
        return_audio: bool = False,
        max_samples: Optional[int] = None,
        vad_dir: Optional[Path] = None,
        use_vad_labels: bool = False,
        vad_soft_mask_floor: float = 0.0,
        complex_mask_clip: float = 5.0,
        snr_range: Tuple[float, float] = (0.0, 20.0),
        deterministic_mixing: bool = False,
        random_seed: int = 0,
    ):
        super().__init__()
        
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir) if noisy_dir else None
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_length = max_length
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.cache_in_memory = cache_in_memory
        self.return_audio = return_audio
        self.vad_dir = Path(vad_dir) if vad_dir else None
        self.use_vad_labels = use_vad_labels
        self.vad_soft_mask_floor = float(max(0.0, min(1.0, vad_soft_mask_floor)))
        self.complex_mask_clip = complex_mask_clip
        self.snr_range = snr_range
        self.deterministic_mixing = deterministic_mixing
        self.random_seed = random_seed

        clean_files_all = sorted(list(self.clean_dir.rglob('*.wav')) + list(self.clean_dir.rglob('*.flac')))
        if self.noise_dir is not None:
            self.noise_files = sorted(list(self.noise_dir.rglob('*.wav')) + list(self.noise_dir.rglob('*.flac')))
            if not clean_files_all:
                raise ValueError(f"No clean audio files found in {self.clean_dir}")
            if not self.noise_files:
                raise ValueError(f"No noise audio files found in {self.noise_dir}")
            self.clean_files = clean_files_all
            self.noisy_files: List[Path] = []
            logger.info(
                "Using on-the-fly mixing: %d clean files x %d noise files, SNR %.1f..%.1f dB",
                len(self.clean_files),
                len(self.noise_files),
                self.snr_range[0],
                self.snr_range[1],
            )
        else:
            if self.noisy_dir is None:
                raise ValueError("Either noise_dir for on-the-fly mixing or noisy_dir for legacy pairs must be provided")
            matched_pairs = find_matched_file_pairs(
                self.clean_dir,
                self.noisy_dir,
                clean_patterns=("*.wav", "*.flac"),
                noisy_patterns=("*.wav",),
            )

            noisy_files_all = sorted(list(self.noisy_dir.rglob('*.wav')))

            if len(matched_pairs) == 0:
                raise ValueError("No matching clean-noisy file pairs found!")

            self.clean_files = [clean_path for clean_path, _ in matched_pairs]
            self.noisy_files = [noisy_path for _, noisy_path in matched_pairs]
            self.noise_files = []
            num_matched_pairs = len(matched_pairs)

            skipped = len(clean_files_all) + len(noisy_files_all) - 2 * num_matched_pairs
            if skipped > 0:
                logger.warning(
                    f"Skipped {skipped} files without pairs. "
                    f"Using {len(self.clean_files)} matched pairs. "
                    f"(Clean: {len(clean_files_all)}, Noisy: {len(noisy_files_all)})"
                )
        
        # Limit number of samples if max_samples is set
        if max_samples is not None and max_samples < len(self.clean_files):
            self.clean_files = self.clean_files[:max_samples]
            if self.noisy_files:
                self.noisy_files = self.noisy_files[:max_samples]
            logger.info(f"Limited dataset to {max_samples} samples")
        

        self.cache = {}
        if cache_in_memory:
            print(f"Caching {len(self)} samples in memory...")
            for idx in range(len(self)):
                self.cache[idx] = self._load_and_process(idx)
            print("Caching complete!")
    
    def __len__(self) -> int:
        return len(self.clean_files)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - 'noisy_mag': Noisy magnitude spectrogram [1, freq, time]
                - 'noisy_complex': Noisy complex STFT as real/imag channels [2, freq, time]
                - 'clean_mag': Clean magnitude spectrogram [1, freq, time]
                - 'complex_mask': Complex ratio mask target [2, freq, time]
                - 'noisy_phase': Noisy phase spectrogram [freq, time]
                - 'noisy_audio': Noisy waveform (if return_audio=True)
                - 'clean_audio': Clean waveform (if return_audio=True)
                - 'filename': File name
        """

        if self.cache_in_memory and idx in self.cache:
            return self.cache[idx]
        
        return self._load_and_process(idx)
    
    def _load_and_process(self, idx: int) -> dict:
        """Load and process a single sample."""
        

        clean_audio, sr_clean = self._load_audio_tensor(self.clean_files[idx])
        crop_start = 0

        if self.noise_dir is not None:
            rng = self._rng_for_idx(idx)
            if self.max_length and len(clean_audio) > self.max_length:
                crop_start = rng.randint(0, len(clean_audio) - self.max_length)
                clean_audio = clean_audio[crop_start:crop_start + self.max_length]
            noise_path = self.noise_files[rng.randrange(len(self.noise_files))]
            noise_audio, _ = self._load_audio_tensor(noise_path)
            noise_audio = self._match_noise_length(noise_audio, len(clean_audio), rng)
            snr_db = rng.uniform(self.snr_range[0], self.snr_range[1])
            noisy_audio, scaled_noise = self._mix_at_snr(clean_audio, noise_audio, snr_db)
        else:
            noisy_audio, _ = self._load_audio_tensor(self.noisy_files[idx])

            min_len = min(len(clean_audio), len(noisy_audio))
            clean_audio = clean_audio[:min_len]
            noisy_audio = noisy_audio[:min_len]

            if self.max_length and len(clean_audio) > self.max_length:
                crop_start = random.randint(0, len(clean_audio) - self.max_length)
                clean_audio = clean_audio[crop_start:crop_start + self.max_length]
                noisy_audio = noisy_audio[crop_start:crop_start + self.max_length]
            scaled_noise = noisy_audio - clean_audio


        if self.preprocessing:
            clean_audio, _ = self.preprocessing(clean_audio.numpy())
            noisy_audio, _ = self.preprocessing(noisy_audio.numpy())
            clean_audio = torch.from_numpy(clean_audio).float()
            noisy_audio = torch.from_numpy(noisy_audio).float()
            scaled_noise = noisy_audio - clean_audio
        

        if self.augmentation:
            noisy_numpy = noisy_audio.numpy() if isinstance(noisy_audio, torch.Tensor) else noisy_audio
            noisy_numpy = self.augmentation(noisy_numpy, is_training=True)
            noisy_audio = torch.from_numpy(noisy_numpy).float() if isinstance(noisy_numpy, np.ndarray) else noisy_numpy
            scaled_noise = noisy_audio - clean_audio
        
        if not isinstance(clean_audio, torch.Tensor):
            clean_audio = torch.from_numpy(clean_audio).float()
        if not isinstance(noisy_audio, torch.Tensor):
            noisy_audio = torch.from_numpy(noisy_audio).float()
        

        # Compute complex spectrograms for clean and noisy.
        clean_complex = compute_stft(
            clean_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
        noisy_complex = compute_stft(
            noisy_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
        clean_mag = clean_complex.abs()
        noisy_mag = noisy_complex.abs()
        noisy_phase = noisy_complex.angle()
        
        noise_complex = compute_stft(
            scaled_noise,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
        noise_mag = noise_complex.abs()
        
        # Add channel dimension
        clean_mag = clean_mag.unsqueeze(0)
        noisy_mag = noisy_mag.unsqueeze(0)
        noise_mag = noise_mag.unsqueeze(0)
        noisy_complex_channels = complex_to_channels(noisy_complex)
        
        # Compute Ideal Ratio Mask (IRM): clean / (clean + noise)
        eps = 1e-8
        ideal_mask = clean_mag / (clean_mag + noise_mag + eps)
        ideal_mask = torch.clamp(ideal_mask, 0.0, 1.0)
        complex_mask = compute_complex_ratio_mask(
            clean_complex,
            noisy_complex,
            eps=eps,
            clip_value=self.complex_mask_clip,
        )
        
        # Apply VAD labels if available (force mask to 0 in silence regions)
        vad_gate = None
        if self.use_vad_labels and self.vad_dir:
            vad_labels = self._load_vad_labels(idx, frame_start=crop_start // self.hop_length)
            if vad_labels is not None:
                # VAD labels are frame-level, align with spectrogram time frames
                vad_tensor = torch.from_numpy(vad_labels).float()
                # Ensure VAD matches spectrogram time dimension
                if len(vad_tensor) != ideal_mask.shape[-1]:
                    # Interpolate or pad/trim to match
                    if len(vad_tensor) > ideal_mask.shape[-1]:
                        vad_tensor = vad_tensor[:ideal_mask.shape[-1]]
                    else:
                        # Pad with last value
                        padding = ideal_mask.shape[-1] - len(vad_tensor)
                        vad_tensor = torch.cat([vad_tensor, vad_tensor[-1].repeat(padding)])
                
                vad_tensor = torch.clamp(vad_tensor, 0.0, 1.0)

                # Soft label sets are treated as frame confidence and keep a small
                # floor so uncertain speech is attenuated rather than erased.
                if torch.is_floating_point(vad_tensor) and not torch.all((vad_tensor == 0) | (vad_tensor == 1)):
                    vad_gate = self.vad_soft_mask_floor + (1.0 - self.vad_soft_mask_floor) * vad_tensor
                else:
                    vad_gate = vad_tensor

                # Shape: [1, freq, time] * [time] -> broadcast over freq dimension
                ideal_mask = ideal_mask * vad_gate.view(1, 1, -1)
                complex_mask = complex_mask * vad_gate.view(1, 1, -1)
        

        output = {
            'noisy_mag': noisy_mag,
            'noisy_complex': noisy_complex_channels,
            'clean_mag': clean_mag,
            'ideal_mask': ideal_mask,
            'complex_mask': complex_mask,
            'noisy_phase': noisy_phase,
            'filename': self.clean_files[idx].name
        }
        

        if self.return_audio:
            output['noisy_audio'] = noisy_audio
            output['clean_audio'] = clean_audio
        
        return output

    def _rng_for_idx(self, idx: int) -> random.Random:
        if self.deterministic_mixing:
            return random.Random(self.random_seed + idx)
        return random

    def _load_audio_tensor(self, path: Path) -> Tuple[torch.Tensor, int]:
        audio, sample_rate = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            audio = resampler(audio)
            sample_rate = self.sample_rate
        return audio.squeeze(0).float(), sample_rate

    def _match_noise_length(self, noise: torch.Tensor, target_len: int, rng: random.Random) -> torch.Tensor:
        if len(noise) == target_len:
            return noise
        if len(noise) > target_len:
            start = rng.randint(0, len(noise) - target_len)
            return noise[start:start + target_len]
        repeats = int(np.ceil(target_len / max(1, len(noise))))
        return noise.repeat(repeats)[:target_len]

    def _mix_at_snr(self, clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-12
        clean_power = torch.mean(clean ** 2) + eps
        noise_power = torch.mean(noise ** 2) + eps
        target_ratio = 10.0 ** (-snr_db / 10.0)
        scale = torch.sqrt(torch.tensor(target_ratio, dtype=clean.dtype) * clean_power / noise_power)
        scaled_noise = noise * scale
        noisy = clean + scaled_noise
        return noisy.float(), scaled_noise.float()
    
    def _load_vad_labels(self, idx: int, frame_start: int = 0) -> Optional[np.ndarray]:
        """Load VAD labels for a given sample index."""
        if not self.vad_dir:
            return None
        
        if self.noisy_files:
            vad_path = self.vad_dir / f"{self.noisy_files[idx].stem}.npy"
        else:
            clean_stem = self.clean_files[idx].stem
            vad_path = self.vad_dir / f"{clean_stem}.npy"
            if not vad_path.exists():
                matches = sorted(self.vad_dir.glob(f"{clean_stem}*.npy"))
                vad_path = matches[0] if matches else vad_path
        
        if not vad_path.exists():
            logger.warning(f"VAD labels not found: {vad_path}")
            return None
        
        try:
            vad_labels = np.load(vad_path)
            if frame_start > 0:
                vad_labels = vad_labels[frame_start:]
            return vad_labels
        except Exception as e:
            logger.error(f"Error loading VAD labels from {vad_path}: {e}")
            return None


def collate_fn_pad(batch):
    """
    Custom collate function to handle variable-length spectrograms.
    Pads all spectrograms to max length in batch.
    Optionally includes audio if present in samples.
    """
    max_time = max(item['noisy_mag'].shape[-1] for item in batch)
    
    batch_noisy_mag = []
    batch_noisy_complex = []
    batch_clean_mag = []
    batch_ideal_mask = []
    batch_complex_mask = []
    batch_noisy_phase = []
    filenames = []
    
    # Check if audio is present in first sample
    has_audio = 'noisy_audio' in batch[0]
    if has_audio:
        batch_noisy_audio = []
        batch_clean_audio = []
    
    for item in batch:
        noisy_mag = item['noisy_mag']
        noisy_complex = item['noisy_complex']
        clean_mag = item['clean_mag']
        ideal_mask = item['ideal_mask']
        complex_mask = item['complex_mask']
        noisy_phase = item['noisy_phase']
        
        time_len = noisy_mag.shape[-1]
        if time_len < max_time:
            pad_len = max_time - time_len
            noisy_mag = torch.nn.functional.pad(noisy_mag, (0, pad_len))
            noisy_complex = torch.nn.functional.pad(noisy_complex, (0, pad_len))
            clean_mag = torch.nn.functional.pad(clean_mag, (0, pad_len))
            ideal_mask = torch.nn.functional.pad(ideal_mask, (0, pad_len))
            complex_mask = torch.nn.functional.pad(complex_mask, (0, pad_len))
            noisy_phase = torch.nn.functional.pad(noisy_phase, (0, pad_len))
        
        batch_noisy_mag.append(noisy_mag)
        batch_noisy_complex.append(noisy_complex)
        batch_clean_mag.append(clean_mag)
        batch_ideal_mask.append(ideal_mask)
        batch_complex_mask.append(complex_mask)
        batch_noisy_phase.append(noisy_phase)
        filenames.append(item['filename'])
        
        # Collect audio if present
        if has_audio:
            batch_noisy_audio.append(item['noisy_audio'])
            batch_clean_audio.append(item['clean_audio'])
    
    result = {
        'noisy_mag': torch.stack(batch_noisy_mag),
        'noisy_complex': torch.stack(batch_noisy_complex),
        'clean_mag': torch.stack(batch_clean_mag),
        'ideal_mask': torch.stack(batch_ideal_mask),
        'complex_mask': torch.stack(batch_complex_mask),
        'noisy_phase': torch.stack(batch_noisy_phase),
        'filename': filenames
    }
    
    # Add audio tensors if present (already same length from dataset)
    if has_audio:
        result['noisy_audio'] = torch.stack(batch_noisy_audio)
        result['clean_audio'] = torch.stack(batch_clean_audio)
    
    return result


def create_dataloaders(
    train_clean_dir: Path,
    train_noise_dir: Path,
    val_clean_dir: Path,
    val_noise_dir: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    max_length: Optional[int] = None,
    use_augmentation: bool = True,
    use_preprocessing: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    train_vad_dir: Optional[Path] = None,
    val_vad_dir: Optional[Path] = None,
    use_vad_labels: bool = False,
    vad_soft_mask_floor: float = 0.0,
    complex_mask_clip: float = 5.0,
    snr_range: Tuple[float, float] = (0.0, 20.0),
    random_seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized train and validation dataloaders.
    
    Args:
        train_clean_dir: Training clean audio directory
        train_noise_dir: Training noise audio directory for on-the-fly mixing
        val_clean_dir: Validation clean audio directory
        val_noise_dir: Validation noise audio directory for deterministic on-the-fly mixing
        batch_size: Batch size
        num_workers: Number of data loading workers
        sample_rate: Target sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        max_length: Maximum audio length
        use_augmentation: Enable data augmentation
        use_preprocessing: Enable preprocessing
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        
    Returns:
        train_loader, val_loader
    """
    

    augmentation = None
    if use_augmentation:
        try:
            augmentation = create_audio_augmenter(sample_rate=sample_rate)
        except Exception as e:
            logger.warning(f"Augmentation init failed: {e}. Disabling augmentation.")
            augmentation = None
    
    # Initialize preprocessing if enabled
    preprocessing = None
    if use_preprocessing:
        try:
            from src.dsp.preprocessing import AudioPreprocessor
            preprocessing = AudioPreprocessor(sample_rate=sample_rate)
            logger.info("Preprocessing initialized")
        except Exception as e:
            logger.warning(f"Preprocessing init failed: {e}. Disabling preprocessing.")
            preprocessing = None
    

    train_dataset = SpeechEnhancementDataset(
        clean_dir=train_clean_dir,
        noise_dir=train_noise_dir,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        max_length=max_length,
        augmentation=augmentation,
        preprocessing=preprocessing,
        cache_in_memory=False,
        return_audio=False,
        max_samples=max_train_samples,
        vad_dir=train_vad_dir,
        use_vad_labels=use_vad_labels,
        vad_soft_mask_floor=vad_soft_mask_floor,
        complex_mask_clip=complex_mask_clip,
        snr_range=snr_range,
        deterministic_mixing=False,
        random_seed=random_seed,
    )
    
    val_dataset = SpeechEnhancementDataset(
        clean_dir=val_clean_dir,
        noise_dir=val_noise_dir,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        max_length=max_length,
        augmentation=None,
        preprocessing=preprocessing,
        cache_in_memory=False,
        return_audio=False,
        max_samples=max_val_samples,
        vad_dir=val_vad_dir,
        use_vad_labels=use_vad_labels,
        vad_soft_mask_floor=vad_soft_mask_floor,
        complex_mask_clip=complex_mask_clip,
        snr_range=snr_range,
        deterministic_mixing=True,
        random_seed=random_seed + 10_000,
    )
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
        collate_fn=collate_fn_pad
    )
    
    # Calculate validation workers separately to avoid errors when num_workers=1
    val_workers = num_workers // 2 if num_workers > 0 else 0
    
    # Build val_loader kwargs conditionally based on val_workers
    val_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': val_workers,
        'pin_memory': pin_memory,
        'drop_last': False,
        'collate_fn': collate_fn_pad
    }
    
    # Only add prefetch_factor and persistent_workers if val_workers > 0
    if val_workers > 0:
        val_loader_kwargs['prefetch_factor'] = prefetch_factor
        val_loader_kwargs['persistent_workers'] = True
    
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    return train_loader, val_loader
