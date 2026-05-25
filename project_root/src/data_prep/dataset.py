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

from src.dsp.stft_utils import compute_stft
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
    - Optional data augmentation
    - Flexible preprocessing
    - Memory-efficient file loading
    - Caching support for small datasets
    
    Args:
        clean_dir: Directory containing clean audio files
        noisy_dir: Directory containing noisy audio files
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
        noisy_dir: Path,
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
    ):
        super().__init__()
        
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
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
        
        matched_pairs = find_matched_file_pairs(
            self.clean_dir,
            self.noisy_dir,
            clean_patterns=("*.wav", "*.flac"),
            noisy_patterns=("*.wav",),
        )

        clean_files_all = sorted(list(self.clean_dir.rglob('*.wav')) + list(self.clean_dir.rglob('*.flac')))
        noisy_files_all = sorted(list(self.noisy_dir.rglob('*.wav')))

        if len(matched_pairs) == 0:
            raise ValueError("No matching clean-noisy file pairs found!")

        self.clean_files = [clean_path for clean_path, _ in matched_pairs]
        self.noisy_files = [noisy_path for _, noisy_path in matched_pairs]
        num_matched_pairs = len(matched_pairs)
        
        # Limit number of samples if max_samples is set
        if max_samples is not None and max_samples < len(self.clean_files):
            self.clean_files = self.clean_files[:max_samples]
            self.noisy_files = self.noisy_files[:max_samples]
            logger.info(f"Limited dataset to {max_samples} samples")
        
        skipped = len(clean_files_all) + len(noisy_files_all) - 2 * num_matched_pairs
        if skipped > 0:
            logger.warning(
                f"Skipped {skipped} files without pairs. "
                f"Using {len(self.clean_files)} matched pairs. "
                f"(Clean: {len(clean_files_all)}, Noisy: {len(noisy_files_all)})"
            )
        

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
                - 'clean_mag': Clean magnitude spectrogram [1, freq, time]
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
        

        clean_audio, sr_clean = torchaudio.load(self.clean_files[idx])
        noisy_audio, sr_noisy = torchaudio.load(self.noisy_files[idx])
        

        if clean_audio.shape[0] > 1:
            clean_audio = clean_audio.mean(dim=0, keepdim=True)
        if noisy_audio.shape[0] > 1:
            noisy_audio = noisy_audio.mean(dim=0, keepdim=True)
        

        if sr_clean != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr_clean, self.sample_rate)
            clean_audio = resampler(clean_audio)
        if sr_noisy != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr_noisy, self.sample_rate)
            noisy_audio = resampler(noisy_audio)
        

        clean_audio = clean_audio.squeeze(0)
        noisy_audio = noisy_audio.squeeze(0)
        

        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        

        if self.max_length and len(clean_audio) > self.max_length:
            start = random.randint(0, len(clean_audio) - self.max_length)
            clean_audio = clean_audio[start:start + self.max_length]
            noisy_audio = noisy_audio[start:start + self.max_length]
        

        if self.preprocessing:
            clean_audio, _ = self.preprocessing(clean_audio.numpy())
            noisy_audio, _ = self.preprocessing(noisy_audio.numpy())
            clean_audio = torch.from_numpy(clean_audio).float()
            noisy_audio = torch.from_numpy(noisy_audio).float()
        

        if self.augmentation:
            noisy_numpy = noisy_audio.numpy() if isinstance(noisy_audio, torch.Tensor) else noisy_audio
            noisy_numpy = self.augmentation(noisy_numpy, is_training=True)
            noisy_audio = torch.from_numpy(noisy_numpy).float() if isinstance(noisy_numpy, np.ndarray) else noisy_numpy
        
        if not isinstance(clean_audio, torch.Tensor):
            clean_audio = torch.from_numpy(clean_audio).float()
        if not isinstance(noisy_audio, torch.Tensor):
            noisy_audio = torch.from_numpy(noisy_audio).float()
        

        # Compute spectrograms for clean and noisy
        clean_mag, _ = compute_stft(
            clean_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        noisy_mag, noisy_phase = compute_stft(
            noisy_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        # Compute noise magnitude for proper IRM calculation
        # Note: |noisy| != |clean| + |noise| due to phase differences
        # We approximate noise as noisy - clean in time domain, then take STFT
        noise_audio = noisy_audio - clean_audio
        noise_mag, _ = compute_stft(
            noise_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        # Add channel dimension
        clean_mag = clean_mag.unsqueeze(0)
        noisy_mag = noisy_mag.unsqueeze(0)
        noise_mag = noise_mag.unsqueeze(0)
        
        # Compute Ideal Ratio Mask (IRM): clean / (clean + noise)
        eps = 1e-8
        ideal_mask = clean_mag / (clean_mag + noise_mag + eps)
        ideal_mask = torch.clamp(ideal_mask, 0.0, 1.0)
        
        # Apply VAD labels if available (force mask to 0 in silence regions)
        if self.use_vad_labels and self.vad_dir:
            vad_labels = self._load_vad_labels(idx)
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
        

        output = {
            'noisy_mag': noisy_mag,
            'clean_mag': clean_mag,
            'ideal_mask': ideal_mask,
            'noisy_phase': noisy_phase,
            'filename': self.clean_files[idx].name
        }
        

        if self.return_audio:
            output['noisy_audio'] = noisy_audio
            output['clean_audio'] = clean_audio
        
        return output
    
    def _load_vad_labels(self, idx: int) -> Optional[np.ndarray]:
        """Load VAD labels for a given sample index."""
        if not self.vad_dir:
            return None
        
        # VAD files are named after noisy files
        noisy_stem = self.noisy_files[idx].stem
        vad_path = self.vad_dir / f"{noisy_stem}.npy"
        
        if not vad_path.exists():
            logger.warning(f"VAD labels not found: {vad_path}")
            return None
        
        try:
            vad_labels = np.load(vad_path)
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
    batch_clean_mag = []
    batch_ideal_mask = []
    batch_noisy_phase = []
    filenames = []
    
    # Check if audio is present in first sample
    has_audio = 'noisy_audio' in batch[0]
    if has_audio:
        batch_noisy_audio = []
        batch_clean_audio = []
    
    for item in batch:
        noisy_mag = item['noisy_mag']
        clean_mag = item['clean_mag']
        ideal_mask = item['ideal_mask']
        noisy_phase = item['noisy_phase']
        
        time_len = noisy_mag.shape[-1]
        if time_len < max_time:
            pad_len = max_time - time_len
            noisy_mag = torch.nn.functional.pad(noisy_mag, (0, pad_len))
            clean_mag = torch.nn.functional.pad(clean_mag, (0, pad_len))
            ideal_mask = torch.nn.functional.pad(ideal_mask, (0, pad_len))
            noisy_phase = torch.nn.functional.pad(noisy_phase, (0, pad_len))
        
        batch_noisy_mag.append(noisy_mag)
        batch_clean_mag.append(clean_mag)
        batch_ideal_mask.append(ideal_mask)
        batch_noisy_phase.append(noisy_phase)
        filenames.append(item['filename'])
        
        # Collect audio if present
        if has_audio:
            batch_noisy_audio.append(item['noisy_audio'])
            batch_clean_audio.append(item['clean_audio'])
    
    result = {
        'noisy_mag': torch.stack(batch_noisy_mag),
        'clean_mag': torch.stack(batch_clean_mag),
        'ideal_mask': torch.stack(batch_ideal_mask),
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
    train_noisy_dir: Path,
    val_clean_dir: Path,
    val_noisy_dir: Path,
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
) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized train and validation dataloaders.
    
    Args:
        train_clean_dir: Training clean audio directory
        train_noisy_dir: Training noisy audio directory
        val_clean_dir: Validation clean audio directory
        val_noisy_dir: Validation noisy audio directory
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
        noisy_dir=train_noisy_dir,
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
    )
    
    val_dataset = SpeechEnhancementDataset(
        clean_dir=val_clean_dir,
        noisy_dir=val_noisy_dir,
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
