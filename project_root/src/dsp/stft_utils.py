import numpy as np
import torch
from src import config

def frame_signal(x: np.ndarray, frame_len: int | None = None, hop_len: int | None = None) -> np.ndarray:
    if frame_len is None:
        frame_len = config.FRAME_LEN
    if hop_len is None:
        hop_len = config.HOP_LEN
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x.astype(np.float32)
    n = len(x)
    if n < frame_len:
        pad = frame_len - n
        x = np.pad(x, (0, pad))
        n = len(x)
    num_frames = 1 + (n - frame_len) // hop_len
    shape = (num_frames, frame_len)
    frames = np.zeros(shape, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        frames[i] = x[start:end]
    return frames

def stft(x: np.ndarray, n_fft: int | None = None, hop_length: int | None = None, win_length: int | None = None) -> np.ndarray:
    if n_fft is None:
        n_fft = config.N_FFT
    if hop_length is None:
        hop_length = config.HOP_LEN
    if win_length is None:
        win_length = config.FRAME_LEN
    frames = frame_signal(x, frame_len=win_length, hop_len=hop_length)
    window = np.hanning(win_length).astype(np.float32)
    frames = frames * window[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return spec.T

def istft(S: np.ndarray, hop_length: int | None = None, win_length: int | None = None, n_fft: int | None = None, length: int | None = None) -> np.ndarray:
    if hop_length is None:
        hop_length = config.HOP_LEN
    if win_length is None:
        win_length = config.FRAME_LEN
    if n_fft is None:
        n_fft = config.N_FFT
    S = S.T
    num_frames = S.shape[0]
    window = np.hanning(win_length).astype(np.float32)
    # Use n_fft for irfft to match forward STFT, then extract win_length samples
    frames_full = np.fft.irfft(S, n=n_fft, axis=1).astype(np.float32)
    frames = frames_full[:, :win_length]
    frames *= window[None, :]
    out_len = hop_length * (num_frames - 1) + win_length
    y = np.zeros(out_len, dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        y[start:end] += frames[i]
        win_sum[start:end] += window
    nonzero = win_sum > 1e-8
    y[nonzero] /= win_sum[nonzero]
    if length is not None:
        if length < len(y):
            y = y[:length]
        elif length > len(y):
            # Pad with zeros to reach target length
            y = np.pad(y, (0, length - len(y)), mode='constant', constant_values=0)
    return y

def mag_phase(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.abs(S)
    phase = np.angle(S)
    return mag, phase

def from_mag_phase(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return mag * np.exp(1j * phase)


def complex_to_channels(spec: torch.Tensor) -> torch.Tensor:
    """Convert a complex spectrogram to real/imaginary channels."""
    if not torch.is_complex(spec):
        raise TypeError("Expected a complex tensor")
    if spec.dim() == 2:
        return torch.stack((spec.real, spec.imag), dim=0)
    if spec.dim() == 3:
        return torch.stack((spec.real, spec.imag), dim=1)
    raise ValueError(f"Expected spectrogram with 2 or 3 dimensions, got {spec.dim()}")


def channels_to_complex(channels: torch.Tensor) -> torch.Tensor:
    """Convert real/imaginary channels back to a complex spectrogram."""
    if channels.dim() == 3 and channels.shape[0] == 2:
        return torch.complex(channels[0], channels[1])
    if channels.dim() == 4 and channels.shape[1] == 2:
        return torch.complex(channels[:, 0], channels[:, 1])
    raise ValueError(
        "Expected real/imaginary channels with shape [2, F, T] or [B, 2, F, T]"
    )


def compute_complex_ratio_mask(
    clean_complex: torch.Tensor,
    noisy_complex: torch.Tensor,
    eps: float = 1e-8,
    clip_value: float | None = 5.0,
) -> torch.Tensor:
    """
    Compute the complex ratio mask (CRM) that maps noisy STFT to clean STFT.

    The returned tensor uses two channels: [real_mask, imaginary_mask].
    """
    noisy_real = noisy_complex.real
    noisy_imag = noisy_complex.imag
    clean_real = clean_complex.real
    clean_imag = clean_complex.imag

    denom = noisy_real.square() + noisy_imag.square() + eps
    mask_real = (clean_real * noisy_real + clean_imag * noisy_imag) / denom
    mask_imag = (clean_imag * noisy_real - clean_real * noisy_imag) / denom

    dim = 0 if clean_complex.dim() == 2 else 1
    mask = torch.stack((mask_real, mask_imag), dim=dim)
    if clip_value is not None and clip_value > 0:
        mask = torch.clamp(mask, -clip_value, clip_value)
    return mask


def apply_complex_mask(noisy_complex: torch.Tensor, complex_mask: torch.Tensor) -> torch.Tensor:
    """Apply a two-channel complex mask to a noisy complex spectrogram."""
    return noisy_complex * channels_to_complex(complex_mask)


def compute_stft(
    audio: torch.Tensor,
    n_fft: int = None,
    hop_length: int = None,
    win_length: int = None,
    return_complex: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute STFT and return magnitude and phase tensors.
    
    Args:
        audio: Input audio tensor [B, T] or [T]
        n_fft: FFT size (default: config.N_FFT)
        hop_length: Hop length (default: config.HOP_LEN)
        win_length: Window length (default: config.FRAME_LEN)
        return_complex: Return complex spectrogram instead of mag/phase
        
    Returns:
        If return_complex=False: (magnitude, phase) tensors [B, F, T] or [F, T]
        If return_complex=True: complex_spectrogram tensor
    """
    if n_fft is None:
        n_fft = config.N_FFT
    if hop_length is None:
        hop_length = config.HOP_LEN
    if win_length is None:
        win_length = config.FRAME_LEN
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    window = torch.hann_window(win_length, device=audio.device)
    
    stft_result = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode='reflect'
    )
    
    if return_complex:
        if squeeze_output:
            return stft_result.squeeze(0)
        return stft_result
    
    magnitude = stft_result.abs()
    phase = stft_result.angle()
    
    if squeeze_output:
        magnitude = magnitude.squeeze(0)
        phase = phase.squeeze(0)
    
    return magnitude, phase


def inverse_stft(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    n_fft: int = None,
    hop_length: int = None,
    win_length: int = None,
    length: int = None
) -> torch.Tensor:
    """
    Inverse STFT from magnitude and phase tensors.
    
    Args:
        magnitude: Magnitude tensor [B, F, T] or [F, T]
        phase: Phase tensor [B, F, T] or [F, T]
        n_fft: FFT size (default: config.N_FFT)
        hop_length: Hop length (default: config.HOP_LEN)
        win_length: Window length (default: config.FRAME_LEN)
        length: Target length for output (optional)
        
    Returns:
        Reconstructed audio tensor [B, L] or [L]
    """
    if n_fft is None:
        n_fft = config.N_FFT
    if hop_length is None:
        hop_length = config.HOP_LEN
    if win_length is None:
        win_length = config.FRAME_LEN
    
    if magnitude.dim() == 2:
        magnitude = magnitude.unsqueeze(0)
        phase = phase.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    complex_spec = magnitude * torch.exp(1j * phase)
    
    window = torch.hann_window(win_length, device=magnitude.device)
    
    audio = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length
    )
    
    if squeeze_output:
        audio = audio.squeeze(0)
    
    return audio


def inverse_complex_stft(
    complex_spectrogram: torch.Tensor,
    n_fft: int = None,
    hop_length: int = None,
    win_length: int = None,
    length: int = None,
) -> torch.Tensor:
    """
    Inverse STFT directly from a complex spectrogram tensor.

    Args:
        complex_spectrogram: Complex tensor [B, F, T] or [F, T]
    """
    if n_fft is None:
        n_fft = config.N_FFT
    if hop_length is None:
        hop_length = config.HOP_LEN
    if win_length is None:
        win_length = config.FRAME_LEN

    if complex_spectrogram.dim() == 2:
        complex_spectrogram = complex_spectrogram.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    window = torch.hann_window(win_length, device=complex_spectrogram.device)
    audio = torch.istft(
        complex_spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )

    if squeeze_output:
        audio = audio.squeeze(0)
    return audio
