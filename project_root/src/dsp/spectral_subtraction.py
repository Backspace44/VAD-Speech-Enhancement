import numpy as np
import torch
from typing import Optional
from src.dsp import stft_utils
from src import config

def estimate_noise_mag(S_noisy: np.ndarray, vad_labels: Optional[np.ndarray] = None, num_noise_frames: int = 6) -> np.ndarray:
    mag, _ = stft_utils.mag_phase(S_noisy)
    num_frames = mag.shape[1]
    if vad_labels is not None:
        vad_labels = vad_labels[:num_frames]
        idx = np.where(vad_labels == 0)[0]
        if len(idx) == 0:
            idx = np.arange(min(num_noise_frames, num_frames))
    else:
        idx = np.arange(min(num_noise_frames, num_frames))
    noise_mag = np.mean(mag[:, idx], axis=1, keepdims=True)
    return noise_mag

def spectral_subtraction_with_tracking(S_noisy: np.ndarray, noise_mag: np.ndarray, vad_labels: Optional[np.ndarray] = None, alpha: float = 1.0, beta: float = 0.02, tracking_alpha: float = 0.95) -> np.ndarray:
    """
    Spectral subtraction with adaptive noise tracking.
    
    Args:
        S_noisy: Noisy STFT spectrogram [F, T]
        noise_mag: Initial noise magnitude estimate [F, 1] or [F, T]
        vad_labels: VAD labels (0=noise, 1=speech) for noise tracking
        alpha: Over-subtraction factor
        beta: Spectral floor factor
        tracking_alpha: Smoothing factor for noise tracking (0.9-0.99)
        
    Returns:
        Enhanced STFT spectrogram
    """
    mag, phase = stft_utils.mag_phase(S_noisy)
    num_frames = mag.shape[1]
    
    # Initialize noise estimate
    if noise_mag.shape[1] == 1:
        noise_est = np.repeat(noise_mag, num_frames, axis=1)
    else:
        noise_est = noise_mag.copy()
    
    # Process frame-by-frame with noise tracking
    mag_enh = np.zeros_like(mag)
    
    for t in range(num_frames):
        # Update noise estimate if current frame is silence
        if vad_labels is not None and t < len(vad_labels) and vad_labels[t] == 0:
            # Adaptive noise tracking: noise_est = alpha * noise_est + (1-alpha) * mag_current
            noise_est[:, t] = tracking_alpha * noise_est[:, t] + (1 - tracking_alpha) * mag[:, t]
        
        # Spectral subtraction
        mag_sub = mag[:, t] - alpha * noise_est[:, t]
        mag_floor = beta * noise_est[:, t]
        mag_enh[:, t] = np.maximum(mag_sub, mag_floor)
    
    S_enh = stft_utils.from_mag_phase(mag_enh, phase)
    return S_enh

def spectral_subtraction(S_noisy: np.ndarray, noise_mag: np.ndarray, alpha: float = 1.0, beta: float = 0.02) -> np.ndarray:
    """
    Basic spectral subtraction without noise tracking (legacy version).
    """
    mag, phase = stft_utils.mag_phase(S_noisy)
    if noise_mag.shape[1] == 1:
        noise_mag = np.repeat(noise_mag, mag.shape[1], axis=1)
    mag_sub = mag - alpha * noise_mag
    mag_floor = beta * noise_mag
    mag_enh = np.maximum(mag_sub, mag_floor)
    S_enh = stft_utils.from_mag_phase(mag_enh, phase)
    return S_enh

def enhance_waveform(noisy: np.ndarray, vad_labels: Optional[np.ndarray] = None, alpha: float = 1.0, beta: float = 0.02, use_tracking: bool = True) -> np.ndarray:
    if noisy.ndim > 1:
        noisy = np.mean(noisy, axis=1)
    noisy = noisy.astype(np.float32)
    
    # Use Torch STFT for consistency with model-based methods
    noisy_tensor = torch.from_numpy(noisy).float()
    noisy_mag, noisy_phase = stft_utils.compute_stft(
        noisy_tensor,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LEN,
        win_length=config.FRAME_LEN
    )
    
    # Convert to complex spectrogram for compatibility with existing functions
    S_noisy = noisy_mag.numpy() * np.exp(1j * noisy_phase.numpy())
    noise_mag = estimate_noise_mag(S_noisy, vad_labels=vad_labels)
    
    # Use tracking version if VAD labels are available
    if use_tracking and vad_labels is not None:
        S_enh = spectral_subtraction_with_tracking(S_noisy, noise_mag, vad_labels=vad_labels, alpha=alpha, beta=beta)
    else:
        S_enh = spectral_subtraction(S_noisy, noise_mag, alpha=alpha, beta=beta)
    
    # Extract magnitude and phase from enhanced complex spectrogram
    mag_enh = np.abs(S_enh)
    phase_enh = np.angle(S_enh)
    
    # Use Torch ISTFT for reconstruction
    mag_enh_tensor = torch.from_numpy(mag_enh).float()
    phase_enh_tensor = torch.from_numpy(phase_enh).float()
    enhanced_tensor = stft_utils.inverse_stft(
        mag_enh_tensor,
        phase_enh_tensor,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LEN,
        win_length=config.FRAME_LEN,
        length=len(noisy)
    )
    
    enhanced = enhanced_tensor.numpy()
    return enhanced
