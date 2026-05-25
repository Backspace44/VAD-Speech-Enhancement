import numpy as np
import torch
from typing import Optional
from src.dsp import stft_utils
from src import config

def estimate_noise_psd(S_noisy: np.ndarray, vad_labels: Optional[np.ndarray] = None, num_noise_frames: int = 6) -> np.ndarray:
    mag, _ = stft_utils.mag_phase(S_noisy)
    num_frames = mag.shape[1]
    if vad_labels is not None:
        vad_labels = vad_labels[:num_frames]
        idx = np.where(vad_labels == 0)[0]
        if len(idx) == 0:
            idx = np.arange(min(num_noise_frames, num_frames))
    else:
        idx = np.arange(min(num_noise_frames, num_frames))
    noise_psd = np.mean(mag[:, idx] ** 2, axis=1, keepdims=True)
    return noise_psd

def wiener_filter_stft_with_tracking(S_noisy: np.ndarray, noise_psd: np.ndarray, vad_labels: Optional[np.ndarray] = None, eps: float = 1e-12, tracking_alpha: float = 0.95) -> np.ndarray:
    """
    Wiener filtering with adaptive noise tracking.
    
    Args:
        S_noisy: Noisy STFT spectrogram [F, T]
        noise_psd: Initial noise PSD estimate [F, 1] or [F, T]
        vad_labels: VAD labels (0=noise, 1=speech) for noise tracking
        eps: Small constant to avoid division by zero
        tracking_alpha: Smoothing factor for noise tracking (0.9-0.99)
        
    Returns:
        Enhanced STFT spectrogram
    """
    mag, phase = stft_utils.mag_phase(S_noisy)
    num_frames = mag.shape[1]
    
    # Initialize noise PSD estimate
    if noise_psd.shape[1] == 1:
        noise_psd_est = np.repeat(noise_psd, num_frames, axis=1)
    else:
        noise_psd_est = noise_psd.copy()
    
    # Process frame-by-frame with noise tracking
    mag_enh = np.zeros_like(mag)
    
    for t in range(num_frames):
        # Update noise PSD if current frame is silence
        if vad_labels is not None and t < len(vad_labels) and vad_labels[t] == 0:
            # Adaptive noise tracking: noise_psd = alpha * noise_psd + (1-alpha) * mag^2_current
            noise_psd_est[:, t] = tracking_alpha * noise_psd_est[:, t] + (1 - tracking_alpha) * (mag[:, t] ** 2)
        
        # Wiener filtering
        Y2 = mag[:, t] ** 2
        snr_post = Y2 / (noise_psd_est[:, t] + eps)
        G = snr_post / (snr_post + 1.0)
        mag_enh[:, t] = G * mag[:, t]
    
    S_enh = stft_utils.from_mag_phase(mag_enh, phase)
    return S_enh

def wiener_filter_stft(S_noisy: np.ndarray, noise_psd: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Basic Wiener filtering without noise tracking (legacy version).
    """
    mag, phase = stft_utils.mag_phase(S_noisy)
    if noise_psd.shape[1] == 1:
        noise_psd = np.repeat(noise_psd, mag.shape[1], axis=1)
    Y2 = mag**2
    snr_post = Y2 / (noise_psd + eps)
    G = snr_post / (snr_post + 1.0)
    mag_enh = G * mag
    S_enh = stft_utils.from_mag_phase(mag_enh, phase)
    return S_enh

def enhance_waveform(noisy: np.ndarray, vad_labels: Optional[np.ndarray] = None, use_tracking: bool = True) -> np.ndarray:
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
    noise_psd = estimate_noise_psd(S_noisy, vad_labels=vad_labels)
    
    # Use tracking version if VAD labels are available
    if use_tracking and vad_labels is not None:
        S_enh = wiener_filter_stft_with_tracking(S_noisy, noise_psd, vad_labels=vad_labels)
    else:
        S_enh = wiener_filter_stft(S_noisy, noise_psd)
    
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
