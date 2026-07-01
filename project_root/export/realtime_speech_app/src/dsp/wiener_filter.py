import numpy as np
import torch
from src.dsp import stft_utils
from src import config

def estimate_noise_psd(S_noisy: np.ndarray, num_noise_frames: int = 6) -> np.ndarray:
    """Estimate noise PSD from the lowest-energy frames, without external VAD labels."""
    mag, _ = stft_utils.mag_phase(S_noisy)
    num_frames = mag.shape[1]
    frame_energy = np.mean(mag ** 2, axis=0)
    idx = np.argsort(frame_energy)[: min(num_noise_frames, num_frames)]
    noise_psd = np.mean(mag[:, idx] ** 2, axis=1, keepdims=True)
    return noise_psd

def wiener_filter_stft(S_noisy: np.ndarray, noise_psd: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Wiener filtering using an internally estimated stationary noise profile.
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

def enhance_waveform(noisy: np.ndarray) -> np.ndarray:
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
    noise_psd = estimate_noise_psd(S_noisy)
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
