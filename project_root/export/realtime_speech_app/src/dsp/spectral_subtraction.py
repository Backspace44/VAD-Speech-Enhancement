import numpy as np
import torch
from src.dsp import stft_utils
from src import config

def estimate_noise_mag(S_noisy: np.ndarray, num_noise_frames: int = 6) -> np.ndarray:
    """Estimate noise from the lowest-energy frames, without external VAD labels."""
    mag, _ = stft_utils.mag_phase(S_noisy)
    num_frames = mag.shape[1]
    frame_energy = np.mean(mag ** 2, axis=0)
    idx = np.argsort(frame_energy)[: min(num_noise_frames, num_frames)]
    noise_mag = np.mean(mag[:, idx], axis=1, keepdims=True)
    return noise_mag

def spectral_subtraction(S_noisy: np.ndarray, noise_mag: np.ndarray, alpha: float = 1.0, beta: float = 0.02) -> np.ndarray:
    """Apply spectral subtraction."""
    mag, phase = stft_utils.mag_phase(S_noisy)
    if noise_mag.shape[1] == 1:
        noise_mag = np.repeat(noise_mag, mag.shape[1], axis=1)
    mag_sub = mag - alpha * noise_mag
    mag_floor = beta * noise_mag
    mag_enh = np.maximum(mag_sub, mag_floor)
    S_enh = stft_utils.from_mag_phase(mag_enh, phase)
    return S_enh

def enhance_waveform(noisy: np.ndarray, alpha: float = 1.0, beta: float = 0.02) -> np.ndarray:
    if noisy.ndim > 1:
        noisy = np.mean(noisy, axis=1)
    noisy = noisy.astype(np.float32)
    
    noisy_tensor = torch.from_numpy(noisy).float()
    noisy_mag, noisy_phase = stft_utils.compute_stft(
        noisy_tensor,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LEN,
        win_length=config.FRAME_LEN
    )
    
    S_noisy = noisy_mag.numpy() * np.exp(1j * noisy_phase.numpy())
    noise_mag = estimate_noise_mag(S_noisy)
    S_enh = spectral_subtraction(S_noisy, noise_mag, alpha=alpha, beta=beta)
    
    mag_enh = np.abs(S_enh)
    phase_enh = np.angle(S_enh)
    
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
