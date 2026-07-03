from __future__ import annotations
from typing import Dict
import numpy as np
from src import config

try:
    from pystoi.stoi import stoi as _stoi
except ImportError:
    _stoi = None

try:
    from pesq import pesq as _pesq
except ImportError:
    _pesq = None

def snr_db(clean: np.ndarray, processed: np.ndarray) -> float:
    if clean.size == 0 or processed.size == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if clean.ndim > 1:
        clean = np.mean(clean, axis=1)
    if processed.ndim > 1:
        processed = np.mean(processed, axis=1)
    length = min(len(clean), len(processed))
    
    if length == 0:
        raise ValueError("Arrays have no overlapping samples")
    
    clean = clean[:length].astype(np.float32)
    processed = processed[:length].astype(np.float32)
    noise = clean - processed
    eps = 1e-12
    p_sig = np.mean(clean**2) + eps
    p_noise = np.mean(noise**2) + eps
    return float(10.0 * np.log10(p_sig / p_noise))

def stoi_score(clean: np.ndarray, processed: np.ndarray, sr: int | None = None) -> float:
    if _stoi is None:
        raise ImportError("pystoi not installed")
    if sr is None:
        sr = config.SAMPLE_RATE
    length = min(len(clean), len(processed))
    clean = clean[:length].astype(np.float32)
    processed = processed[:length].astype(np.float32)
    return float(_stoi(clean, processed, sr, extended=False))

def pesq_score(clean: np.ndarray, processed: np.ndarray, sr: int | None = None) -> float:
    if _pesq is None:
        raise ImportError("pesq not installed")
    if sr is None:
        sr = config.SAMPLE_RATE
    length = min(len(clean), len(processed))
    clean = clean[:length].astype(np.float32)
    processed = processed[:length].astype(np.float32)
    return float(_pesq(sr, clean, processed, "wb"))

# Backward-compatible names.
def compute_snr(clean: np.ndarray, processed: np.ndarray, sr: int | None = None) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    return snr_db(clean, processed)

def compute_stoi(clean: np.ndarray, processed: np.ndarray, sr: int | None = None) -> float:
    """Compute Short-Time Objective Intelligibility."""
    return stoi_score(clean, processed, sr)

def compute_pesq(clean: np.ndarray, processed: np.ndarray, sr: int | None = None) -> float:
    """Compute Perceptual Evaluation of Speech Quality."""
    return pesq_score(clean, processed, sr)

def compute_sisdr(clean: np.ndarray, processed: np.ndarray, sr: int | None = None) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio."""
    if clean.ndim > 1:
        clean = np.mean(clean, axis=1)
    if processed.ndim > 1:
        processed = np.mean(processed, axis=1)
    
    length = min(len(clean), len(processed))
    clean = clean[:length].astype(np.float32)
    processed = processed[:length].astype(np.float32)
    
    alpha = np.dot(processed, clean) / (np.dot(clean, clean) + 1e-12)
    s_target = alpha * clean
    e_noise = processed - s_target
    
    eps = 1e-12
    sisdr = 10 * np.log10(np.sum(s_target**2) / (np.sum(e_noise**2) + eps) + eps)
    return float(sisdr)

def evaluate_vad(vad_true: np.ndarray, vad_pred: np.ndarray) -> Dict[str, float]:
    vad_true = vad_true.astype(int)
    vad_pred = vad_pred.astype(int)
    n = min(len(vad_true), len(vad_pred))
    vad_true = vad_true[:n]
    vad_pred = vad_pred[:n]
    tp = int(((vad_true == 1) & (vad_pred == 1)).sum())
    tn = int(((vad_true == 0) & (vad_pred == 0)).sum())
    fp = int(((vad_true == 0) & (vad_pred == 1)).sum())
    fn = int(((vad_true == 1) & (vad_pred == 0)).sum())
    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    miss = fn / (tp + fn + eps)
    fa = fp / (fp + tn + eps)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "miss_rate": float(miss),
        "false_alarm_rate": float(fa),
    }


def segmental_snr(clean: np.ndarray, processed: np.ndarray, frame_len: int = 256) -> float:
    """Compute Segmental SNR."""
    if clean.ndim > 1:
        clean = np.mean(clean, axis=1)
    if processed.ndim > 1:
        processed = np.mean(processed, axis=1)
    
    length = min(len(clean), len(processed))
    clean = clean[:length].astype(np.float32)
    processed = processed[:length].astype(np.float32)
    
    noise = clean - processed
    num_frames = length // frame_len
    
    if num_frames == 0:
        return 0.0
    
    snr_vals = []
    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len
        
        clean_frame = clean[start:end]
        noise_frame = noise[start:end]
        
        p_sig = np.mean(clean_frame**2) + 1e-12
        p_noise = np.mean(noise_frame**2) + 1e-12
        
        snr_frame = 10.0 * np.log10(p_sig / p_noise)
        # Cap outliers.
        snr_frame = np.clip(snr_frame, -10, 35)
        snr_vals.append(snr_frame)
    
    return float(np.mean(snr_vals))


def log_spectral_distance(clean: np.ndarray, processed: np.ndarray) -> float:
    """Compute Log Spectral Distance."""
    if clean.ndim > 1:
        clean = np.mean(clean, axis=1)
    if processed.ndim > 1:
        processed = np.mean(processed, axis=1)
    
    length = min(len(clean), len(processed))
    clean = clean[:length].astype(np.float32)
    processed = processed[:length].astype(np.float32)
    
    from scipy import signal
    nperseg = 512
    f, _, Pxx_clean = signal.spectrogram(clean, nperseg=nperseg, noverlap=nperseg//2)
    f, _, Pxx_proc = signal.spectrogram(processed, nperseg=nperseg, noverlap=nperseg//2)
    
    min_time = min(Pxx_clean.shape[1], Pxx_proc.shape[1])
    Pxx_clean = Pxx_clean[:, :min_time]
    Pxx_proc = Pxx_proc[:, :min_time]
    
    eps = 1e-12
    lsd_frames = np.sqrt(np.mean((10 * np.log10(Pxx_clean + eps) - 10 * np.log10(Pxx_proc + eps))**2, axis=0))
    return float(np.mean(lsd_frames))


def evaluate_speech_enhancement(
    clean: np.ndarray, 
    enhanced: np.ndarray, 
    sr: int | None = None
) -> Dict[str, float]:
    """Compute enhancement metrics."""
    if sr is None:
        sr = config.SAMPLE_RATE
    
    metrics = {}
    
    try:
        metrics['pesq'] = pesq_score(clean, enhanced, sr)
    except Exception:
        metrics['pesq'] = 0.0
    
    try:
        metrics['stoi'] = stoi_score(clean, enhanced, sr)
    except Exception:
        metrics['stoi'] = 0.0
    
    try:
        metrics['snr'] = snr_db(clean, enhanced)
    except Exception:
        metrics['snr'] = 0.0
    
    try:
        metrics['segsnr'] = segmental_snr(clean, enhanced)
    except Exception:
        metrics['segsnr'] = 0.0
    
    try:
        metrics['lsd'] = log_spectral_distance(clean, enhanced)
    except Exception:
        metrics['lsd'] = 0.0
    
    return metrics
