"""
Advanced Audio Preprocessing and Normalization for Speech Enhancement.
Includes multiple normalization strategies, VAD-based processing, and quality checks.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy import signal
from scipy.signal import resample_poly
import warnings

from src import config


class AudioNormalizer:
    """Comprehensive audio normalization with multiple strategies."""
    
    def __init__(
        self,
        method: str = 'peak',
        target_level: float = -3.0,
        sample_rate: int = 16000
    ):
        """
        Initialize audio normalizer.
        
        Args:
            method: Normalization method ('peak', 'rms', 'loudness', 'percentile')
            target_level: Target level in dB (for RMS/loudness) or peak value
            sample_rate: Audio sample rate
        """
        self.method = method
        self.target_level = target_level
        self.sample_rate = sample_rate
        
        self.methods = {
            'peak': self.normalize_peak,
            'rms': self.normalize_rms,
            'loudness': self.normalize_loudness,
            'percentile': self.normalize_percentile,
            'zscore': self.normalize_zscore,
        }
        
        if method not in self.methods:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply normalization to audio."""
        return self.methods[self.method](audio)
    
    def normalize_peak(self, audio: np.ndarray, target_peak: Optional[float] = None) -> np.ndarray:
        """
        Peak normalization - scale so maximum absolute value equals target.
        
        Args:
            audio: Input audio signal
            target_peak: Target peak value (default: 0.95 for headroom)
            
        Returns:
            Normalized audio
        """
        if target_peak is None:
            target_peak = 0.95
        
        peak = np.max(np.abs(audio))
        
        if peak < 1e-8:
            warnings.warn("Audio is silent, skipping normalization")
            return audio
        
        normalized = audio * (target_peak / peak)
        
        return normalized.astype(np.float32)
    
    def normalize_rms(self, audio: np.ndarray, target_db: Optional[float] = None) -> np.ndarray:
        """
        RMS (Root Mean Square) normalization - scale to target RMS level.
        
        Args:
            audio: Input audio signal
            target_db: Target RMS level in dB (default: -20 dB)
            
        Returns:
            Normalized audio
        """
        if target_db is None:
            target_db = self.target_level if self.target_level < 0 else -20.0
        

        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 1e-8:
            warnings.warn("Audio is silent, skipping normalization")
            return audio
        

        current_db = 20 * np.log10(rms)
        

        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)
        
        normalized = audio * gain
        

        peak = np.max(np.abs(normalized))
        if peak > 0.99:
            normalized = normalized * (0.99 / peak)
        
        return normalized.astype(np.float32)
    
    def normalize_loudness(self, audio: np.ndarray, target_lufs: Optional[float] = None) -> np.ndarray:
        """
        Loudness normalization (ITU-R BS.1770 inspired).
        Perceptually-weighted loudness measurement.
        
        Args:
            audio: Input audio signal
            target_lufs: Target loudness in LUFS (default: -23 LUFS)
            
        Returns:
            Normalized audio
        """
        if target_lufs is None:
            target_lufs = -23.0
        


        audio_weighted = self._apply_k_weighting(audio)
        

        mean_square = np.mean(audio_weighted ** 2)
        
        if mean_square < 1e-16:
            warnings.warn("Audio is silent, skipping normalization")
            return audio
        

        current_lufs = -0.691 + 10 * np.log10(mean_square)
        

        gain_db = target_lufs - current_lufs
        gain = 10 ** (gain_db / 20)
        
        normalized = audio * gain
        

        peak = np.max(np.abs(normalized))
        if peak > 0.99:
            normalized = normalized * (0.99 / peak)
        
        return normalized.astype(np.float32)
    
    def normalize_percentile(
        self, 
        audio: np.ndarray, 
        percentile: float = 95.0,
        target_level: float = 0.95
    ) -> np.ndarray:
        """
        Percentile-based normalization - more robust to outliers.
        
        Args:
            audio: Input audio signal
            percentile: Percentile to use (default: 95th percentile)
            target_level: Target level for the percentile
            
        Returns:
            Normalized audio
        """
        abs_audio = np.abs(audio)
        percentile_value = np.percentile(abs_audio, percentile)
        
        if percentile_value < 1e-8:
            warnings.warn("Audio is silent, skipping normalization")
            return audio
        
        normalized = audio * (target_level / percentile_value)
        

        normalized = np.tanh(normalized)
        
        return normalized.astype(np.float32)
    
    def normalize_zscore(self, audio: np.ndarray) -> np.ndarray:
        """
        Z-score normalization - zero mean, unit variance.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Normalized audio
        """
        mean = np.mean(audio)
        std = np.std(audio)
        
        if std < 1e-8:
            warnings.warn("Audio has zero variance, skipping normalization")
            return audio - mean
        
        normalized = (audio - mean) / std
        
        return normalized.astype(np.float32)
    
    def _apply_k_weighting(self, audio: np.ndarray) -> np.ndarray:
        """Apply K-weighting filter (simplified version of ITU-R BS.1770)."""

        b_shelf, a_shelf = signal.butter(2, 1500 / (self.sample_rate / 2), btype='high')
        audio_shelf = signal.lfilter(b_shelf, a_shelf, audio)
        

        b_hp, a_hp = signal.butter(2, 50 / (self.sample_rate / 2), btype='high')
        audio_weighted = signal.lfilter(b_hp, a_hp, audio_shelf)
        
        return audio_weighted


class AudioPreprocessor:
    """Comprehensive audio preprocessing pipeline."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        normalize: bool = True,
        normalization_method: str = 'peak',
        remove_dc: bool = True,
        preemphasis: bool = False,
        preemphasis_coef: float = 0.97,
        apply_agc: bool = False,
        trim_silence: bool = False,
        silence_threshold: float = 0.01,
        vad_based: bool = False
    ):
        """
        Initialize preprocessor.
        
        Args:
            sample_rate: Target sample rate
            normalize: Whether to apply normalization
            normalization_method: Normalization strategy
            remove_dc: Remove DC offset
            preemphasis: Apply preemphasis filter
            preemphasis_coef: Preemphasis coefficient (typical: 0.97)
            apply_agc: Apply automatic gain control
            trim_silence: Trim leading/trailing silence
            silence_threshold: Threshold for silence detection (relative)
            vad_based: Use VAD for processing
        """
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.remove_dc = remove_dc
        self.preemphasis = preemphasis
        self.preemphasis_coef = preemphasis_coef
        self.apply_agc = apply_agc
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold
        self.vad_based = vad_based
        
        if normalize:
            self.normalizer = AudioNormalizer(
                method=normalization_method,
                sample_rate=sample_rate
            )
    
    def __call__(
        self, 
        audio: np.ndarray, 
        sr: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply preprocessing pipeline.
        
        Args:
            audio: Input audio signal
            sr: Sample rate of input audio (if different from target)
            
        Returns:
            Tuple of (preprocessed_audio, metadata_dict)
        """
        metadata = {'original_length': len(audio)}
        

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            metadata['converted_to_mono'] = True
        

        if sr is not None and sr != self.sample_rate:
            audio = self.resample_audio(audio, sr, self.sample_rate)
            metadata['resampled'] = True
            metadata['original_sr'] = sr
            metadata['target_sr'] = self.sample_rate
        

        if self.remove_dc:
            audio = self.remove_dc_offset(audio)
            metadata['dc_removed'] = True
        

        if self.trim_silence:
            audio, trim_info = self.trim_silence_edges(audio, self.silence_threshold)
            metadata['silence_trimmed'] = trim_info
        

        if self.normalize:
            audio = self.normalizer(audio)
            metadata['normalized'] = self.normalization_method
        

        if self.apply_agc:
            audio = self.automatic_gain_control(audio)
            metadata['agc_applied'] = True
        

        if self.preemphasis:
            audio = self.apply_preemphasis(audio, self.preemphasis_coef)
            metadata['preemphasis_applied'] = self.preemphasis_coef
        

        quality_info = self.check_audio_quality(audio)
        metadata['quality'] = quality_info
        
        metadata['final_length'] = len(audio)
        
        return audio.astype(np.float32), metadata
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate using polyphase resampling."""
        if orig_sr == target_sr:
            return audio
        
        resampled = resample_poly(audio, target_sr, orig_sr)
        
        return resampled
    
    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset (subtract mean)."""
        return audio - np.mean(audio)
    
    def apply_preemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """
        Apply preemphasis filter to boost high frequencies.
        Formula: y[n] = x[n] - coef * x[n-1]
        """
        emphasized = np.append(audio[0], audio[1:] - coef * audio[:-1])
        return emphasized
    
    def deemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """
        Reverse preemphasis filter.
        Formula: y[n] = x[n] + coef * y[n-1]
        """
        deemphasized = np.zeros_like(audio)
        deemphasized[0] = audio[0]
        
        for i in range(1, len(audio)):
            deemphasized[i] = audio[i] + coef * deemphasized[i - 1]
        
        return deemphasized
    
    def trim_silence_edges(
        self, 
        audio: np.ndarray, 
        threshold: float = 0.01
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Trim silence from beginning and end.
        
        Args:
            audio: Input audio
            threshold: Relative threshold (0-1)
            
        Returns:
            Tuple of (trimmed_audio, {start_trim, end_trim})
        """

        energy = audio ** 2
        

        window_size = int(0.02 * self.sample_rate)
        kernel = np.ones(window_size) / window_size
        smoothed_energy = np.convolve(energy, kernel, mode='same')
        

        max_energy = np.max(smoothed_energy)
        threshold_value = threshold * max_energy
        

        non_silent = smoothed_energy > threshold_value
        
        if not np.any(non_silent):

            return audio, {'start_samples': 0, 'end_samples': 0}
        

        non_silent_indices = np.where(non_silent)[0]
        start = non_silent_indices[0]
        end = non_silent_indices[-1] + 1
        

        margin = int(0.05 * self.sample_rate)
        start = max(0, start - margin)
        end = min(len(audio), end + margin)
        
        trimmed = audio[start:end]
        
        trim_info = {
            'start_samples': start,
            'end_samples': len(audio) - end,
            'total_trimmed': start + (len(audio) - end)
        }
        
        return trimmed, trim_info
    
    def automatic_gain_control(
        self, 
        audio: np.ndarray,
        attack_time: float = 0.01,
        release_time: float = 0.1,
        target_level: float = 0.5
    ) -> np.ndarray:
        """
        Apply automatic gain control (AGC) to maintain consistent volume.
        
        Args:
            audio: Input audio
            attack_time: Attack time in seconds
            release_time: Release time in seconds
            target_level: Target RMS level
            
        Returns:
            Audio with AGC applied
        """

        frame_len = int(0.02 * self.sample_rate)
        hop_len = frame_len // 2
        
        num_frames = (len(audio) - frame_len) // hop_len + 1
        

        rms = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_len
            end = start + frame_len
            if end > len(audio):
                break
            frame = audio[start:end]
            rms[i] = np.sqrt(np.mean(frame ** 2))
        

        attack_samples = int(attack_time * self.sample_rate / hop_len)
        release_samples = int(release_time * self.sample_rate / hop_len)
        
        smoothed_rms = np.zeros_like(rms)
        smoothed_rms[0] = rms[0]
        
        for i in range(1, len(rms)):
            if rms[i] > smoothed_rms[i - 1]:

                alpha = 1.0 / attack_samples if attack_samples > 0 else 1.0
            else:

                alpha = 1.0 / release_samples if release_samples > 0 else 1.0
            
            smoothed_rms[i] = alpha * rms[i] + (1 - alpha) * smoothed_rms[i - 1]
        

        gain = np.zeros_like(smoothed_rms)
        for i in range(len(smoothed_rms)):
            if smoothed_rms[i] > 1e-6:
                gain[i] = target_level / smoothed_rms[i]
            else:
                gain[i] = 1.0
        

        gain = np.clip(gain, 0.1, 10.0)
        

        gain_interp = np.interp(
            np.arange(len(audio)),
            np.arange(len(gain)) * hop_len + frame_len // 2,
            gain
        )
        

        output = audio * gain_interp
        
        return output
    
    def check_audio_quality(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Check audio quality and return diagnostic information.
        
        Returns:
            Dictionary with quality metrics
        """
        quality = {}
        

        clipped_samples = np.sum(np.abs(audio) > 0.99)
        quality['clipped_samples'] = int(clipped_samples)
        quality['clipping_ratio'] = float(clipped_samples / len(audio))
        

        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms > 1e-8:
            dynamic_range_db = 20 * np.log10(peak / rms)
        else:
            dynamic_range_db = 0.0
        
        quality['peak'] = float(peak)
        quality['rms'] = float(rms)
        quality['dynamic_range_db'] = float(dynamic_range_db)
        

        silence_ratio = np.sum(np.abs(audio) < 0.01) / len(audio)
        quality['silence_ratio'] = float(silence_ratio)
        

        sorted_energy = np.sort(audio ** 2)
        noise_floor = np.mean(sorted_energy[:len(sorted_energy) // 10])
        signal_energy = np.mean(sorted_energy[len(sorted_energy) // 2:])
        
        if noise_floor > 1e-12:
            estimated_snr = 10 * np.log10(signal_energy / noise_floor)
        else:
            estimated_snr = 60.0
        
        quality['estimated_snr_db'] = float(estimated_snr)
        

        score = 100.0
        

        if quality['clipping_ratio'] > 0.01:
            score -= 20
        elif quality['clipping_ratio'] > 0.001:
            score -= 10
        

        if dynamic_range_db < 6:
            score -= 15
        

        if silence_ratio > 0.5:
            score -= 10
        

        if estimated_snr < 10:
            score -= 15
        elif estimated_snr < 20:
            score -= 5
        
        quality['quality_score'] = float(max(0, score))
        
        return quality


def create_preprocessor(config_dict: Optional[Dict[str, Any]] = None) -> AudioPreprocessor:
    """
    Create preprocessor from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configured AudioPreprocessor instance
    """
    if config_dict is None:
        config_dict = {}
    
    return AudioPreprocessor(
        sample_rate=config_dict.get('sample_rate', 16000),
        normalize=config_dict.get('normalize', True),
        normalization_method=config_dict.get('normalization_method', 'peak'),
        remove_dc=config_dict.get('remove_dc', True),
        preemphasis=config_dict.get('preemphasis', False),
        preemphasis_coef=config_dict.get('preemphasis_coef', 0.97),
        apply_agc=config_dict.get('apply_agc', False),
        trim_silence=config_dict.get('trim_silence', False),
        silence_threshold=config_dict.get('silence_threshold', 0.01),
        vad_based=config_dict.get('vad_based', False)
    )
