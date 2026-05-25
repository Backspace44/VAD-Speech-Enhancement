"""
Advanced Data Augmentation for Speech Enhancement
Implements multiple augmentation techniques for robust model training.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy.signal import resample_poly, butter, lfilter
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class AudioAugmentation:
    """Comprehensive audio augmentation pipeline for speech enhancement."""
    
    def __init__(self, config: Dict[str, Any], sample_rate: int = 16000):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Augmentation configuration dictionary
            sample_rate: Audio sample rate in Hz
        """
        self.config = config
        self.sample_rate = sample_rate
        self.enabled = config.get("enabled", True)
        
    def __call__(self, audio: np.ndarray, is_training: bool = True) -> np.ndarray:
        """
        Apply augmentation pipeline to audio.
        
        Args:
            audio: Input audio signal (1D numpy array)
            is_training: Whether in training mode (augmentation only during training)
            
        Returns:
            Augmented audio signal
        """
        if not self.enabled or not is_training:
            return audio
            

        audio = self._maybe_apply(audio, self.random_gain, "random_gain")
        audio = self._maybe_apply(audio, self.add_noise, "add_noise")
        audio = self._maybe_apply(audio, self.time_stretch, "time_stretch")
        audio = self._maybe_apply(audio, self.pitch_shift, "pitch_shift")
        audio = self._maybe_apply(audio, self.apply_reverb, "apply_reverb")
        audio = self._maybe_apply(audio, self.random_eq, "random_eq")
        audio = self._maybe_apply(audio, self.dynamic_range_compression, "dynamic_range_compression")
        audio = self._maybe_apply(audio, self.add_clipping, "add_clipping")
        

        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _maybe_apply(
        self, 
        audio: np.ndarray, 
        augment_fn, 
        config_key: str
    ) -> np.ndarray:
        """Apply augmentation with probability."""
        if config_key not in self.config:
            return audio
            
        aug_config = self.config[config_key]
        if isinstance(aug_config, dict):
            prob = aug_config.get("prob", 0.0)
        else:
            prob = 0.0
            
        if np.random.rand() < prob:
            try:
                return augment_fn(audio, aug_config)
            except Exception as e:
                logger.warning(f"Augmentation {config_key} failed: {e}")
                return audio
        return audio
    
    def random_gain(self, audio: np.ndarray, config: Dict) -> np.ndarray:
        """
        Apply random gain/volume change.
        
        Args:
            audio: Input audio
            config: {"prob": 0.5, "min_gain_db": -6, "max_gain_db": 6}
        """
        min_gain_db = config.get("min_gain_db", -6)
        max_gain_db = config.get("max_gain_db", 6)
        
        gain_db = np.random.uniform(min_gain_db, max_gain_db)
        gain = 10 ** (gain_db / 20.0)
        
        return audio * gain
    
    def add_noise(self, audio: np.ndarray, config: Dict) -> np.ndarray:
        """
        Add random white/colored noise.
        
        Args:
            audio: Input audio
            config: {"prob": 0.3, "snr_db_range": [5, 20], "noise_type": "white"}
        """
        snr_db_range = config.get("snr_db_range", [5, 20])
        noise_type = config.get("noise_type", "white")
        

        if noise_type == "white":
            noise = np.random.randn(len(audio))
        elif noise_type == "pink":
            noise = self._generate_pink_noise(len(audio))
        elif noise_type == "brown":
            noise = self._generate_brown_noise(len(audio))
        else:
            noise = np.random.randn(len(audio))
        

        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        

        snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
        snr_linear = 10 ** (snr_db / 10.0)
        

        if noise_power > 0:
            noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
            noise = noise * noise_scale
        
        return audio + noise
    
    def time_stretch(self, audio: np.ndarray, config: Dict) -> np.ndarray:
        """
        Time stretching (change speed without changing pitch).
        
        Args:
            audio: Input audio
            config: {"prob": 0.3, "rate_range": [0.9, 1.1]}
        """
        rate_range = config.get("rate_range", [0.9, 1.1])
        rate = np.random.uniform(rate_range[0], rate_range[1])
        

        new_length = int(len(audio) / rate)
        # Use resample_poly for better quality (avoid FFT artifacts)
        # Since we need arbitrary length, we'll use the ratio-based approach
        stretched = resample_poly(audio, new_length, len(audio))
        

        if len(stretched) > len(audio):
            return stretched[:len(audio)]
        elif len(stretched) < len(audio):
            return np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
        return stretched
    
    def pitch_shift(self, audio: np.ndarray, config: Dict) -> np.ndarray:
        """
        Pitch shifting (change pitch without changing duration).
        
        Args:
            audio: Input audio
            config: {"prob": 0.2, "semitone_range": [-2, 2]}
        """
        semitone_range = config.get("semitone_range", [-2, 2])
        semitones = np.random.uniform(semitone_range[0], semitone_range[1])
        

        rate = 2 ** (semitones / 12.0)
        

        shifted_length = int(len(audio) * rate)
        shifted = resample_poly(audio, shifted_length, len(audio))
        

        final = resample_poly(shifted, len(audio), shifted_length)
        
        return final
    
    def apply_reverb(self, audio: np.ndarray, config: Dict) -> np.ndarray:
        """
        Apply simple room reverb simulation.
        
        Args:
            audio: Input audio
            config: {"prob": 0.3, "room_size": [0.1, 0.5], "damping": [0.3, 0.7]}
        """
        room_size_range = config.get("room_size", [0.1, 0.5])
        damping_range = config.get("damping", [0.3, 0.7])
        
        room_size = np.random.uniform(room_size_range[0], room_size_range[1])
        damping = np.random.uniform(damping_range[0], damping_range[1])
        

        reverb_time = int(self.sample_rate * room_size)
        if reverb_time < 1:
            return audio
            

        impulse = np.exp(-np.arange(reverb_time) * damping / reverb_time)
        impulse = impulse / np.sum(impulse)
        

        reverb_audio = np.convolve(audio, impulse, mode='same')
        

        wet_dry = config.get("wet_dry_mix", 0.3)
        return (1 - wet_dry) * audio + wet_dry * reverb_audio
    
    def random_eq(self, audio: np.ndarray, config: Dict) -> np.ndarray:
        """
        Apply random equalization (frequency filtering).
        
        Args:
            audio: Input audio
            config: {"prob": 0.4, "bands": 3, "gain_range_db": [-6, 6]}
        """
        num_bands = config.get("bands", 3)
        gain_range_db = config.get("gain_range_db", [-6, 6])
        

        if num_bands == 3:
            bands = [
                (20, 500),
                (500, 2000),
                (2000, 8000),
            ]
        else:

            nyquist = self.sample_rate / 2
            bands = []
            freqs = np.logspace(np.log10(20), np.log10(nyquist), num_bands + 1)
            for i in range(num_bands):
                bands.append((freqs[i], freqs[i + 1]))
        
        result = audio.copy()
        
        for low_freq, high_freq in bands:

            gain_db = np.random.uniform(gain_range_db[0], gain_range_db[1])
            gain = 10 ** (gain_db / 20.0)
            

            filtered = self._bandpass_filter(audio, low_freq, high_freq)
            

            result = result - filtered + gain * filtered
        
        return result
    
    def dynamic_range_compression(self, audio: np.ndarray, config: Dict) -> np.ndarray:
        """
        Apply dynamic range compression.
        
        Args:
            audio: Input audio
            config: {"prob": 0.3, "threshold_db": -20, "ratio": 4, "attack": 0.005, "release": 0.1}
        """
        threshold_db = config.get("threshold_db", -20)
        ratio = config.get("ratio", 4)
        attack_time = config.get("attack", 0.005)
        release_time = config.get("release", 0.1)
        
        threshold = 10 ** (threshold_db / 20.0)
        

        attack_samples = int(attack_time * self.sample_rate)
        release_samples = int(release_time * self.sample_rate)
        
        envelope = np.abs(audio)
        smoothed_envelope = np.copy(envelope)
        
        for i in range(1, len(envelope)):
            if envelope[i] > smoothed_envelope[i - 1]:

                alpha = 1.0 / attack_samples if attack_samples > 0 else 1.0
            else:

                alpha = 1.0 / release_samples if release_samples > 0 else 1.0
            
            smoothed_envelope[i] = (alpha * envelope[i] + 
                                    (1 - alpha) * smoothed_envelope[i - 1])
        

        gain = np.ones_like(audio)
        over_threshold = smoothed_envelope > threshold
        
        if np.any(over_threshold):

            gain[over_threshold] = (
                threshold / smoothed_envelope[over_threshold]
            ) ** (1 - 1 / ratio)
        
        return audio * gain
    
    def add_clipping(self, audio: np.ndarray, config: Dict) -> np.ndarray:
        """
        Add subtle clipping distortion.
        
        Args:
            audio: Input audio
            config: {"prob": 0.2, "threshold_range": [0.7, 0.95]}
        """
        threshold_range = config.get("threshold_range", [0.7, 0.95])
        threshold = np.random.uniform(threshold_range[0], threshold_range[1])
        

        return np.tanh(audio / threshold) * threshold
    
    def _bandpass_filter(
        self, 
        audio: np.ndarray, 
        low_freq: float, 
        high_freq: float, 
        order: int = 4
    ) -> np.ndarray:
        """Apply bandpass filter."""
        nyquist = self.sample_rate / 2
        low = max(low_freq / nyquist, 0.001)
        high = min(high_freq / nyquist, 0.999)
        
        if low >= high:
            return audio
        
        try:
            b, a = butter(order, [low, high], btype='band')
            return lfilter(b, a, audio)
        except Exception:
            return audio
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise (1/f spectrum)."""

        white = np.random.randn(length)
        pink = np.copy(white)
        

        for _ in range(3):
            pink = np.cumsum(pink)
            pink = pink - np.mean(pink)
        

        if np.std(pink) > 0:
            pink = pink / np.std(pink)
        
        return pink
    
    def _generate_brown_noise(self, length: int) -> np.ndarray:
        """Generate brown noise (1/f^2 spectrum)."""
        white = np.random.randn(length)
        brown = np.cumsum(white)
        

        if np.std(brown) > 0:
            brown = brown / np.std(brown)
        
        return brown


class SpecAugment:
    """SpecAugment for spectrograms (frequency and time masking)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SpecAugment.
        
        Args:
            config: Configuration with freq_mask_param, time_mask_param, num_masks
        """
        self.config = config
        self.enabled = config.get("enabled", True)
    
    def __call__(
        self, 
        spectrogram: np.ndarray, 
        is_training: bool = True
    ) -> np.ndarray:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram (freq x time)
            is_training: Whether in training mode
            
        Returns:
            Augmented spectrogram
        """
        if not self.enabled or not is_training:
            return spectrogram
        
        prob = self.config.get("prob", 0.5)
        if np.random.rand() > prob:
            return spectrogram
        
        spec = spectrogram.copy()
        

        freq_mask_param = self.config.get("freq_mask_param", 15)
        num_freq_masks = self.config.get("num_freq_masks", 1)
        
        for _ in range(num_freq_masks):
            spec = self._freq_mask(spec, freq_mask_param)
        

        time_mask_param = self.config.get("time_mask_param", 25)
        num_time_masks = self.config.get("num_time_masks", 1)
        
        for _ in range(num_time_masks):
            spec = self._time_mask(spec, time_mask_param)
        
        return spec
    
    def _freq_mask(self, spec: np.ndarray, F: int) -> np.ndarray:
        """Apply frequency masking."""
        freq_bins = spec.shape[0]
        f = np.random.randint(0, min(F, freq_bins))
        f0 = np.random.randint(0, freq_bins - f)
        
        spec_masked = spec.copy()
        spec_masked[f0:f0 + f, :] = 0
        
        return spec_masked
    
    def _time_mask(self, spec: np.ndarray, T: int) -> np.ndarray:
        """Apply time masking."""
        time_steps = spec.shape[1]
        t = np.random.randint(0, min(T, time_steps))
        t0 = np.random.randint(0, time_steps - t)
        
        spec_masked = spec.copy()
        spec_masked[:, t0:t0 + t] = 0
        
        return spec_masked


class MixupAugmentation:
    """Mixup augmentation for speech enhancement."""
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.3):
        """
        Initialize Mixup augmentation.
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying mixup
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self, 
        batch_x: np.ndarray, 
        batch_y: np.ndarray, 
        is_training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup to a batch.
        
        Args:
            batch_x: Input batch (B, ...)
            batch_y: Target batch (B, ...)
            is_training: Whether in training mode
            
        Returns:
            Mixed batch_x, batch_y
        """
        if not is_training or np.random.rand() > self.prob:
            return batch_x, batch_y
        
        batch_size = len(batch_x)
        if batch_size < 2:
            return batch_x, batch_y
        

        lam = np.random.beta(self.alpha, self.alpha)
        

        indices = np.random.permutation(batch_size)
        

        mixed_x = lam * batch_x + (1 - lam) * batch_x[indices]
        mixed_y = lam * batch_y + (1 - lam) * batch_y[indices]
        
        return mixed_x, mixed_y


def create_audio_augmenter(level: str = 'moderate', sample_rate: int = 16000, **kwargs) -> AudioAugmentation:
    """
    Create audio augmentation pipeline based on level.
    
    Args:
        level: Augmentation level ('light', 'moderate', 'aggressive', 'none')
        sample_rate: Sample rate (default: 16000)
        **kwargs: Additional parameters (ignored for compatibility)
        
    Returns:
        AudioAugmentation instance configured for the specified level
    """
    if level == 'none' or level is None:
        return None
    
    if level == 'light':
        config = {
            "enabled": True,
            "random_gain": {"prob": 0.3, "min_gain_db": -3, "max_gain_db": 3},
            "add_noise": {"prob": 0.2, "snr_db_range": [30, 50]},
            "time_stretch": {"prob": 0.2, "rate_range": [0.95, 1.05]},
            "pitch_shift": {"prob": 0.2, "semitone_range": [-1, 1]}
        }
    elif level == 'moderate':
        config = {
            "enabled": True,
            "random_gain": {"prob": 0.5, "min_gain_db": -6, "max_gain_db": 6},
            "add_noise": {"prob": 0.3, "snr_db_range": [20, 40]},
            "time_stretch": {"prob": 0.3, "rate_range": [0.9, 1.1]},
            "pitch_shift": {"prob": 0.3, "semitone_range": [-2, 2]},
            "apply_reverb": {"prob": 0.2}
        }
    elif level == 'aggressive':
        config = {
            "enabled": True,
            "random_gain": {"prob": 0.7, "min_gain_db": -10, "max_gain_db": 10},
            "add_noise": {"prob": 0.5, "snr_db_range": [10, 30]},
            "time_stretch": {"prob": 0.5, "rate_range": [0.85, 1.15]},
            "pitch_shift": {"prob": 0.5, "semitone_range": [-3, 3]},
            "apply_reverb": {"prob": 0.3},
            "random_eq": {"prob": 0.3},
            "dynamic_range_compression": {"prob": 0.2}
        }
    else:
        raise ValueError(f"Unknown augmentation level: {level}")
    
    return AudioAugmentation(config=config, sample_rate=sample_rate)


def create_specaugment(
    freq_mask_param: int = 15,
    time_mask_param: int = 35,
    num_freq_masks: int = 2,
    num_time_masks: int = 2
) -> SpecAugment:
    """
    Create SpecAugment instance with specified parameters.
    
    Args:
        freq_mask_param: Maximum frequency mask size
        time_mask_param: Maximum time mask size
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply
        
    Returns:
        SpecAugment instance
    """
    return SpecAugment(
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        num_time_masks=num_time_masks
    )
