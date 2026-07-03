"""Adaptive VAD using energy, ZCR, and spectral cues."""

from dataclasses import dataclass
import numpy as np

from src import config
from src.dsp.stft_utils import frame_signal
from src.dsp.metrics import evaluate_vad as eval_vad_metrics


@dataclass
class EnergyZCRVADConfig:
    """Configuration for the upgraded VAD detector."""
    frame_len: int = config.FRAME_LEN
    hop_len: int = config.HOP_LEN

    # Legacy knobs.
    energy_thresh_ratio: float = 0.25
    zcr_max_speech: float = 0.2
    min_speech_frames: int = 3
    min_silence_frames: int = 4

    # Adaptive controls.
    speech_start_threshold: float = 0.70
    speech_end_threshold: float = 0.54
    energy_weight: float = 0.45
    band_ratio_weight: float = 0.30
    zcr_weight: float = 0.15
    flatness_weight: float = 0.10
    min_energy_score_for_speech: float = 0.18
    speech_band_low_hz: float = 300.0
    speech_band_high_hz: float = 3400.0
    flatness_max_speech: float = 0.72
    hangover_frames: int = 1
    median_filter_size: int = 5


def remove_short_islands(labels: np.ndarray, value: int, min_len: int) -> np.ndarray:
    """Remove short segments of a specific value from binary labels."""
    labels = labels.copy()
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == value:
            j = i
            while j < n and labels[j] == value:
                j += 1
            if j - i < min_len:
                labels[i:j] = 1 - value
            i = j
        else:
            i += 1
    return labels


def median_smooth(labels: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a compact majority smoother to binary labels."""
    if kernel_size <= 1 or len(labels) == 0:
        return labels

    radius = kernel_size // 2
    padded = np.pad(labels, (radius, radius), mode='edge')
    smoothed = np.zeros_like(labels)
    threshold = kernel_size // 2 + 1
    for idx in range(len(labels)):
        window = padded[idx:idx + kernel_size]
        smoothed[idx] = 1 if np.sum(window) >= threshold else 0
    return smoothed


def moving_average_smooth(values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a lightweight moving-average smoother to frame scores."""
    if kernel_size <= 1 or len(values) == 0:
        return values

    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    padded = np.pad(values, (kernel_size // 2, kernel_size - 1 - kernel_size // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid').astype(np.float32)


def compute_energy_zcr(x: np.ndarray, frame_len: int, hop_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute frame-level energy and zero-crossing rate."""
    frames = frame_signal(x, frame_len=frame_len, hop_len=hop_len)
    energy = np.mean(frames ** 2, axis=1)
    signs = np.sign(frames)
    signs[signs == 0.0] = -1.0
    crossings = signs[:, :-1] * signs[:, 1:] < 0
    zcr = np.mean(crossings, axis=1)
    return energy, zcr


def _safe_percentile_range(values: np.ndarray, low: float, high: float, eps: float = 1e-8) -> tuple[float, float]:
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi - lo < eps:
        hi = lo + eps
    return lo, hi


def _normalize_feature(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((values - low) / (high - low + 1e-8), 0.0, 1.0)


def _invert_normalize_feature(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return 1.0 - _normalize_feature(values, low, high)


class EnergyZCRVAD:
    """Adaptive VAD using energy, spectral cues, and temporal hysteresis."""

    def __init__(self, cfg: EnergyZCRVADConfig | None = None):
        self.cfg = cfg or EnergyZCRVADConfig()

    def _decode_labels(self, score: np.ndarray, energy_score: np.ndarray) -> np.ndarray:
        """Convert frame scores into binary speech labels with hysteresis."""
        labels = np.zeros(len(score), dtype=np.int32)
        state = 0
        hangover = 0

        for idx, current_score in enumerate(score):
            if state == 0:
                if (
                    current_score >= self.cfg.speech_start_threshold
                    and energy_score[idx] >= self.cfg.min_energy_score_for_speech
                ):
                    state = 1
                    hangover = self.cfg.hangover_frames
            else:
                if (
                    current_score < self.cfg.speech_end_threshold
                    or energy_score[idx] < self.cfg.min_energy_score_for_speech * 0.6
                ):
                    if hangover > 0:
                        hangover -= 1
                    else:
                        state = 0
                else:
                    hangover = self.cfg.hangover_frames

            labels[idx] = state

        labels = median_smooth(labels, self.cfg.median_filter_size)
        labels = remove_short_islands(labels, 1, self.cfg.min_speech_frames)
        labels = remove_short_islands(labels, 0, self.cfg.min_silence_frames)
        return labels.astype(np.int32)

    def compute_features(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Compute frame-level features used by the detector."""
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        x = x.astype(np.float32)

        frames = frame_signal(x, frame_len=self.cfg.frame_len, hop_len=self.cfg.hop_len)
        if len(frames) == 0:
            empty = np.zeros(0, dtype=np.float32)
            return {
                'energy': empty,
                'log_energy': empty,
                'zcr': empty,
                'band_ratio': empty,
                'flatness': empty,
            }

        window = np.hanning(self.cfg.frame_len).astype(np.float32)
        windowed_frames = frames * window[None, :]

        energy, zcr = compute_energy_zcr(x, self.cfg.frame_len, self.cfg.hop_len)
        log_energy = np.log10(energy + 1e-8)

        spec = np.abs(np.fft.rfft(windowed_frames, n=self.cfg.frame_len, axis=1)).astype(np.float32)
        freqs = np.fft.rfftfreq(self.cfg.frame_len, d=1.0 / config.SAMPLE_RATE)
        speech_band = (freqs >= self.cfg.speech_band_low_hz) & (freqs <= self.cfg.speech_band_high_hz)

        total_spec = np.sum(spec, axis=1) + 1e-8
        speech_spec = np.sum(spec[:, speech_band], axis=1)
        band_ratio = speech_spec / total_spec
        flatness = np.exp(np.mean(np.log(spec + 1e-8), axis=1)) / (np.mean(spec + 1e-8, axis=1))

        return {
            'energy': energy.astype(np.float32),
            'log_energy': log_energy.astype(np.float32),
            'zcr': zcr.astype(np.float32),
            'band_ratio': band_ratio.astype(np.float32),
            'flatness': flatness.astype(np.float32),
        }

    def compute_scores(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute fused speech score per frame."""
        features = self.compute_features(x)
        if len(features['energy']) == 0:
            return np.zeros(0, dtype=np.float32), features

        log_energy_low, log_energy_high = _safe_percentile_range(features['log_energy'], 10, 95)
        band_low, band_high = _safe_percentile_range(features['band_ratio'], 10, 90)
        zcr_low, _ = _safe_percentile_range(features['zcr'], 10, 95)
        flat_low, _ = _safe_percentile_range(features['flatness'], 5, 95)

        energy_score = _normalize_feature(features['log_energy'], log_energy_low, log_energy_high)
        band_ratio_score = _normalize_feature(features['band_ratio'], band_low, band_high)

        zcr_cap = max(self.cfg.zcr_max_speech, zcr_low + 1e-4)
        zcr_score = np.clip(_invert_normalize_feature(features['zcr'], zcr_low, zcr_cap), 0.0, 1.0)

        flatness_cap = max(self.cfg.flatness_max_speech, flat_low + 1e-4)
        flatness_score = np.clip(_invert_normalize_feature(features['flatness'], flat_low, flatness_cap), 0.0, 1.0)

        weights = np.array([
            self.cfg.energy_weight,
            self.cfg.band_ratio_weight,
            self.cfg.zcr_weight,
            self.cfg.flatness_weight,
        ], dtype=np.float32)
        weights /= np.sum(weights)

        score = (
            weights[0] * energy_score
            + weights[1] * band_ratio_score
            + weights[2] * zcr_score
            + weights[3] * flatness_score
        )

        energy_floor = np.clip(self.cfg.energy_thresh_ratio, 0.0, 1.0)
        score = np.where(energy_score >= energy_floor, score, score * 0.35)

        features.update({
            'energy_score': energy_score.astype(np.float32),
            'band_ratio_score': band_ratio_score.astype(np.float32),
            'zcr_score': zcr_score.astype(np.float32),
            'flatness_score': flatness_score.astype(np.float32),
        })
        return score.astype(np.float32), features

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return binary speech labels per frame."""
        score, features = self.compute_scores(x)
        if len(score) == 0:
            return np.zeros(0, dtype=np.int32)

        return self._decode_labels(score, features['energy_score'])

    def predict_soft(self, x: np.ndarray) -> np.ndarray:
        """Return soft speech confidence per frame."""
        score, features = self.compute_scores(x)
        if len(score) == 0:
            return np.zeros(0, dtype=np.float32)

        labels = self._decode_labels(score, features['energy_score']).astype(np.float32)
        soft = np.clip(score, 0.0, 1.0)
        soft = np.where(labels > 0.5, np.maximum(soft, self.cfg.speech_end_threshold), soft)
        soft = moving_average_smooth(soft, self.cfg.median_filter_size)
        return np.clip(soft, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def align_vad_to_frames(vad: np.ndarray, num_frames: int) -> np.ndarray:
        """Align VAD labels to match a specific number of frames."""
        if len(vad) == num_frames:
            return vad
        if len(vad) > num_frames:
            return vad[:num_frames]
        if len(vad) == 0:
            return np.zeros(num_frames, dtype=np.int32)
        out = np.zeros(num_frames, dtype=vad.dtype)
        out[: len(vad)] = vad
        out[len(vad):] = vad[-1]
        return out


def evaluate_vad_on_pair(
    x: np.ndarray,
    vad_true: np.ndarray,
    vad_model: EnergyZCRVAD | None = None
) -> dict:
    """Evaluate VAD on one audio file."""
    if vad_model is None:
        vad_model = EnergyZCRVAD()
    vad_pred = vad_model.predict(x)
    n = min(len(vad_pred), len(vad_true))
    vad_pred = vad_pred[:n]
    vad_true = vad_true[:n]
    return eval_vad_metrics(vad_true, vad_pred)
