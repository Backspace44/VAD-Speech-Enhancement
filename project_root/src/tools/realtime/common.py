"""Shared helpers and device utilities for the realtime demo."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from src import config
from src.data_prep.augmentation import AudioAugmentation

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - handled at runtime
    sd = None


LOGGER = logging.getLogger("realtime_demo")
SILENT_INPUT_RMS = 1e-4
SILENT_INPUT_WARNING_CALLBACKS = 40


def resolve_default_checkpoint() -> Path:
    """Pick the best known local checkpoint for realtime enhancement."""
    candidates = [
        config.CHECKPOINTS_DIR / "voicebank_quality_final_checkpoint_v2" / "masknet_best.pth",
        config.CHECKPOINTS_DIR / "voicebank_quality_final_checkpoint_v1" / "masknet_best.pth",
        config.CHECKPOINTS_DIR / "masknet_best.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    latest_named = sorted(config.CHECKPOINTS_DIR.glob("**/masknet_best.pth"))
    if latest_named:
        return latest_named[-1]

    raise FileNotFoundError(
        "No checkpoint found. Provide --checkpoint or train/export a model first."
    )


def ms_to_samples(value_ms: float, sample_rate: int) -> int:
    return max(1, int(sample_rate * value_ms / 1000.0))


def normalize_audio_block(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    """Keep blocks bounded while preserving most of the original dynamics."""
    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    current_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current_peak > peak and current_peak > 1e-8:
        audio = audio * (peak / current_peak)
    return np.clip(audio, -1.0, 1.0)


def rms_dbfs(audio: np.ndarray) -> float:
    """Compute a simple RMS level in dBFS for UI display."""
    if audio.size == 0:
        return -80.0
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
    return 20.0 * np.log10(max(rms, 1e-6))


def block_rms(audio: np.ndarray) -> float:
    """Return linear RMS for silence heuristics."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def choose_audio_file(title: str) -> Path | None:
    """Open a file picker for WAV/FLAC/OGG/MP3 style audio sources."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        LOGGER.warning("File picker unavailable: %s", exc)
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Audio files", "*.wav *.flac *.ogg *.mp3 *.m4a"),
            ("WAV files", "*.wav"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        return None
    return Path(path)


@dataclass
class StreamStats:
    callbacks: int = 0
    processed_blocks: int = 0
    input_overflows: int = 0
    output_underflows: int = 0
    status_warnings: int = 0
    silent_input_warnings: int = 0
    processing_time_sum: float = 0.0
    max_processing_time: float = 0.0
    last_processing_time: float = 0.0

    def update_processing_time(self, elapsed: float) -> None:
        self.processed_blocks += 1
        self.processing_time_sum += elapsed
        self.max_processing_time = max(self.max_processing_time, elapsed)
        self.last_processing_time = elapsed

    @property
    def mean_processing_time(self) -> float:
        if self.processed_blocks == 0:
            return 0.0
        return self.processing_time_sum / self.processed_blocks


class RecordingSession:
    """Collect and persist raw/enhanced demo audio for later playback."""

    def __init__(self, sample_rate: int, output_dir: Path | None = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = output_dir or (config.AUDIO_SAMPLES_DIR / f"realtime_demo_{timestamp}")
        self.output_dir = Path(root)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.raw_blocks: list[np.ndarray] = []
        self.enhanced_blocks: list[np.ndarray] = []
        self.enabled = False

    def start(self) -> None:
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def add(self, raw_block: np.ndarray, enhanced_block: np.ndarray) -> None:
        if not self.enabled:
            return
        self.raw_blocks.append(np.copy(raw_block))
        self.enhanced_blocks.append(np.copy(enhanced_block))

    def save(self) -> tuple[Path, Path] | None:
        if not self.raw_blocks or not self.enhanced_blocks:
            return None
        raw_audio = np.concatenate(self.raw_blocks).astype(np.float32, copy=False)
        enhanced_audio = np.concatenate(self.enhanced_blocks).astype(np.float32, copy=False)
        raw_path = self.output_dir / "raw_microphone.wav"
        enhanced_path = self.output_dir / "enhanced_output.wav"
        sf.write(raw_path, raw_audio, self.sample_rate)
        sf.write(enhanced_path, enhanced_audio, self.sample_rate)
        return raw_path, enhanced_path


class FileDemoSource:
    """Optional speech+noise file source that can replace the microphone feed."""

    SCENARIO_PRESETS = {
        "neutral": {
            "snr_db": 10.0,
            "distortions": {},
        },
        "office": {
            "snr_db": 12.0,
            "distortions": {
                "random_eq": True,
                "dynamic_range_compression": True,
            },
        },
        "street": {
            "snr_db": 2.0,
            "distortions": {
                "random_gain": True,
                "apply_reverb": True,
                "add_clipping": True,
            },
        },
        "hard": {
            "snr_db": -2.0,
            "distortions": {
                "random_gain": True,
                "pitch_shift": True,
                "apply_reverb": True,
                "random_eq": True,
                "dynamic_range_compression": True,
                "add_clipping": True,
            },
        },
    }

    DISTORTION_ORDER = (
        "random_gain",
        "time_stretch",
        "pitch_shift",
        "apply_reverb",
        "random_eq",
        "dynamic_range_compression",
        "add_clipping",
    )

    def __init__(self, sample_rate: int, mix_snr_db: float = 5.0):
        self.sample_rate = sample_rate
        self.mix_snr_db = mix_snr_db
        self.augmenter = AudioAugmentation(config.AUGMENTATION_CONFIG, sample_rate=sample_rate)
        self.speech_path: Path | None = None
        self.noise_path: Path | None = None
        self.raw_speech: np.ndarray | None = None
        self.raw_noise: np.ndarray | None = None
        self.processed_speech: np.ndarray | None = None
        self.mixed_input: np.ndarray | None = None
        self.position = 0
        self.source_mode = "microphone"
        self.distortions = {name: False for name in self.DISTORTION_ORDER}
        self.lock = threading.Lock()

    def load_speech_file(self, path: Path) -> None:
        audio = self._load_audio(path)
        with self.lock:
            self.speech_path = path
            self.raw_speech = audio
            self.position = 0
            self._rebuild_mix()

    def load_noise_file(self, path: Path) -> None:
        audio = self._load_audio(path)
        with self.lock:
            self.noise_path = path
            self.raw_noise = audio
            self.position = 0
            self._rebuild_mix()

    def set_source_mode(self, mode: str) -> None:
        with self.lock:
            self.source_mode = mode
            self.position = 0

    def set_distortions(self, enabled_map: dict[str, bool]) -> None:
        with self.lock:
            for name in self.DISTORTION_ORDER:
                self.distortions[name] = bool(enabled_map.get(name, False))
            self.position = 0
            self._rebuild_mix()

    def set_mix_snr_db(self, snr_db: float) -> None:
        with self.lock:
            self.mix_snr_db = float(snr_db)
            self.position = 0
            self._rebuild_mix()

    def apply_scenario_preset(self, scenario_name: str) -> dict[str, bool]:
        preset = self.SCENARIO_PRESETS.get(scenario_name, self.SCENARIO_PRESETS["neutral"])
        distortion_map = {name: False for name in self.DISTORTION_ORDER}
        distortion_map.update(preset.get("distortions", {}))
        self.set_distortions(distortion_map)
        self.set_mix_snr_db(preset.get("snr_db", self.mix_snr_db))
        return distortion_map

    def get_block(self, frames: int) -> np.ndarray | None:
        with self.lock:
            if self.source_mode != "speech+noise":
                return None
            if self.mixed_input is None or len(self.mixed_input) == 0:
                return None

            end = self.position + frames
            if end <= len(self.mixed_input):
                block = self.mixed_input[self.position:end]
                self.position = end % len(self.mixed_input)
                return block.astype(np.float32, copy=True)

            first = self.mixed_input[self.position:]
            remaining = frames - len(first)
            reps = int(np.ceil(remaining / len(self.mixed_input)))
            tail = np.tile(self.mixed_input, reps)[:remaining]
            self.position = remaining % len(self.mixed_input)
            return np.concatenate([first, tail]).astype(np.float32, copy=False)

    def has_file_source_ready(self) -> bool:
        with self.lock:
            return self.mixed_input is not None and len(self.mixed_input) > 0

    def get_status(self) -> dict[str, str]:
        with self.lock:
            return {
                "source_mode": self.source_mode,
                "speech_file": self.speech_path.name if self.speech_path else "None",
                "noise_file": self.noise_path.name if self.noise_path else "None",
                "mix_snr_db": f"{self.mix_snr_db:.1f}",
            }

    def _load_audio(self, path: Path) -> np.ndarray:
        audio, sample_rate = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate != self.sample_rate:
            audio = self._resample_audio(audio, sample_rate, self.sample_rate)
        return normalize_audio_block(audio.astype(np.float32, copy=False))

    def _resample_audio(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        common = gcd(from_sr, to_sr)
        up = to_sr // common
        down = from_sr // common
        return resample_poly(audio, up, down).astype(np.float32, copy=False)

    def _apply_selected_distortions(self, audio: np.ndarray) -> np.ndarray:
        processed = np.copy(audio).astype(np.float32, copy=False)
        for name in self.DISTORTION_ORDER:
            if not self.distortions.get(name, False):
                continue
            cfg = config.AUGMENTATION_CONFIG.get(name, {})
            if name == "random_gain":
                processed = self.augmenter.random_gain(processed, cfg)
            elif name == "time_stretch":
                processed = self.augmenter.time_stretch(processed, cfg)
            elif name == "pitch_shift":
                processed = self.augmenter.pitch_shift(processed, cfg)
            elif name == "apply_reverb":
                processed = self.augmenter.apply_reverb(processed, cfg)
            elif name == "random_eq":
                processed = self.augmenter.random_eq(processed, cfg)
            elif name == "dynamic_range_compression":
                processed = self.augmenter.dynamic_range_compression(processed, cfg)
            elif name == "add_clipping":
                processed = self.augmenter.add_clipping(processed, cfg)
        return normalize_audio_block(processed)

    def _match_length(self, noise: np.ndarray, target_len: int) -> np.ndarray:
        if len(noise) == target_len:
            return noise
        if len(noise) > target_len:
            return noise[:target_len]
        reps = int(np.ceil(target_len / len(noise)))
        return np.tile(noise, reps)[:target_len]

    def _mix_at_snr(self, clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        eps = 1e-8
        clean = clean.astype(np.float32, copy=False)
        noise = noise.astype(np.float32, copy=False)
        p_clean = np.mean(clean ** 2) + eps
        p_noise = np.mean(noise ** 2) + eps
        target_ratio = 10.0 ** (-snr_db / 10.0)
        scale = np.sqrt(target_ratio * p_clean / p_noise)
        return normalize_audio_block(clean + scale * noise)

    def _rebuild_mix(self) -> None:
        if self.raw_speech is None:
            self.processed_speech = None
            self.mixed_input = None
            return

        self.processed_speech = self._apply_selected_distortions(self.raw_speech)
        if self.raw_noise is None:
            self.mixed_input = np.copy(self.processed_speech)
            return

        matched_noise = self._match_length(self.raw_noise, len(self.processed_speech))
        self.mixed_input = self._mix_at_snr(self.processed_speech, matched_noise, self.mix_snr_db)


def print_devices() -> None:
    if sd is None:
        raise ImportError("sounddevice is not installed")

    devices = sd.query_devices()
    LOGGER.info("Available audio devices:")
    for index, device in enumerate(devices):
        LOGGER.info(
            "[%s] %s | in=%s out=%s default_sr=%s",
            index,
            device["name"],
            device["max_input_channels"],
            device["max_output_channels"],
            int(device["default_samplerate"]),
        )


def _device_keyword_bonus(name: str, positive: tuple[str, ...], negative: tuple[str, ...]) -> float:
    score = 0.0
    for token in positive:
        if token in name:
            score += 6.0
    for token in negative:
        if token in name:
            score -= 12.0
    return score


def _score_device_candidate(index: int, device: dict, *, kind: str, sample_rate: int) -> float:
    if kind == "input":
        positive = ("headset", "microphone", "mic", "array", "realtek", "ath", "usb", "bluetooth")
        negative = ("steam streaming", "mapper", "stereo mix", "line in", "virtual")
        default_index = sd.default.device[0]
    else:
        positive = ("headset", "speaker", "speakers", "headphones", "realtek", "ath", "usb", "bluetooth")
        negative = ("steam streaming", "mapper", "digital output", "virtual")
        default_index = sd.default.device[1]

    name = str(device["name"]).lower()
    score = _device_keyword_bonus(name, positive, negative)
    if sample_rate:
        default_sr = int(device["default_samplerate"])
        if default_sr == sample_rate:
            score += 4.0
        elif abs(default_sr - sample_rate) <= 2000:
            score += 2.0
    if index == default_index:
        score += 3.0
    if "mme" in name:
        score -= 1.0
    if "windows directsound" in name:
        score -= 0.5
    return score


def pick_best_device(kind: str, sample_rate: int) -> int | None:
    if sd is None:
        return None

    devices = sd.query_devices()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"

    best_index: int | None = None
    best_score = float("-inf")
    for index, device in enumerate(devices):
        if int(device[channel_key]) <= 0:
            continue
        score = _score_device_candidate(index, device, kind=kind, sample_rate=sample_rate)
        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def resolve_device(requested: str | None, kind: str, sample_rate: int) -> int | str | None:
    if requested is None:
        return pick_best_device(kind=kind, sample_rate=sample_rate)
    requested = str(requested).strip()
    if requested.isdigit():
        return int(requested)
    return requested


def pick_best_device_pair(
    requested_input: str | None,
    requested_output: str | None,
    sample_rate: int,
) -> tuple[int | str | None, int | str | None]:
    input_device = resolve_device(requested_input, kind="input", sample_rate=sample_rate)
    output_device = resolve_device(requested_output, kind="output", sample_rate=sample_rate)

    if sd is None:
        return input_device, output_device
    if requested_input is not None or requested_output is not None:
        return input_device, output_device
    if not isinstance(input_device, int) or not isinstance(output_device, int):
        return input_device, output_device

    try:
        input_info = sd.query_devices(input_device)
        output_info = sd.query_devices(output_device)
    except Exception:
        return input_device, output_device

    if input_info["hostapi"] == output_info["hostapi"]:
        return input_device, output_device

    devices = sd.query_devices()
    best_pair = (input_device, output_device)
    best_score = float("-inf")
    for in_index, in_device in enumerate(devices):
        if int(in_device["max_input_channels"]) <= 0:
            continue
        for out_index, out_device in enumerate(devices):
            if int(out_device["max_output_channels"]) <= 0:
                continue
            if in_device["hostapi"] != out_device["hostapi"]:
                continue

            pair_score = _score_device_candidate(
                in_index,
                in_device,
                kind="input",
                sample_rate=sample_rate,
            )
            pair_score += _score_device_candidate(
                out_index,
                out_device,
                kind="output",
                sample_rate=sample_rate,
            )
            if str(in_device["name"]).lower() == str(out_device["name"]).lower():
                pair_score += 8.0
            elif "headset" in str(in_device["name"]).lower() and "headset" in str(out_device["name"]).lower():
                pair_score += 5.0
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (in_index, out_index)

    return best_pair


def describe_device(device_ref: int | str | None, kind: str) -> str:
    if device_ref is None:
        return f"auto-{kind}"
    if sd is None:
        return str(device_ref)
    try:
        device = sd.query_devices(device_ref, kind=kind)
    except Exception:
        return str(device_ref)
    return f"{device_ref}: {device['name']}"


def list_device_candidates(kind: str, sample_rate: int, limit: int = 6) -> list[tuple[str, int]]:
    if sd is None:
        return []

    devices = sd.query_devices()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    scored: list[tuple[float, str, int]] = []
    for index, device in enumerate(devices):
        if int(device[channel_key]) <= 0:
            continue
        score = _score_device_candidate(index, device, kind=kind, sample_rate=sample_rate)
        label = f"{index} | {str(device['name'])[:36]}"
        scored.append((score, label, index))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(label, index) for _, label, index in scored[:limit]]
