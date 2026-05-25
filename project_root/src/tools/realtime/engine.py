"""Realtime audio engine for the speech enhancement demo."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from src.dsp.spectral_subtraction import enhance_waveform as spectral_subtraction_enhance
from src.dsp.vad_energy_zcr import EnergyZCRVAD
from src.dsp.wiener_filter import enhance_waveform as wiener_enhance
from src.models import enhance_audio_with_masknet, load_masknet_checkpoint
from src.tools.realtime.common import (
    LOGGER,
    SILENT_INPUT_RMS,
    SILENT_INPUT_WARNING_CALLBACKS,
    FileDemoSource,
    RecordingSession,
    StreamStats,
    block_rms,
    choose_audio_file,
    describe_device,
    normalize_audio_block,
    sd,
)
from src.tools.realtime.ui import LiveComparisonUI


class StreamingEnhancer:
    """Stateful enhancer that processes audio blocks with a rolling context buffer."""

    def __init__(
        self,
        method: str,
        device: torch.device,
        block_size: int,
        context_size: int,
        tail_trim_size: int,
        dry_wet: float = 1.0,
        checkpoint_path: Path | None = None,
    ):
        self.device = device
        self.block_size = block_size
        self.context_size = context_size
        self.tail_trim_size = tail_trim_size
        self.dry_wet = float(np.clip(dry_wet, 0.0, 1.0))
        self.checkpoint_path = checkpoint_path
        self.total_buffer_size = max(context_size + tail_trim_size + block_size, block_size * 4)
        self.input_buffer = np.zeros(self.total_buffer_size, dtype=np.float32)
        self.vad = EnergyZCRVAD()
        self.model = None
        self.method = "bypass"
        self.vad_mode = "adaptive_hard"
        self.set_method(method)

    def set_method(self, method: str) -> None:
        """Switch the realtime enhancement backend on the fly."""
        if method == "masknet" and self.model is None:
            if self.checkpoint_path is None:
                raise ValueError("checkpoint_path is required for masknet realtime demo")
            self.model = load_masknet_checkpoint(self.checkpoint_path, self.device)
        self.method = method

    def set_vad_mode(self, vad_mode: str) -> None:
        self.vad_mode = vad_mode

    def set_dry_wet(self, dry_wet: float) -> None:
        self.dry_wet = float(np.clip(dry_wet, 0.0, 1.0))

    def _compute_vad_labels(self, noisy_window: np.ndarray) -> np.ndarray | None:
        if self.vad_mode == "off":
            return None
        if self.vad_mode == "adaptive_soft":
            soft = self.vad.predict_soft(noisy_window)
            return (soft >= 0.5).astype(np.int32)
        return self.vad.predict(noisy_window)

    def warmup(self) -> None:
        """Run one synthetic pass to reduce first-block latency."""
        silence = np.zeros(self.total_buffer_size, dtype=np.float32)
        self._enhance_window(silence)

    def process_block(self, input_block: np.ndarray) -> np.ndarray:
        input_block = input_block.astype(np.float32, copy=False)
        if input_block.ndim != 1:
            input_block = input_block.reshape(-1)

        if len(input_block) != self.block_size:
            adjusted = np.zeros(self.block_size, dtype=np.float32)
            copy_len = min(len(input_block), self.block_size)
            adjusted[:copy_len] = input_block[:copy_len]
            input_block = adjusted

        self.input_buffer = np.roll(self.input_buffer, -self.block_size)
        self.input_buffer[-self.block_size:] = input_block

        enhanced_window = self._enhance_window(self.input_buffer)
        output_end = len(enhanced_window) - self.tail_trim_size
        output_start = max(0, output_end - self.block_size)
        output_block = enhanced_window[output_start:output_end]

        if len(output_block) < self.block_size:
            padded = np.zeros(self.block_size, dtype=np.float32)
            padded[-len(output_block):] = output_block
            output_block = padded

        mixed = self.dry_wet * output_block + (1.0 - self.dry_wet) * input_block
        return normalize_audio_block(mixed)

    def _enhance_window(self, noisy_window: np.ndarray) -> np.ndarray:
        if self.method == "masknet":
            enhanced = enhance_audio_with_masknet(noisy_window, self.model, self.device)
        elif self.method == "spectral_subtraction":
            vad_labels = self._compute_vad_labels(noisy_window)
            enhanced = spectral_subtraction_enhance(noisy_window, vad_labels=vad_labels, use_tracking=True)
        elif self.method == "wiener":
            vad_labels = self._compute_vad_labels(noisy_window)
            enhanced = wiener_enhance(noisy_window, vad_labels=vad_labels, use_tracking=True)
        elif self.method == "bypass":
            enhanced = noisy_window
        else:
            raise ValueError(f"Unsupported realtime method: {self.method}")

        enhanced = enhanced.astype(np.float32, copy=False)
        if len(enhanced) != len(noisy_window):
            aligned = np.zeros_like(noisy_window)
            copy_len = min(len(enhanced), len(noisy_window))
            aligned[:copy_len] = enhanced[:copy_len]
            enhanced = aligned
        return normalize_audio_block(enhanced)


class RealtimeDenoiseApp:
    """Microphone-to-speaker realtime denoising loop with a worker thread."""

    def __init__(
        self,
        enhancer: StreamingEnhancer,
        sample_rate: int,
        block_size: int,
        queue_size: int,
        input_device: int | str | None,
        output_device: int | str | None,
        latency: str,
        stats_interval: float,
        visualizer: LiveComparisonUI | None = None,
        recorder: RecordingSession | None = None,
        snapshot_dir: Path | None = None,
        file_source: FileDemoSource | None = None,
        output_gain_db: float = 0.0,
    ):
        self.enhancer = enhancer
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.input_device = input_device
        self.output_device = output_device
        self.latency = latency
        self.stats_interval = stats_interval
        self.visualizer = visualizer
        self.recorder = recorder
        self.snapshot_dir = snapshot_dir
        self.file_source = file_source
        self.output_gain_db = float(output_gain_db)

        self.stats = StreamStats()
        self.stop_event = threading.Event()
        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self.output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self.processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
        self.silent_input_callbacks = 0
        self.silent_input_warned = False

    def run(self) -> int:
        self.enhancer.warmup()
        self.processor_thread.start()

        LOGGER.info("Starting realtime denoise stream")
        LOGGER.info("Press Ctrl+C to stop")
        LOGGER.info(
            "Method=%s | SampleRate=%s Hz | Block=%s samples | Context=%s samples | Device=%s",
            self.enhancer.method,
            self.sample_rate,
            self.block_size,
            self.enhancer.context_size,
            self.enhancer.device,
        )
        LOGGER.info(
            "Audio devices | input=%s | output=%s",
            describe_device(self.input_device, kind="input"),
            describe_device(self.output_device, kind="output"),
        )
        last_good_input_device = self.input_device
        last_good_output_device = self.output_device

        try:
            while not self.stop_event.is_set():
                last_report = time.time()
                restart_stream = False
                device = (self.input_device, self.output_device)
                if self.visualizer is not None:
                    self.visualizer.set_active_devices(self.input_device, self.output_device)
                try:
                    with sd.Stream(
                        samplerate=self.sample_rate,
                        blocksize=self.block_size,
                        dtype="float32",
                        channels=1,
                        latency=self.latency,
                        device=device,
                        callback=self._audio_callback,
                    ):
                        last_good_input_device = self.input_device
                        last_good_output_device = self.output_device
                        if self.visualizer is not None:
                            self.visualizer.set_device_status_message("Audio stream running.")
                        while not self.stop_event.is_set():
                            time.sleep(0.2)
                            now = time.time()
                            if self.visualizer is not None:
                                requested_method = self.visualizer.consume_requested_method()
                                if requested_method is not None and requested_method != self.enhancer.method:
                                    LOGGER.info("Switching realtime method to %s", requested_method)
                                    self.enhancer.set_method(requested_method)
                                requested_vad_mode = self.visualizer.consume_requested_vad_mode()
                                if requested_vad_mode is not None:
                                    LOGGER.info("Switching VAD mode to %s", requested_vad_mode)
                                    self.enhancer.set_vad_mode(requested_vad_mode)
                                requested_source_mode = self.visualizer.consume_requested_source_mode()
                                if requested_source_mode is not None and self.file_source is not None:
                                    if requested_source_mode == "speech+noise" and not self.file_source.has_file_source_ready():
                                        LOGGER.warning("speech+noise source selected but no speech file is loaded yet")
                                    self.file_source.set_source_mode(requested_source_mode)
                                    LOGGER.info("Switching source mode to %s", requested_source_mode)
                                requested_scenario = self.visualizer.consume_requested_scenario()
                                if requested_scenario is not None and self.file_source is not None:
                                    distortion_map = self.file_source.apply_scenario_preset(requested_scenario)
                                    self.visualizer.sync_distortion_state(distortion_map)
                                    self.visualizer.set_mix_snr_db(self.file_source.mix_snr_db)
                                    LOGGER.info("Applied scenario preset: %s", requested_scenario)
                                requested_mix_snr_db = self.visualizer.consume_requested_mix_snr_db()
                                if requested_mix_snr_db is not None and self.file_source is not None:
                                    self.file_source.set_mix_snr_db(requested_mix_snr_db)
                                    LOGGER.info("Updated speech+noise SNR to %.1f dB", requested_mix_snr_db)
                                requested_dry_wet = self.visualizer.consume_requested_dry_wet()
                                if requested_dry_wet is not None:
                                    self.enhancer.set_dry_wet(requested_dry_wet)
                                    LOGGER.info("Updated dry/wet to %.2f", requested_dry_wet)
                                requested_output_gain_db = self.visualizer.consume_requested_output_gain_db()
                                if requested_output_gain_db is not None:
                                    self.output_gain_db = float(requested_output_gain_db)
                                    LOGGER.info("Updated output gain to %.1f dB", requested_output_gain_db)
                                distortion_state = self.visualizer.consume_distortion_state()
                                if distortion_state is not None and self.file_source is not None:
                                    self.file_source.set_distortions(distortion_state)
                                    LOGGER.info("Updated distortions: %s", distortion_state)
                                device_change = self.visualizer.consume_requested_device_change()
                                if device_change is not None:
                                    self.input_device, self.output_device = device_change
                                    self.silent_input_callbacks = 0
                                    self.silent_input_warned = False
                                    LOGGER.info(
                                        "Applying device change | input=%s | output=%s",
                                        describe_device(self.input_device, "input"),
                                        describe_device(self.output_device, "output"),
                                    )
                                    restart_stream = True
                                if self.visualizer.consume_speech_file_request() and self.file_source is not None:
                                    path = choose_audio_file("Select speech file")
                                    if path is not None:
                                        self.file_source.load_speech_file(path)
                                        status = self.file_source.get_status()
                                        self.visualizer.set_loaded_files(status["speech_file"], status["noise_file"])
                                        self.visualizer.set_mix_snr_db(float(status["mix_snr_db"]))
                                        LOGGER.info("Loaded speech file: %s", path)
                                if self.visualizer.consume_noise_file_request() and self.file_source is not None:
                                    path = choose_audio_file("Select noise file")
                                    if path is not None:
                                        self.file_source.load_noise_file(path)
                                        status = self.file_source.get_status()
                                        self.visualizer.set_loaded_files(status["speech_file"], status["noise_file"])
                                        self.visualizer.set_mix_snr_db(float(status["mix_snr_db"]))
                                        LOGGER.info("Loaded noise file: %s", path)
                                if self.recorder is not None and self.visualizer.consume_recording_toggle():
                                    enabled = self.recorder.toggle()
                                    self.visualizer.set_recording_state(enabled)
                                    LOGGER.info("Recording %s", "enabled" if enabled else "disabled")
                                if self.visualizer.consume_snapshot_request():
                                    snapshot_path = self.visualizer.save_snapshot(self.snapshot_dir)
                                    LOGGER.info("Saved UI snapshot: %s", snapshot_path)
                                self.visualizer.refresh(self.stats, self.enhancer.method)
                                if self.visualizer.closed or self.visualizer.stop_requested:
                                    LOGGER.info("Visualization window closed, stopping realtime demo")
                                    self.stop_event.set()
                                    break
                                if restart_stream:
                                    LOGGER.info("Restarting stream to apply selected audio devices")
                                    break
                            if now - last_report >= self.stats_interval:
                                self._log_stats()
                                last_report = now
                except Exception as exc:
                    error_message = str(exc).replace("\r", " ").replace("\n", " ")
                    LOGGER.error(
                        "Failed to start audio stream | input=%s | output=%s | error=%s",
                        describe_device(self.input_device, "input"),
                        describe_device(self.output_device, "output"),
                        error_message,
                    )
                    if self.visualizer is not None:
                        self.visualizer.set_device_status_message(
                            f"Invalid device pair. Reverted to last working pair. Error: {error_message[:120]}"
                        )
                    if (
                        self.input_device != last_good_input_device
                        or self.output_device != last_good_output_device
                    ):
                        self.input_device = last_good_input_device
                        self.output_device = last_good_output_device
                        restart_stream = True
                        LOGGER.info(
                            "Reverting to last working audio devices | input=%s | output=%s",
                            describe_device(self.input_device, "input"),
                            describe_device(self.output_device, "output"),
                        )
                        continue
                    else:
                        while not self.stop_event.is_set():
                            time.sleep(0.2)
                            if self.visualizer is None:
                                break
                            self.visualizer.set_active_devices(self.input_device, self.output_device)
                            requested_method = self.visualizer.consume_requested_method()
                            if requested_method is not None and requested_method != self.enhancer.method:
                                LOGGER.info("Switching realtime method to %s", requested_method)
                                self.enhancer.set_method(requested_method)
                            requested_vad_mode = self.visualizer.consume_requested_vad_mode()
                            if requested_vad_mode is not None:
                                LOGGER.info("Switching VAD mode to %s", requested_vad_mode)
                                self.enhancer.set_vad_mode(requested_vad_mode)
                            device_change = self.visualizer.consume_requested_device_change()
                            if device_change is not None:
                                self.input_device, self.output_device = device_change
                                self.silent_input_callbacks = 0
                                self.silent_input_warned = False
                                LOGGER.info(
                                    "Retrying with new audio devices | input=%s | output=%s",
                                    describe_device(self.input_device, "input"),
                                    describe_device(self.output_device, "output"),
                                )
                                restart_stream = True
                            self.visualizer.refresh(self.stats, self.enhancer.method)
                            if self.visualizer.closed or self.visualizer.stop_requested:
                                LOGGER.info("Visualization window closed, stopping realtime demo")
                                self.stop_event.set()
                                break
                            if restart_stream:
                                break
                if not restart_stream:
                    break
        except KeyboardInterrupt:
            LOGGER.info("Stopping realtime denoise stream")
        finally:
            self.stop_event.set()
            self.processor_thread.join(timeout=2.0)
            if self.visualizer is not None:
                self.visualizer.close()
            if self.recorder is not None:
                saved_paths = self.recorder.save()
                if saved_paths is not None:
                    raw_path, enhanced_path = saved_paths
                    LOGGER.info("Saved realtime recording: raw=%s enhanced=%s", raw_path, enhanced_path)
            self._log_stats()

        return 0

    def _audio_callback(self, indata, outdata, frames, _time_info, status) -> None:
        self.stats.callbacks += 1
        if status:
            self.stats.status_warnings += 1
            LOGGER.warning("Audio callback status: %s", status)

        input_block = np.copy(indata[:, 0])
        if self.file_source is not None:
            file_block = self.file_source.get_block(frames)
            if file_block is not None:
                input_block = file_block
        self._update_input_activity_warning(input_block)
        try:
            self.input_queue.put_nowait(input_block)
        except queue.Full:
            self.stats.input_overflows += 1
            try:
                _ = self.input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.input_queue.put_nowait(input_block)
            except queue.Full:
                pass

        try:
            output_block = self.output_queue.get_nowait()
        except queue.Empty:
            self.stats.output_underflows += 1
            output_block = np.zeros(frames, dtype=np.float32)

        if len(output_block) != frames:
            adjusted = np.zeros(frames, dtype=np.float32)
            copy_len = min(len(output_block), frames)
            adjusted[:copy_len] = output_block[:copy_len]
            output_block = adjusted

        if abs(self.output_gain_db) > 1e-6:
            gain = 10.0 ** (self.output_gain_db / 20.0)
            output_block = normalize_audio_block(output_block * gain)

        outdata[:, 0] = output_block
        if self.recorder is not None:
            self.recorder.add(input_block, output_block)
        if self.visualizer is not None:
            self.visualizer.push(input_block, output_block)

    def _update_input_activity_warning(self, input_block: np.ndarray) -> None:
        using_microphone = self.file_source is None or self.file_source.source_mode == "microphone"
        if not using_microphone:
            self.silent_input_callbacks = 0
            self.silent_input_warned = False
            return

        if block_rms(input_block) < SILENT_INPUT_RMS:
            self.silent_input_callbacks += 1
        else:
            self.silent_input_callbacks = 0
            self.silent_input_warned = False
            return

        if (
            self.silent_input_callbacks >= SILENT_INPUT_WARNING_CALLBACKS
            and not self.silent_input_warned
        ):
            self.silent_input_warned = True
            self.stats.silent_input_warnings += 1
            LOGGER.warning(
                "Microphone input appears nearly silent. Selected input device: %s. "
                "Try --list-devices and choose a specific mic with --input-device.",
                describe_device(self.input_device, kind="input"),
            )

    def _processor_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                block = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            start = time.perf_counter()
            enhanced = self.enhancer.process_block(block)
            elapsed = time.perf_counter() - start
            self.stats.update_processing_time(elapsed)

            try:
                self.output_queue.put_nowait(enhanced)
            except queue.Full:
                try:
                    _ = self.output_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.output_queue.put_nowait(enhanced)
                except queue.Full:
                    pass

    def _log_stats(self) -> None:
        mean_ms = self.stats.mean_processing_time * 1000.0
        max_ms = self.stats.max_processing_time * 1000.0
        LOGGER.info(
            "Stats | callbacks=%s processed=%s mean_proc=%.1f ms max_proc=%.1f ms in_drop=%s out_silence=%s warnings=%s silent_input=%s",
            self.stats.callbacks,
            self.stats.processed_blocks,
            mean_ms,
            max_ms,
            self.stats.input_overflows,
            self.stats.output_underflows,
            self.stats.status_warnings,
            self.stats.silent_input_warnings,
        )


