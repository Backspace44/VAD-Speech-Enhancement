"""Tkinter dashboard for the realtime raw-vs-enhanced comparison view."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src import config
from src.dsp.vad_energy_zcr import EnergyZCRVAD
from src.tools.realtime.common import rms_dbfs


class LiveComparisonUI:
    """Tkinter desktop dashboard for realtime raw vs enhanced monitoring."""

    def __init__(
        self,
        sample_rate: int,
        display_samples: int,
        initial_method: str,
        recording_enabled: bool = False,
        checkpoint_label: str | None = None,
        fullscreen: bool = False,
        initial_vad_mode: str = "adaptive_hard",
        initial_source_mode: str = "microphone",
        initial_distortions: dict[str, bool] | None = None,
        initial_mix_snr_db: float = 5.0,
        initial_dry_wet: float = 1.0,
        initial_output_gain_db: float = 0.0,
        initial_scenario: str = "neutral",
        input_device_options: list[tuple[str, int]] | None = None,
        output_device_options: list[tuple[str, int]] | None = None,
        initial_input_device: int | str | None = None,
        initial_output_device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.display_samples = display_samples
        self.raw_buffer = np.zeros(display_samples, dtype=np.float32)
        self.output_buffer = np.zeros(display_samples, dtype=np.float32)
        self.raw_rms_db = -80.0
        self.output_rms_db = -80.0
        self.current_method = initial_method
        self.pending_method = initial_method
        self.current_vad_mode = initial_vad_mode
        self.pending_vad_mode = initial_vad_mode
        self.current_source_mode = initial_source_mode
        self.pending_source_mode = initial_source_mode
        self.current_scenario = initial_scenario
        self.pending_scenario = initial_scenario
        self.stop_requested = False
        self.recording_enabled = recording_enabled
        self.pending_recording_toggle = False
        self.pending_snapshot = False
        self.checkpoint_label = checkpoint_label or "N/A"
        self.fullscreen = fullscreen
        distortion_defaults = initial_distortions or {}
        self.distortion_labels = [
            ("random_gain", "Gain"),
            ("time_stretch", "Time Stretch"),
            ("pitch_shift", "Pitch Shift"),
            ("apply_reverb", "Reverb"),
            ("random_eq", "EQ"),
            ("dynamic_range_compression", "Compress"),
            ("add_clipping", "Clip"),
        ]
        self.current_distortions = {name: bool(distortion_defaults.get(name, False)) for name, _ in self.distortion_labels}
        self.pending_distortions = dict(self.current_distortions)
        self.pending_distortion_sync = False
        self.pending_speech_file_request = False
        self.pending_noise_file_request = False
        self.current_mix_snr_db = float(initial_mix_snr_db)
        self.pending_mix_snr_db = float(initial_mix_snr_db)
        self.pending_mix_snr_sync = False
        self.current_dry_wet = float(initial_dry_wet)
        self.pending_dry_wet = float(initial_dry_wet)
        self.pending_dry_wet_sync = False
        self.current_output_gain_db = float(initial_output_gain_db)
        self.pending_output_gain_db = float(initial_output_gain_db)
        self.pending_output_gain_sync = False
        self.loaded_speech_file = "None"
        self.loaded_noise_file = "None"
        self.input_device_options = input_device_options or []
        self.output_device_options = output_device_options or []
        self.current_input_device = initial_input_device
        self.current_output_device = initial_output_device
        self.pending_input_device = initial_input_device
        self.pending_output_device = initial_output_device
        self.pending_device_change = False
        self.last_signal_present = False
        self.device_status_message = "Audio stream ready."
        self.lock = threading.Lock()
        self.closed = False
        self.last_refresh_time = 0.0

        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("Realtime Speech Enhancement Monitor")
        self.root.configure(bg="#f6f3ee")
        self.root.geometry("1500x980")
        if fullscreen:
            try:
                self.root.state("zoomed")
            except Exception:
                pass
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Card.TFrame", background="#fffdf8")
        style.configure("Panel.TFrame", background="#fbf7f0")
        style.configure("Section.TLabelframe", background="#fffdf8")
        style.configure("Section.TLabelframe.Label", background="#fffdf8", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", background="#fffdf8", font=("Segoe UI", 10), foreground="#31424f")
        style.configure("Muted.TLabel", background="#fffdf8", font=("Segoe UI", 9), foreground="#736b63")
        style.configure("Header.TLabel", background="#f6f3ee", font=("Segoe UI", 16, "bold"), foreground="#22313a")
        style.configure("SubHeader.TLabel", background="#fbf7f0", font=("Segoe UI", 11, "bold"), foreground="#22313a")
        style.configure("App.TNotebook", background="#fffdf8", borderwidth=0)
        style.configure("App.TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(14, 8), background="#efe7da", foreground="#4a4138")
        style.map("App.TNotebook.Tab", background=[("selected", "#d9e7f5")], foreground=[("selected", "#22313a")])
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 8))
        style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 8))
        style.configure("Slim.Horizontal.TProgressbar", troughcolor="#ebe3d8", background="#c44e52", thickness=10, borderwidth=0)
        style.configure("Green.Horizontal.TProgressbar", troughcolor="#ebe3d8", background="#55a868", thickness=10, borderwidth=0)

        main = ttk.Frame(self.root, padding=14)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(1, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(header, text="Realtime Speech Enhancement Monitor", style="Header.TLabel").pack(side="left")
        self.signal_var = tk.StringVar(value="NO MIC SIGNAL")
        self.signal_label = ttk.Label(header, textvariable=self.signal_var, style="Header.TLabel", foreground="#a94442")
        self.signal_label.pack(side="right")

        left = ttk.Frame(main, style="Card.TFrame", padding=12)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        status_frame = ttk.Frame(left, style="Panel.TFrame", padding=10)
        status_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="method=idle")
        self.summary_var = tk.StringVar(value="stream_status=Audio stream ready.")
        self.files_var = tk.StringVar(value="speech_file=None | noise_file=None")
        ttk.Label(status_frame, text="Session Summary", style="SubHeader.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel", wraplength=880, justify="left").grid(row=1, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.summary_var, style="Status.TLabel", wraplength=880, justify="left").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(status_frame, textvariable=self.files_var, style="Muted.TLabel", wraplength=880, justify="left").grid(row=3, column=0, sticky="w", pady=(6, 0))

        fig = Figure(figsize=(10, 8), facecolor="#fffdf8")
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.22)
        self.fig = fig
        self.ax_raw = fig.add_subplot(gs[0, :])
        self.ax_out = fig.add_subplot(gs[1, :], sharex=self.ax_raw)
        self.ax_raw_spec = fig.add_subplot(gs[2, 0])
        self.ax_out_spec = fig.add_subplot(gs[2, 1], sharex=self.ax_raw_spec, sharey=self.ax_raw_spec)
        for ax in (self.ax_raw, self.ax_out, self.ax_raw_spec, self.ax_out_spec):
            ax.set_facecolor("#fffdf8")
            ax.grid(True, alpha=0.18)
        time_axis = np.linspace(-display_samples / sample_rate, 0.0, display_samples, endpoint=False)
        self.raw_line, = self.ax_raw.plot(time_axis, self.raw_buffer, color="#c44e52", linewidth=1.1)
        self.out_line, = self.ax_out.plot(time_axis, self.output_buffer, color="#55a868", linewidth=1.1)
        self.ax_raw.set_title("Raw Input")
        self.ax_out.set_title("Enhanced Output")
        self.ax_raw.set_ylim(-1.05, 1.05)
        self.ax_out.set_ylim(-1.05, 1.05)
        self.ax_out.set_xlabel("Time (s)")
        self.ax_raw_spec.set_title("Raw Spectrogram")
        self.ax_out_spec.set_title("Enhanced Spectrogram")
        self.ax_raw_spec.set_ylabel("Frequency (Hz)")
        self.ax_raw_spec.set_xlabel("Time (s)")
        self.ax_out_spec.set_xlabel("Time (s)")
        self.raw_spec_image = self.ax_raw_spec.imshow(
            np.zeros((config.N_FREQ_BINS, 32), dtype=np.float32),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[-1.0, 0.0, 0.0, self.sample_rate / 2.0],
            vmin=-80.0,
            vmax=0.0,
            cmap="magma",
        )
        self.out_spec_image = self.ax_out_spec.imshow(
            np.zeros((config.N_FREQ_BINS, 32), dtype=np.float32),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[-1.0, 0.0, 0.0, self.sample_rate / 2.0],
            vmin=-80.0,
            vmax=0.0,
            cmap="viridis",
        )
        self.canvas = FigureCanvasTkAgg(fig, master=left)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew")

        self.vad_fig = Figure(figsize=(4.2, 1.6), facecolor="#fffdf8")
        self.vad_ax = self.vad_fig.add_subplot(111)
        self.vad_ax.set_facecolor("#fffdf8")
        self.vad_ax.set_title("Live VAD Confidence")
        self.vad_ax.set_xlim(-display_samples / sample_rate, 0.0)
        self.vad_ax.set_ylim(0.0, 1.05)
        self.vad_ax.set_xlabel("Time (s)")
        self.vad_ax.set_ylabel("Speech")
        self.vad_ax.grid(True, alpha=0.18)
        self.vad_line, = self.vad_ax.plot(
            np.linspace(-display_samples / sample_rate, 0.0, 32, endpoint=False),
            np.zeros(32, dtype=np.float32),
            color="#4c72b0",
            linewidth=1.5,
        )

        right = ttk.Frame(main, style="Card.TFrame", padding=10)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=0)
        right.columnconfigure(0, weight=1)

        level_frame = ttk.LabelFrame(right, text="Levels", style="Section.TLabelframe", padding=10)
        level_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        level_frame.columnconfigure(1, weight=1)
        level_frame.columnconfigure(2, weight=0)
        ttk.Label(level_frame, text="Input RMS", style="Status.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(level_frame, text="Output RMS", style="Status.TLabel").grid(row=1, column=0, sticky="w")
        self.input_db_var = tk.StringVar(value="-80.0 dBFS")
        self.output_db_var = tk.StringVar(value="-80.0 dBFS")
        self.input_level_var = tk.DoubleVar(value=0.0)
        self.output_level_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(level_frame, maximum=60.0, variable=self.input_level_var, style="Slim.Horizontal.TProgressbar").grid(row=0, column=1, sticky="ew", padx=(10, 8))
        ttk.Progressbar(level_frame, maximum=60.0, variable=self.output_level_var, style="Green.Horizontal.TProgressbar").grid(row=1, column=1, sticky="ew", padx=(10, 8))
        ttk.Label(level_frame, textvariable=self.input_db_var, style="Status.TLabel").grid(row=0, column=2, sticky="e")
        ttk.Label(level_frame, textvariable=self.output_db_var, style="Status.TLabel").grid(row=1, column=2, sticky="e")

        notebook = ttk.Notebook(right, style="App.TNotebook")
        notebook.grid(row=1, column=0, sticky="nsew")
        self.notebook = notebook

        vad_panel = ttk.LabelFrame(right, text="Speech Activity", style="Section.TLabelframe", padding=8)
        vad_panel.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.vad_canvas = FigureCanvasTkAgg(self.vad_fig, master=vad_panel)
        self.vad_canvas_widget = self.vad_canvas.get_tk_widget()
        self.vad_canvas_widget.pack(fill="both", expand=True)

        audio_tab = ttk.Frame(notebook, padding=10)
        processing_tab = ttk.Frame(notebook, padding=10)
        files_tab = ttk.Frame(notebook, padding=10)
        augment_tab = ttk.Frame(notebook, padding=10)
        notebook.add(audio_tab, text="Audio")
        notebook.add(processing_tab, text="Processing")
        notebook.add(files_tab, text="Files")
        notebook.add(augment_tab, text="Augment")

        self.source_var = tk.StringVar(value=initial_source_mode)
        self.input_device_var = tk.StringVar(value=self._find_device_label(self.input_device_options, initial_input_device))
        self.output_device_var = tk.StringVar(value=self._find_device_label(self.output_device_options, initial_output_device))
        self.method_var = tk.StringVar(value=initial_method)
        self.vad_var = tk.StringVar(value=initial_vad_mode)
        self.scenario_var = tk.StringVar(value=initial_scenario)
        self.mix_snr_var = tk.DoubleVar(value=float(initial_mix_snr_db))
        self.dry_wet_var = tk.DoubleVar(value=float(initial_dry_wet))
        self.output_gain_var = tk.DoubleVar(value=float(initial_output_gain_db))
        self.record_var = tk.StringVar(value="Stop Recording" if recording_enabled else "Start Recording")

        ttk.Label(audio_tab, text="Choose the active live source", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(audio_tab, text="Source").grid(row=1, column=0, sticky="w")
        ttk.Combobox(audio_tab, textvariable=self.source_var, values=("microphone", "speech+noise"), state="readonly", height=6).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(audio_tab, text="Mic Input").grid(row=3, column=0, sticky="w")
        ttk.Combobox(audio_tab, textvariable=self.input_device_var, values=[label for label, _ in self.input_device_options], state="readonly", height=8).grid(row=4, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(audio_tab, text="Speaker Output").grid(row=5, column=0, sticky="w")
        ttk.Combobox(audio_tab, textvariable=self.output_device_var, values=[label for label, _ in self.output_device_options], state="readonly", height=8).grid(row=6, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(audio_tab, text="Apply Audio Devices", command=self._on_apply_devices_clicked, style="Primary.TButton").grid(row=7, column=0, sticky="ew")
        audio_tab.columnconfigure(0, weight=1)

        ttk.Label(processing_tab, text="Choose denoising and speech detection behavior", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(processing_tab, text="Enhancement").grid(row=1, column=0, sticky="w")
        ttk.Combobox(processing_tab, textvariable=self.method_var, values=("masknet", "spectral_subtraction", "wiener", "bypass"), state="readonly", height=6).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(processing_tab, text="Speech Detect").grid(row=3, column=0, sticky="w")
        ttk.Combobox(processing_tab, textvariable=self.vad_var, values=("off", "adaptive_hard", "adaptive_soft"), state="readonly", height=6).grid(row=4, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(processing_tab, text="Dry/Wet").grid(row=5, column=0, sticky="w")
        self.dry_slider = ttk.Scale(processing_tab, from_=0.0, to=1.0, variable=self.dry_wet_var, command=self._on_dry_wet_changed)
        self.dry_slider.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(processing_tab, text="Output Gain").grid(row=7, column=0, sticky="w")
        self.gain_slider = ttk.Scale(processing_tab, from_=-12.0, to=12.0, variable=self.output_gain_var, command=self._on_output_gain_changed)
        self.gain_slider.grid(row=8, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(processing_tab, text="Snapshot", command=self._on_snapshot_clicked).grid(row=9, column=0, sticky="ew", pady=(0, 6))
        self.record_button = ttk.Button(processing_tab, textvariable=self.record_var, command=self._on_record_clicked, style="Primary.TButton")
        self.record_button.grid(row=10, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(processing_tab, text="Stop Demo", command=self._on_stop_clicked, style="Danger.TButton").grid(row=11, column=0, sticky="ew")
        processing_tab.columnconfigure(0, weight=1)

        ttk.Label(files_tab, text="Load local files for synthetic speech+noise demos", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Button(files_tab, text="Choose Speech File", command=self._on_speech_file_clicked, style="Primary.TButton").grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(files_tab, text="Choose Noise File", command=self._on_noise_file_clicked, style="Primary.TButton").grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.speech_file_var = tk.StringVar(value="Speech: None")
        self.noise_file_var = tk.StringVar(value="Noise: None")
        ttk.Label(files_tab, textvariable=self.speech_file_var, style="Status.TLabel", wraplength=340, justify="left").grid(row=3, column=0, sticky="w", pady=(6, 4))
        ttk.Label(files_tab, textvariable=self.noise_file_var, style="Status.TLabel", wraplength=340, justify="left").grid(row=4, column=0, sticky="w")
        files_tab.columnconfigure(0, weight=1)

        ttk.Label(augment_tab, text="Configure synthetic scene difficulty and artifacts", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(augment_tab, text="Scenario").grid(row=1, column=0, sticky="w")
        ttk.Combobox(augment_tab, textvariable=self.scenario_var, values=("neutral", "office", "street", "hard"), state="readonly", height=6).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(augment_tab, text="Mix SNR").grid(row=3, column=0, sticky="w")
        self.snr_slider = ttk.Scale(augment_tab, from_=-5.0, to=30.0, variable=self.mix_snr_var, command=self._on_snr_changed)
        self.snr_slider.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        distort_frame = ttk.LabelFrame(augment_tab, text="Distortions", style="Section.TLabelframe", padding=8)
        distort_frame.grid(row=5, column=0, sticky="nsew")
        self.distortion_vars: dict[str, tk.BooleanVar] = {}
        for idx, (name, label) in enumerate(self.distortion_labels):
            var = tk.BooleanVar(value=self.current_distortions[name])
            self.distortion_vars[name] = var
            ttk.Checkbutton(distort_frame, text=label, variable=var, command=self._on_distortion_toggled).grid(row=idx, column=0, sticky="w")
        augment_tab.columnconfigure(0, weight=1)

        self.source_var.trace_add("write", self._on_source_var_changed)
        self.input_device_var.trace_add("write", self._on_input_device_var_changed)
        self.output_device_var.trace_add("write", self._on_output_device_var_changed)
        self.method_var.trace_add("write", self._on_method_var_changed)
        self.vad_var.trace_add("write", self._on_vad_var_changed)
        self.scenario_var.trace_add("write", self._on_scenario_var_changed)

        self.root.update_idletasks()

    def _find_device_label(self, options: list[tuple[str, int]], device_id: int | str | None) -> str:
        for label, candidate_id in options:
            if candidate_id == device_id:
                return label
        return options[0][0] if options else "No device"

    def _on_close(self, _event=None) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _on_stop_clicked(self, _event=None) -> None:
        self.stop_requested = True

    def _on_record_clicked(self, _event=None) -> None:
        self.pending_recording_toggle = True

    def _on_snapshot_clicked(self, _event=None) -> None:
        self.pending_snapshot = True

    def _on_apply_devices_clicked(self, _event=None) -> None:
        self.pending_device_change = True

    def _on_method_var_changed(self, *_args) -> None:
        self.pending_method = self.method_var.get()

    def _on_source_var_changed(self, *_args) -> None:
        self.pending_source_mode = self.source_var.get()

    def _on_input_device_var_changed(self, *_args) -> None:
        label = self.input_device_var.get()
        for option_label, option_id in self.input_device_options:
            if option_label == label:
                self.pending_input_device = option_id
                break

    def _on_output_device_var_changed(self, *_args) -> None:
        label = self.output_device_var.get()
        for option_label, option_id in self.output_device_options:
            if option_label == label:
                self.pending_output_device = option_id
                break

    def _on_vad_var_changed(self, *_args) -> None:
        self.pending_vad_mode = self.vad_var.get()

    def _on_scenario_var_changed(self, *_args) -> None:
        self.pending_scenario = self.scenario_var.get()

    def _on_snr_changed(self, value) -> None:
        self.pending_mix_snr_db = float(value)
        self.pending_mix_snr_sync = True

    def _on_dry_wet_changed(self, value) -> None:
        self.pending_dry_wet = float(value)
        self.pending_dry_wet_sync = True

    def _on_output_gain_changed(self, value) -> None:
        self.pending_output_gain_db = float(value)
        self.pending_output_gain_sync = True

    def _on_speech_file_clicked(self, _event=None) -> None:
        self.pending_speech_file_request = True

    def _on_noise_file_clicked(self, _event=None) -> None:
        self.pending_noise_file_request = True

    def _on_distortion_toggled(self, _label: str | None = None) -> None:
        self.pending_distortions = {
            name: bool(self.distortion_vars[name].get()) for name, _ in self.distortion_labels
        }
        self.pending_distortion_sync = True

    def push(self, raw_block: np.ndarray, output_block: np.ndarray) -> None:
        with self.lock:
            self.raw_buffer = np.roll(self.raw_buffer, -len(raw_block))
            self.raw_buffer[-len(raw_block):] = raw_block
            self.output_buffer = np.roll(self.output_buffer, -len(output_block))
            self.output_buffer[-len(output_block):] = output_block
            self.raw_rms_db = rms_dbfs(raw_block)
            self.output_rms_db = rms_dbfs(output_block)

    def consume_requested_method(self) -> str | None:
        if self.pending_method != self.current_method:
            self.current_method = self.pending_method
            return self.current_method
        return None

    def consume_requested_source_mode(self) -> str | None:
        if self.pending_source_mode != self.current_source_mode:
            self.current_source_mode = self.pending_source_mode
            return self.current_source_mode
        return None

    def consume_requested_device_change(self) -> tuple[int | str | None, int | str | None] | None:
        if self.pending_device_change:
            self.pending_device_change = False
            self.current_input_device = self.pending_input_device
            self.current_output_device = self.pending_output_device
            return self.current_input_device, self.current_output_device
        return None

    def consume_requested_vad_mode(self) -> str | None:
        if self.pending_vad_mode != self.current_vad_mode:
            self.current_vad_mode = self.pending_vad_mode
            return self.current_vad_mode
        return None

    def consume_requested_scenario(self) -> str | None:
        if self.pending_scenario != self.current_scenario:
            self.current_scenario = self.pending_scenario
            return self.current_scenario
        return None

    def consume_requested_mix_snr_db(self) -> float | None:
        if self.pending_mix_snr_sync:
            self.pending_mix_snr_sync = False
            self.current_mix_snr_db = self.pending_mix_snr_db
            return self.current_mix_snr_db
        return None

    def consume_requested_dry_wet(self) -> float | None:
        if self.pending_dry_wet_sync:
            self.pending_dry_wet_sync = False
            self.current_dry_wet = self.pending_dry_wet
            return self.current_dry_wet
        return None

    def consume_requested_output_gain_db(self) -> float | None:
        if self.pending_output_gain_sync:
            self.pending_output_gain_sync = False
            self.current_output_gain_db = self.pending_output_gain_db
            return self.current_output_gain_db
        return None

    def consume_recording_toggle(self) -> bool:
        if self.pending_recording_toggle:
            self.pending_recording_toggle = False
            return True
        return False

    def consume_snapshot_request(self) -> bool:
        if self.pending_snapshot:
            self.pending_snapshot = False
            return True
        return False

    def consume_speech_file_request(self) -> bool:
        if self.pending_speech_file_request:
            self.pending_speech_file_request = False
            return True
        return False

    def consume_noise_file_request(self) -> bool:
        if self.pending_noise_file_request:
            self.pending_noise_file_request = False
            return True
        return False

    def consume_distortion_state(self) -> dict[str, bool] | None:
        if self.pending_distortion_sync:
            self.pending_distortion_sync = False
            self.current_distortions = dict(self.pending_distortions)
            return dict(self.current_distortions)
        return None

    def set_recording_state(self, enabled: bool) -> None:
        self.recording_enabled = enabled
        self.record_var.set("Stop Recording" if enabled else "Start Recording")

    def set_loaded_files(self, speech_file: str, noise_file: str) -> None:
        self.loaded_speech_file = speech_file
        self.loaded_noise_file = noise_file
        self.speech_file_var.set(f"Speech: {speech_file}")
        self.noise_file_var.set(f"Noise: {noise_file}")

    def set_active_devices(self, input_device: int | str | None, output_device: int | str | None) -> None:
        self.current_input_device = input_device
        self.current_output_device = output_device

    def set_device_status_message(self, message: str) -> None:
        self.device_status_message = message

    def set_mix_snr_db(self, snr_db: float) -> None:
        snr_db = float(snr_db)
        self.current_mix_snr_db = snr_db
        self.pending_mix_snr_db = snr_db
        self.mix_snr_var.set(snr_db)

    def set_dry_wet(self, value: float) -> None:
        value = float(value)
        self.current_dry_wet = value
        self.pending_dry_wet = value
        self.dry_wet_var.set(value)

    def set_output_gain_db(self, value: float) -> None:
        value = float(value)
        self.current_output_gain_db = value
        self.pending_output_gain_db = value
        self.output_gain_var.set(value)

    def set_scenario(self, scenario_name: str) -> None:
        self.current_scenario = scenario_name
        self.pending_scenario = scenario_name
        self.scenario_var.set(scenario_name)

    def sync_distortion_state(self, distortion_map: dict[str, bool]) -> None:
        self.pending_distortions = dict(distortion_map)
        self.current_distortions = dict(distortion_map)
        for name, _label in self.distortion_labels:
            if name in self.distortion_vars:
                self.distortion_vars[name].set(bool(distortion_map.get(name, False)))

    def _compute_display_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        window = np.hanning(config.FRAME_LEN).astype(np.float32)
        if len(audio) < config.FRAME_LEN:
            padded = np.zeros(config.FRAME_LEN, dtype=np.float32)
            padded[: len(audio)] = audio
            audio = padded

        frames = 1 + max(0, (len(audio) - config.FRAME_LEN) // config.HOP_LEN)
        if frames <= 0:
            frames = 1

        spec = np.zeros((config.N_FREQ_BINS, frames), dtype=np.float32)
        for idx in range(frames):
            start = idx * config.HOP_LEN
            frame = audio[start:start + config.FRAME_LEN]
            if len(frame) < config.FRAME_LEN:
                padded = np.zeros(config.FRAME_LEN, dtype=np.float32)
                padded[: len(frame)] = frame
                frame = padded
            fft_frame = np.fft.rfft(frame * window, n=config.N_FFT)
            mag_db = 20.0 * np.log10(np.maximum(np.abs(fft_frame), 1e-4))
            spec[:, idx] = np.clip(mag_db, -80.0, 0.0)
        return spec

    def refresh(self, stats: StreamStats, method: str) -> None:
        if self.closed:
            return

        with self.lock:
            raw = self.raw_buffer.copy()
            out = self.output_buffer.copy()
            raw_db = self.raw_rms_db
            out_db = self.output_rms_db

        self.raw_line.set_ydata(raw)
        self.out_line.set_ydata(out)
        raw_spec = self._compute_display_spectrogram(raw)
        out_spec = self._compute_display_spectrogram(out)
        history_sec = self.display_samples / self.sample_rate
        self.raw_spec_image.set_data(raw_spec)
        self.out_spec_image.set_data(out_spec)
        self.raw_spec_image.set_extent([-history_sec, 0.0, 0.0, self.sample_rate / 2.0])
        self.out_spec_image.set_extent([-history_sec, 0.0, 0.0, self.sample_rate / 2.0])

        vad_soft = EnergyZCRVAD().predict_soft(raw)
        if len(vad_soft) == 0:
            vad_soft = np.zeros(1, dtype=np.float32)
        vad_time = np.linspace(-history_sec, 0.0, len(vad_soft), endpoint=False)
        self.vad_line.set_data(vad_time, vad_soft)
        self.vad_ax.set_xlim(-history_sec, 0.0)

        raw_level = np.clip(raw_db, -60.0, 0.0)
        out_level = np.clip(out_db, -60.0, 0.0)
        signal_present = raw_db > -45.0
        self.last_signal_present = signal_present
        self.input_level_var.set(raw_level + 60.0)
        self.output_level_var.set(out_level + 60.0)
        self.status_var.set(
            "method={method} | mean_proc={mean:.1f} ms | max_proc={maxv:.1f} ms | in_drop={drop} | out_silence={silence} | recording={recording}".format(
                method=method,
                mean=stats.mean_processing_time * 1000.0,
                maxv=stats.max_processing_time * 1000.0,
                drop=stats.input_overflows,
                silence=stats.output_underflows,
                recording="on" if self.recording_enabled else "off",
            )
        )
        self.summary_var.set(
            "checkpoint={checkpoint} | source={source} | vad={vad} | scenario={scenario} | snr={snr:.1f} dB | dry_wet={dry_wet:.2f} | gain={gain:+.1f} dB | latency={latency:.1f} ms | stream_status={stream_status}".format(
                checkpoint=self.checkpoint_label,
                source=self.current_source_mode,
                vad=self.current_vad_mode,
                scenario=self.current_scenario,
                snr=self.current_mix_snr_db,
                dry_wet=self.current_dry_wet,
                gain=self.current_output_gain_db,
                latency=stats.last_processing_time * 1000.0,
                stream_status=self.device_status_message,
            )
        )
        self.files_var.set(
            "input={input_dev} | output={output_dev} | speech_file={speech} | noise_file={noise}".format(
                input_dev=describe_device(self.current_input_device, "input"),
                output_dev=describe_device(self.current_output_device, "output"),
                speech=self.loaded_speech_file,
                noise=self.loaded_noise_file,
            )
        )
        self.input_db_var.set(f"{raw_db:.1f} dBFS")
        self.output_db_var.set(f"{out_db:.1f} dBFS")

        signal_text = "MIC SIGNAL DETECTED" if signal_present else "NO MIC SIGNAL"
        self.signal_var.set(signal_text)
        self.signal_label.configure(foreground="#2f6b3b" if signal_present else "#a94442")
        self.ax_raw.set_title(f"Raw Microphone Input  |  RMS {raw_db:.1f} dBFS  |  {signal_text}")
        self.ax_out.set_title(f"Enhanced Speaker Output  |  RMS {out_db:.1f} dBFS")
        now = time.time()
        if now - self.last_refresh_time >= 0.08:
            self.canvas.draw_idle()
            self.last_refresh_time = now
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self.closed = True

    def save_snapshot(self, output_dir: Path | None = None) -> Path:
        snapshot_dir = Path(output_dir or (config.RESULTS_DIR / "realtime_snapshots"))
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = snapshot_dir / f"realtime_demo_snapshot_{timestamp}.png"
        self.fig.savefig(snapshot_path, dpi=150, bbox_inches="tight")
        return snapshot_path

    def close(self) -> None:
        if not self.closed:
            try:
                self.root.destroy()
            except Exception:
                pass
        self.closed = True


