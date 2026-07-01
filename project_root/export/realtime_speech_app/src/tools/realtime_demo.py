"""Realtime microphone-to-speaker speech enhancement demo entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src import config
from src.tools.realtime import (
    FileDemoSource,
    LiveComparisonUI,
    RecordingSession,
    RealtimeDenoiseApp,
    StreamingEnhancer,
    list_device_candidates,
    ms_to_samples,
    pick_best_device_pair,
    print_devices,
    resolve_default_checkpoint,
    sd,
)
from src.utils.logger import setup_script_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime speech enhancement: microphone in, denoised speech out to speakers",
    )
    parser.add_argument(
        "--method",
        choices=["masknet", "spectral_subtraction", "wiener", "bypass"],
        default="masknet",
        help="Realtime enhancement backend",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used for --method masknet. Defaults to the best local LibriSpeech-trained checkpoint.",
    )
    parser.add_argument("--device", default=None, help="Torch device for masknet, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--sample-rate", type=int, default=config.SAMPLE_RATE, help="Audio sample rate")
    parser.add_argument("--block-ms", type=float, default=32.0, help="I/O block duration in milliseconds")
    parser.add_argument(
        "--context-ms",
        type=float,
        default=640.0,
        help="Past context window used for each enhancement call",
    )
    parser.add_argument(
        "--tail-ms",
        type=float,
        default=config.FRAME_LEN_MS,
        help="Extra algorithmic delay trimmed from the chunk tail to reduce edge artifacts",
    )
    parser.add_argument("--dry-wet", type=float, default=1.0, help="0=dry monitor, 1=fully enhanced")
    parser.add_argument("--queue-size", type=int, default=8, help="Internal worker queue size")
    parser.add_argument("--latency", choices=["low", "high"], default="low", help="Requested sounddevice latency")
    parser.add_argument("--stats-interval", type=float, default=5.0, help="Seconds between runtime stats lines")
    parser.add_argument("--input-device", default=None, help="Input device id or exact name")
    parser.add_argument("--output-device", default=None, help="Output device id or exact name")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--visualize", action="store_true", help="Open a realtime UI with raw vs enhanced audio")
    parser.add_argument("--record", action="store_true", help="Record raw/enhanced audio while the demo runs")
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="Optional directory where realtime demo recordings will be written",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Optional directory where UI PNG snapshots will be written",
    )
    parser.add_argument(
        "--display-ms",
        type=float,
        default=2500.0,
        help="History duration shown in the realtime UI",
    )
    parser.add_argument("--fullscreen", action="store_true", help="Open the visualization window in fullscreen mode")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_script_logger("realtime_demo")

    if sd is None:
        raise SystemExit(
            "sounddevice is not installed. Run `pip install sounddevice` or `pip install -r requirements.txt`."
        )

    if args.list_devices:
        print_devices()
        return 0

    checkpoint_path = args.checkpoint
    if args.method == "masknet":
        checkpoint_path = checkpoint_path or resolve_default_checkpoint()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    block_size = ms_to_samples(args.block_ms, args.sample_rate)
    context_size = ms_to_samples(args.context_ms, args.sample_rate)
    tail_trim_size = ms_to_samples(args.tail_ms, args.sample_rate)

    enhancer = StreamingEnhancer(
        method=args.method,
        device=device,
        block_size=block_size,
        context_size=context_size,
        tail_trim_size=tail_trim_size,
        dry_wet=args.dry_wet,
        checkpoint_path=checkpoint_path,
    )
    file_source = FileDemoSource(sample_rate=args.sample_rate)

    visualizer = None
    recorder = None
    input_device, output_device = pick_best_device_pair(
        args.input_device,
        args.output_device,
        sample_rate=args.sample_rate,
    )
    input_device_options = list_device_candidates("input", sample_rate=args.sample_rate, limit=5)
    output_device_options = list_device_candidates("output", sample_rate=args.sample_rate, limit=5)

    if args.record:
        recorder = RecordingSession(sample_rate=args.sample_rate, output_dir=args.record_dir)
        recorder.start()

    if args.visualize:
        display_samples = ms_to_samples(args.display_ms, args.sample_rate)
        checkpoint_label = checkpoint_path.name if checkpoint_path is not None else "classic_dsp"
        visualizer = LiveComparisonUI(
            sample_rate=args.sample_rate,
            display_samples=display_samples,
            initial_method=args.method,
            recording_enabled=args.record,
            checkpoint_label=checkpoint_label,
            fullscreen=args.fullscreen,
            initial_vad_mode=enhancer.vad_mode,
            initial_source_mode=file_source.source_mode,
            initial_distortions=file_source.distortions,
            initial_mix_snr_db=file_source.mix_snr_db,
            initial_dry_wet=args.dry_wet,
            initial_output_gain_db=0.0,
            initial_scenario="neutral",
            input_device_options=input_device_options,
            output_device_options=output_device_options,
            initial_input_device=input_device,
            initial_output_device=output_device,
        )
        visualizer.set_loaded_files("None", "None")
        visualizer.set_active_devices(input_device, output_device)

    logger.info(
        "Realtime demo note: laptop speakers can leak back into the mic. Headphones give cleaner results."
    )

    app = RealtimeDenoiseApp(
        enhancer=enhancer,
        sample_rate=args.sample_rate,
        block_size=block_size,
        queue_size=args.queue_size,
        input_device=input_device,
        output_device=output_device,
        latency=args.latency,
        stats_interval=args.stats_interval,
        visualizer=visualizer,
        recorder=recorder,
        snapshot_dir=args.snapshot_dir,
        file_source=file_source,
        output_gain_db=0.0,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
