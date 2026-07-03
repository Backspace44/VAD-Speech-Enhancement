"""Quick demo for VAD, enhancement, metrics, plots, and audio export."""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import soundfile as sf

from src.dsp.metrics import evaluate_speech_enhancement
from src.dsp.vad_energy_zcr import EnergyZCRVAD
from src.dsp.stft_utils import compute_stft
from src.models import load_masknet_checkpoint
from src.utils.audio_io import load_audio_mono
from src.utils.enhancement_pipeline import run_enhancement_methods
from src.utils.metric_reporting import build_demo_metric_rows, format_metric_lines
from src import config


def load_demo_file(test_dir: Path, max_duration: float = 3.0):
    """Load a short clean/noisy demo pair."""
    clean_dir = test_dir / "clean"
    noisy_dir = test_dir / "noisy"
    
    for clean_path in sorted(clean_dir.glob("*.wav"))[:10]:
        noisy_path = noisy_dir / clean_path.name
        if not noisy_path.exists():
            continue
        
        clean, sr = load_audio_mono(clean_path)
        if len(clean) / sr <= max_duration:
            noisy, _ = load_audio_mono(noisy_path)
            return clean, noisy, clean_path.name, sr
    
    clean_path = sorted(clean_dir.glob("*.wav"))[0]
    noisy_path = noisy_dir / clean_path.name
    clean, sr = load_audio_mono(clean_path)
    noisy, _ = load_audio_mono(noisy_path)
    
    max_samples = int(max_duration * sr)
    return clean[:max_samples], noisy[:max_samples], clean_path.name, sr


def demo_vad(audio: np.ndarray, sr: int):
    """Run the VAD demo."""
    print("\n" + "="*70)
    print(" 1. VOICE ACTIVITY DETECTION (Energy + ZCR)")
    print("="*70)
    
    vad_model = EnergyZCRVAD()
    vad_labels = vad_model.predict(audio)
    
    speech_ratio = np.mean(vad_labels) * 100
    num_frames = len(vad_labels)
    
    print(f"Audio duration:  {len(audio)/sr:.2f} seconds")
    print(f"Total frames:    {num_frames}")
    print(f"Speech frames:   {np.sum(vad_labels)} ({speech_ratio:.1f}%)")
    print(f"Silence frames:  {num_frames - np.sum(vad_labels)} ({100-speech_ratio:.1f}%)")
    
    return vad_labels


def demo_enhancement(clean: np.ndarray, noisy: np.ndarray, model, device: torch.device):
    """Run the enhancement methods."""
    print("\n" + "="*70)
    print(" 2. SPEECH ENHANCEMENT METHODS")
    print("="*70)
    
    print("\nProcessing: shared enhancement pipeline...")
    results = run_enhancement_methods(noisy, model=model, device=device)
    
    return results


def demo_metrics(clean: np.ndarray, results: dict, sr: int):
    """Compute and display performance metrics."""
    print("\n" + "="*70)
    print(" 3. PERFORMANCE METRICS")
    print("="*70)
    
    evaluated_metrics = {}
    for method, enhanced in results.items():
        evaluated_metrics[method] = evaluate_speech_enhancement(clean, enhanced, sr)
        print(f"\n{method.replace('_', ' ')}:")
        for line in format_metric_lines(evaluated_metrics[method]):
            print(line)

    return build_demo_metric_rows(evaluated_metrics)


def demo_visualization(clean: np.ndarray, noisy: np.ndarray, results: dict, 
                       vad_labels: np.ndarray, metrics_data: list, sr: int, output_dir: Path):
    """Create demo plots."""
    print("\n" + "="*70)
    print(" 4. GENERATING VISUALIZATIONS")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    time = np.arange(len(clean)) / sr
    
    axes[0].plot(time, clean, linewidth=0.5)
    axes[0].set_title('Clean Speech', fontsize=12, weight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(time, noisy, linewidth=0.5, color='red')
    axes[1].set_title('Noisy Speech', fontsize=12, weight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(time, results['Spectral_Subtraction'], linewidth=0.5, color='orange')
    axes[2].set_title('Enhanced (Spectral Subtraction)', fontsize=12, weight='bold')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(alpha=0.3)
    
    axes[3].plot(time, results['MaskNet'], linewidth=0.5, color='green')
    axes[3].set_title('Enhanced (MaskNet)', fontsize=12, weight='bold')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'demo_waveforms.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_waveforms.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 4))
    frame_times = np.arange(len(vad_labels)) * config.HOP_LEN / sr
    ax.fill_between(frame_times, 0, vad_labels, alpha=0.3, color='green', label='Speech')
    ax.plot(time, clean / np.max(np.abs(clean)), linewidth=0.5, alpha=0.7, label='Clean Audio')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('VAD / Normalized Amplitude')
    ax.set_title('Voice Activity Detection', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'demo_vad.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_vad.png")
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    methods = [m['Method'] for m in metrics_data]
    pesq_vals = [m['PESQ'] for m in metrics_data]
    stoi_vals = [m['STOI'] for m in metrics_data]
    snr_vals = [m['SNR'] for m in metrics_data]
    
    colors = ['red', 'orange', 'blue', 'green']
    
    axes[0].bar(methods, pesq_vals, color=colors, alpha=0.7)
    axes[0].set_ylabel('PESQ')
    axes[0].set_title('Perceptual Quality', fontsize=12, weight='bold')
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].grid(alpha=0.3, axis='y')
    
    axes[1].bar(methods, stoi_vals, color=colors, alpha=0.7)
    axes[1].set_ylabel('STOI')
    axes[1].set_title('Speech Intelligibility', fontsize=12, weight='bold')
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].grid(alpha=0.3, axis='y')
    
    axes[2].bar(methods, snr_vals, color=colors, alpha=0.7)
    axes[2].set_ylabel('SNR (dB)')
    axes[2].set_title('Signal-to-Noise Ratio', fontsize=12, weight='bold')
    axes[2].set_xticklabels(methods, rotation=45, ha='right')
    axes[2].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'demo_metrics.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_metrics.png")
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    plot_data = [('Clean', clean), ('Noisy', noisy), 
                 ('Spectral Subtraction', results['Spectral_Subtraction']),
                 ('MaskNet', results['MaskNet'])]
    
    for ax, (name, audio) in zip(axes.flat, plot_data):
        mag, _ = compute_stft(torch.from_numpy(audio).float(), 
                              n_fft=config.N_FFT, hop_length=config.HOP_LEN, 
                              win_length=config.FRAME_LEN)
        mag_db = 20 * torch.log10(mag + 1e-8)
        
        im = ax.imshow(mag_db.numpy(), aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(name, fontsize=11, weight='bold')
        ax.set_xlabel('Time Frame')
        ax.set_ylabel('Frequency Bin')
        plt.colorbar(im, ax=ax, label='Magnitude (dB)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'demo_spectrograms.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_spectrograms.png")
    plt.close()


def demo_audio_export(results: dict, sr: int, output_dir: Path):
    """Export enhanced audio files."""
    print("\n" + "="*70)
    print(" 5. EXPORTING AUDIO FILES")
    print("="*70)
    
    audio_dir = output_dir / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    for method, audio in results.items():
        filename = f"demo_{method.lower()}.wav"
        sf.write(audio_dir / filename, audio, sr)
        print(f"Saved: audio/{filename}")


def main():
    """Run the demo."""
    parser = argparse.ArgumentParser(
        description='Demo script for project presentation'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/masknet_best.pth',
        help='Path to MaskNet checkpoint'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='data/voicebank_demand/test',
        help='Directory with test files (clean/ and noisy/ subdirs)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='demo_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*70)
    print(" SPEECH ENHANCEMENT - PROJECT DEMONSTRATION")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    print("\nLoading MaskNet model...")
    model = load_masknet_checkpoint(args.checkpoint, device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params:,} parameters")
    
    print("\nLoading demo audio file (< 3 seconds for fast processing)...")
    clean, noisy, filename, sr = load_demo_file(test_dir)
    print(f"Loaded: {filename} ({len(clean)/sr:.2f}s, {sr}Hz)")
    
    vad_labels = demo_vad(clean, sr)
    results = demo_enhancement(clean, noisy, model, device)
    metrics_data = demo_metrics(clean, results, sr)
    demo_visualization(clean, noisy, results, vad_labels, metrics_data, sr, output_dir)
    demo_audio_export(results, sr, output_dir)
    
    print("\n" + "="*70)
    print(" DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - demo_waveforms.png     : Waveform comparisons")
    print("  - demo_vad.png           : Voice activity detection")
    print("  - demo_metrics.png       : Performance metrics")
    print("  - demo_spectrograms.png  : Frequency domain analysis")
    print("  - audio/*.wav            : Enhanced audio files")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
