"""Generate dataset metadata JSON files."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import argparse

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("Warning: soundfile not installed. Duration info will be skipped.")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable


def extract_snr_from_filename(filename: str) -> Optional[float]:
    """Extract SNR value from a filename."""
    match = re.search(r'snr(-?\d+)dB', filename)
    if match:
        return float(match.group(1))
    return None


def extract_noise_type_from_filename(filename: str) -> Optional[str]:
    """Extract noise type from a filename."""
    match = re.search(r'_(T[A-Z]+|[A-Z]+_16k)_', filename)
    if match:
        return match.group(1).replace('_16k', '')
    return None


def get_audio_info(filepath: Path) -> Dict:
    """Get audio file information (duration, sample rate, channels)."""
    if not HAS_SOUNDFILE:
        return {
            'duration_s': None,
            'sample_rate': None,
            'channels': None,
            'frames': None
        }
    
    try:
        info = sf.info(filepath)
        return {
            'duration_s': float(info.duration),
            'sample_rate': int(info.samplerate),
            'channels': int(info.channels),
            'frames': int(info.frames)
        }
    except Exception as e:
        return {
            'error': str(e),
            'duration_s': None,
            'sample_rate': None,
            'channels': None,
            'frames': None
        }


def find_clean_match(noisy_file: Path, clean_files: List[Path]) -> Optional[Path]:
    """Find the clean file for a noisy file."""
    noisy_stem = noisy_file.stem
    
    base_name = re.sub(r'_snr-?\d+dB.*', '', noisy_stem)
    base_name = re.sub(r'_[A-Z]+_16k', '', base_name)
    base_name = re.sub(r'_[A-Z]+$', '', base_name)
    
    for clean_file in clean_files:
        if clean_file.stem == base_name or clean_file.stem.startswith(base_name):
            return clean_file
    
    return None


def generate_librispeech_metadata(split: str = 'train', data_root: Path = None) -> Dict:
    """Generate LibriSpeech-DEMAND metadata."""
    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent.parent / "data"
    
    librispeech_root = data_root / "librispeech_demand"
    clean_dir = librispeech_root / "clean" / split
    noisy_dir = librispeech_root / "noisy" / split
    
    if not clean_dir.exists():
        print(f"Warning: {clean_dir} does not exist")
        return {}
    if not noisy_dir.exists():
        print(f"Warning: {noisy_dir} does not exist")
        return {}
    
    clean_files = sorted(clean_dir.rglob("*.flac"))
    noisy_files = sorted(noisy_dir.glob("*.wav"))
    
    print(f"\n{split.upper()} set:")
    print(f"  Clean files: {len(clean_files)}")
    print(f"  Noisy files: {len(noisy_files)}")
    
    metadata = {}
    
    for noisy_file in tqdm(noisy_files, desc=f"Processing {split}"):
        noisy_name = noisy_file.name
        
        snr_db = extract_snr_from_filename(noisy_name)
        noise_type = extract_noise_type_from_filename(noisy_name)
        
        clean_match = find_clean_match(noisy_file, clean_files)
        
        audio_info = get_audio_info(noisy_file)
        
        file_metadata = {
            'noisy_file': noisy_name,
            'clean_file': clean_match.name if clean_match else None,
            'clean_relative_path': str(clean_match.relative_to(clean_dir)) if clean_match else None,
            'snr_db': snr_db,
            'noise_type': noise_type,
            'split': split,
            **audio_info
        }
        
        metadata[noisy_name] = file_metadata
    
    return metadata


def generate_voicebank_metadata(split: str = 'train', data_root: Path = None) -> Dict:
    """Generate VoiceBank-DEMAND metadata."""
    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent.parent / "data"
    
    voicebank_root = data_root / "voicebank_demand"
    clean_dir = voicebank_root / "clean" / split
    noisy_dir = voicebank_root / "noisy" / split
    
    if not clean_dir.exists():
        print(f"Warning: {clean_dir} does not exist")
        return {}
    if not noisy_dir.exists():
        print(f"Warning: {noisy_dir} does not exist")
        return {}
    
    clean_files = sorted(clean_dir.glob("*.wav"))
    noisy_files = sorted(noisy_dir.glob("*.wav"))
    
    print(f"\n{split.upper()} set:")
    print(f"  Clean files: {len(clean_files)}")
    print(f"  Noisy files: {len(noisy_files)}")
    
    metadata = {}
    
    for noisy_file in tqdm(noisy_files, desc=f"Processing {split}"):
        noisy_name = noisy_file.name
        
        snr_db = extract_snr_from_filename(noisy_name)
        noise_type = extract_noise_type_from_filename(noisy_name)
        
        clean_match = find_clean_match(noisy_file, clean_files)
        
        audio_info = get_audio_info(noisy_file)
        
        speaker_id = noisy_name.split('_')[0] if '_' in noisy_name else None
        
        file_metadata = {
            'noisy_file': noisy_name,
            'clean_file': clean_match.name if clean_match else None,
            'speaker_id': speaker_id,
            'snr_db': snr_db,
            'noise_type': noise_type,
            'split': split,
            **audio_info
        }
        
        metadata[noisy_name] = file_metadata
    
    return metadata


def compute_statistics(metadata: Dict) -> Dict:
    """Compute statistics from metadata."""
    if not metadata:
        return {}
    
    total_files = len(metadata)
    total_duration = sum(m['duration_s'] for m in metadata.values() if m.get('duration_s'))
    
    snr_values = [m['snr_db'] for m in metadata.values() if m.get('snr_db') is not None]
    noise_types = [m['noise_type'] for m in metadata.values() if m.get('noise_type')]
    
    snr_distribution = {}
    for snr in snr_values:
        snr_distribution[snr] = snr_distribution.get(snr, 0) + 1
    
    noise_distribution = {}
    for noise in noise_types:
        noise_distribution[noise] = noise_distribution.get(noise, 0) + 1
    
    stats = {
        'total_files': total_files,
        'total_duration_hours': round(total_duration / 3600, 2),
        'total_duration_seconds': round(total_duration, 2),
        'snr_distribution': snr_distribution,
        'noise_type_distribution': noise_distribution,
        'num_unique_snr_levels': len(snr_distribution),
        'num_unique_noise_types': len(noise_distribution)
    }
    
    return stats


def save_metadata(metadata: Dict, output_path: Path, stats: Dict = None):
    """Save metadata to JSON file."""
    output_data = {
        'metadata': metadata,
        'statistics': stats if stats else compute_statistics(metadata),
        'num_files': len(metadata)
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n Metadata saved: {output_path}")
    print(f"   Files: {len(metadata)}")
    if stats:
        print(f"   Duration: {stats.get('total_duration_hours', 0):.2f} hours")


def main():
    """Generate metadata files."""
    parser = argparse.ArgumentParser(description='Generate metadata for datasets')
    parser.add_argument('--dataset', type=str, choices=['librispeech', 'voicebank', 'both'], 
                       default='both', help='Which dataset to process')
    parser.add_argument('--splits', nargs='+', default=['train', 'test', 'val'],
                       help='Splits to process')
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root data directory (default: ./data)')
    args = parser.parse_args()
    
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(__file__).resolve().parent.parent.parent / "data"
    
    print("="*70)
    print("GENERATING DATASET METADATA")
    print("="*70)
    print(f"Data root: {data_root}")
    
    if args.dataset in ['librispeech', 'both']:
        print("\n Processing LibriSpeech-DEMAND...")
        librispeech_root = data_root / "librispeech_demand"
        
        for split in args.splits:
            if split == 'val' or split in ['train', 'test']:
                metadata = generate_librispeech_metadata(split, data_root)
                
                if metadata:
                    stats = compute_statistics(metadata)
                    
                    output_path = librispeech_root / "metadata" / f"metadata_{split}.json"
                    save_metadata(metadata, output_path, stats)
                    
                    print(f"\n  {split.upper()} Statistics:")
                    print(f"    Total files: {stats['total_files']}")
                    print(f"    Duration: {stats['total_duration_hours']} hours")
                    print(f"    SNR levels: {list(stats['snr_distribution'].keys())}")
                    print(f"    Noise types: {list(stats['noise_type_distribution'].keys())}")
    
    if args.dataset in ['voicebank', 'both']:
        print("\n Processing VoiceBank-DEMAND...")
        voicebank_root = data_root / "voicebank_demand"
        
        for split in ['train', 'test']:
            if split in args.splits:
                metadata = generate_voicebank_metadata(split, data_root)
                
                if metadata:
                    stats = compute_statistics(metadata)
                    
                    output_path = voicebank_root / "metadata" / f"metadata_{split}.json"
                    save_metadata(metadata, output_path, stats)
                    
                    print(f"\n  {split.upper()} Statistics:")
                    print(f"    Total files: {stats['total_files']}")
                    print(f"    Duration: {stats['total_duration_hours']} hours")
                    if stats['snr_distribution']:
                        print(f"    SNR levels: {list(stats['snr_distribution'].keys())}")
                    if stats['noise_type_distribution']:
                        print(f"    Noise types: {list(stats['noise_type_distribution'].keys())}")
    
    print("\n" + "="*70)
    print(" METADATA GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
