"""
Validate dataset integrity and structure.
Checks for missing files, corrupt audio, clean-noisy correspondence.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("Warning: soundfile not installed. Audio validation will be limited.")


def check_file_exists(filepath: Path) -> bool:
    """Check if file exists and is not empty."""
    return filepath.exists() and filepath.stat().st_size > 0


def check_audio_file(filepath: Path) -> Tuple[bool, str]:
    """
    Validate audio file can be read.
    Returns (is_valid, error_message)
    """
    if not HAS_SOUNDFILE:
        return True, ""
    
    try:
        info = sf.info(filepath)
        if info.duration == 0:
            return False, "Zero duration"
        if info.samplerate != 16000:
            return False, f"Wrong sample rate: {info.samplerate}"
        if info.channels > 2:
            return False, f"Too many channels: {info.channels}"
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_clean_noisy_correspondence(
    clean_dir: Path,
    noisy_dir: Path,
    verbose: bool = False
) -> Dict:
    """
    Validate that each noisy file has a corresponding clean file.
    Returns validation results dictionary.
    """
    results = {
        'total_noisy': 0,
        'matched': 0,
        'unmatched': [],
        'corrupt_clean': [],
        'corrupt_noisy': []
    }
    
    if not clean_dir.exists():
        print(f"Error: Clean directory not found: {clean_dir}")
        return results
    
    if not noisy_dir.exists():
        print(f"Error: Noisy directory not found: {noisy_dir}")
        return results
    
    clean_files = {f.stem: f for f in clean_dir.rglob("*.flac")}
    clean_files.update({f.stem: f for f in clean_dir.rglob("*.wav")})
    
    noisy_files = list(noisy_dir.glob("*.wav"))
    results['total_noisy'] = len(noisy_files)
    
    print(f"  Clean files: {len(clean_files)}")
    print(f"  Noisy files: {len(noisy_files)}")
    
    for noisy_file in noisy_files:
        import re
        noisy_stem = noisy_file.stem
        base_name = re.sub(r'_snr-?\d+dB.*', '', noisy_stem)
        base_name = re.sub(r'_[A-Z]+_16k', '', base_name)
        base_name = re.sub(r'_[A-Z]+$', '', base_name)
        
        clean_match = None
        if base_name in clean_files:
            clean_match = clean_files[base_name]
        else:
            for stem, path in clean_files.items():
                if stem.startswith(base_name):
                    clean_match = path
                    break
        
        if clean_match:
            results['matched'] += 1
            
            is_valid, error = check_audio_file(clean_match)
            if not is_valid:
                results['corrupt_clean'].append((clean_match.name, error))
            
            is_valid, error = check_audio_file(noisy_file)
            if not is_valid:
                results['corrupt_noisy'].append((noisy_file.name, error))
        else:
            results['unmatched'].append(noisy_file.name)
            if verbose:
                print(f"    Warning: No clean match for: {noisy_file.name}")
    
    return results


def validate_metadata(metadata_path: Path) -> Dict:
    """
    Validate metadata file.
    Returns validation results.
    """
    results = {
        'exists': False,
        'valid_json': False,
        'has_metadata': False,
        'has_statistics': False,
        'num_entries': 0
    }
    
    if not metadata_path.exists():
        return results
    
    results['exists'] = True
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results['valid_json'] = True
        
        if 'metadata' in data:
            results['has_metadata'] = True
            results['num_entries'] = len(data['metadata'])
        
        if 'statistics' in data:
            results['has_statistics'] = True
        
    except json.JSONDecodeError:
        pass
    
    return results


def validate_dataset(dataset_name: str, data_root: Path) -> Dict:
    """
    Validate entire dataset.
    Returns comprehensive validation results.
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING {dataset_name.upper()}")
    print(f"{'='*70}")
    
    results = {
        'dataset': dataset_name,
        'splits': {}
    }
    
    if dataset_name == 'librispeech':
        root = data_root / "librispeech_demand"
        splits = ['train', 'test', 'val']
    else:
        root = data_root / "voicebank_demand"
        splits = ['train', 'test']
    
    for split in splits:
        print(f"\n Validating {split} split...")
        split_results = {}
        
        if dataset_name == 'librispeech':
            clean_dir = root / "clean" / split
            noisy_dir = root / "noisy" / split
        else:
            clean_dir = root / split / "clean"
            noisy_dir = root / split / "noisy"
        
        if clean_dir.exists() and noisy_dir.exists():
            corr_results = validate_clean_noisy_correspondence(
                clean_dir, noisy_dir, verbose=False
            )
            split_results['correspondence'] = corr_results
            
            match_rate = (corr_results['matched'] / corr_results['total_noisy'] * 100 
                         if corr_results['total_noisy'] > 0 else 0)
            
            print(f"  Matched: {corr_results['matched']}/{corr_results['total_noisy']} ({match_rate:.1f}%)")
            
            if corr_results['unmatched']:
                print(f"  Warning: Unmatched: {len(corr_results['unmatched'])}")
            
            if corr_results['corrupt_clean']:
                print(f"  Error: Corrupt clean files: {len(corr_results['corrupt_clean'])}")
                for fname, error in corr_results['corrupt_clean'][:5]:
                    print(f"      - {fname}: {error}")
            
            if corr_results['corrupt_noisy']:
                print(f"   Corrupt noisy files: {len(corr_results['corrupt_noisy'])}")
                for fname, error in corr_results['corrupt_noisy'][:5]:
                    print(f"      - {fname}: {error}")
        else:
            print(f"  ️  Directories not found")
        
        metadata_path = root / f"metadata_{split}.json"
        meta_results = validate_metadata(metadata_path)
        split_results['metadata'] = meta_results
        
        if meta_results['exists']:
            print(f"   Metadata exists: {meta_results['num_entries']} entries")
        else:
            print(f"  ️  Metadata not found: {metadata_path}")
        
        results['splits'][split] = split_results
    
    return results


def print_summary(results: Dict):
    """Print validation summary."""
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    total_issues = 0
    
    for dataset_result in results:
        dataset_name = dataset_result['dataset']
        print(f"\n {dataset_name.upper()}:")
        
        for split, split_data in dataset_result['splits'].items():
            print(f"  {split}:")
            
            if 'correspondence' in split_data:
                corr = split_data['correspondence']
                unmatched = len(corr['unmatched'])
                corrupt_clean = len(corr['corrupt_clean'])
                corrupt_noisy = len(corr['corrupt_noisy'])
                
                issues = unmatched + corrupt_clean + corrupt_noisy
                total_issues += issues
                
                if issues == 0:
                    print(f"     All files valid")
                else:
                    if unmatched > 0:
                        print(f"    ️  {unmatched} unmatched files")
                    if corrupt_clean > 0:
                        print(f"     {corrupt_clean} corrupt clean files")
                    if corrupt_noisy > 0:
                        print(f"     {corrupt_noisy} corrupt noisy files")
            
            if 'metadata' in split_data:
                meta = split_data['metadata']
                if not meta['exists']:
                    print(f"    ️  Metadata missing")
                    total_issues += 1
    
    print(f"\n{'='*70}")
    if total_issues == 0:
        print(" All validation checks passed!")
    else:
        print(f"️  Found {total_issues} issues that need attention")
    print(f"{'='*70}")


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate dataset integrity')
    parser.add_argument('--dataset', type=str, choices=['librispeech', 'voicebank', 'both'],
                       default='both', help='Which dataset to validate')
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root data directory')
    args = parser.parse_args()
    
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(__file__).resolve().parent.parent.parent / "data"
    
    print("="*70)
    print("DATASET VALIDATION")
    print("="*70)
    print(f"Data root: {data_root}")
    
    all_results = []
    
    if args.dataset in ['librispeech', 'both']:
        results = validate_dataset('librispeech', data_root)
        all_results.append(results)
    
    if args.dataset in ['voicebank', 'both']:
        results = validate_dataset('voicebank', data_root)
        all_results.append(results)
    
    print_summary(all_results)


if __name__ == "__main__":
    main()
