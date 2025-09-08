# debug/debug_audio_noise.py

import os
import json
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Tuple, Optional
import sys

# Add project root to path to allow importing our modules if needed in the future
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ... (calculate_rms and add_noise_with_snr_debug functions are unchanged and correct) ...
def calculate_rms(audio: np.ndarray) -> float:
    """Calculate Root Mean Square of audio signal"""
    return np.sqrt(np.mean(audio**2))

def add_noise_with_snr_debug(clean_audio: np.ndarray, snr_db: float, verbose: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Add noise with SNR tracking and debugging info
    Returns: (noisy_audio, debug_info)
    """
    debug_info = {}
    if clean_audio.ndim == 2:
        debug_info['audio_type'] = 'stereo'
        debug_info['channels'] = clean_audio.shape[1]
        noisy_channels, channel_debug = [], []
        for i in range(clean_audio.shape[1]):
            channel = clean_audio[:, i]
            noisy_channel, ch_debug = add_noise_with_snr_debug(channel, snr_db, verbose=False)
            noisy_channels.append(noisy_channel)
            channel_debug.append(ch_debug)
        debug_info['channel_info'] = channel_debug
        noisy_audio = np.stack(noisy_channels, axis=1)
        if verbose: print(f"  Processed stereo audio ({clean_audio.shape[1]} channels)")
    else:
        debug_info['audio_type'] = 'mono'; debug_info['length'] = len(clean_audio)
        rms_clean = calculate_rms(clean_audio)
        debug_info['rms_clean'] = rms_clean
        if rms_clean == 0:
            debug_info['warning'] = 'Clean audio has zero RMS'
            return clean_audio, debug_info
        rms_noise_target = rms_clean / (10**(snr_db / 20))
        debug_info['rms_noise_target'] = rms_noise_target
        noise = np.random.normal(0, 1, len(clean_audio))
        rms_current_noise = calculate_rms(noise)
        debug_info['rms_noise_unscaled'] = rms_current_noise
        if rms_current_noise == 0:
            debug_info['warning'] = 'Generated noise has zero RMS'
            return clean_audio, debug_info
        scaled_noise = noise * (rms_noise_target / rms_current_noise)
        debug_info['rms_noise_scaled'] = calculate_rms(scaled_noise)
        noisy_audio = clean_audio + scaled_noise
        debug_info['rms_noisy'] = calculate_rms(noisy_audio)
        actual_snr = 20 * np.log10(rms_clean / calculate_rms(scaled_noise))
        debug_info['actual_snr'] = actual_snr; debug_info['target_snr'] = snr_db
        debug_info['snr_error'] = abs(actual_snr - snr_db)
        max_val = np.max(np.abs(noisy_audio))
        debug_info['max_amplitude'] = max_val; debug_info['clipped'] = max_val > 1.0
        if debug_info['clipped']:
            original_max = max_val
            noisy_audio = noisy_audio / max_val * 0.95
            debug_info['normalized'] = True; debug_info['normalization_factor'] = 0.95 / original_max
            if verbose: print(f"  WARNING: Audio clipped (max={original_max:.3f}), normalized")
        if verbose: print(f"  Target SNR: {snr_db}dB, Actual SNR: {actual_snr:.2f}dB (error: {debug_info['snr_error']:.2f}dB)")
    return noisy_audio, debug_info

def verify_existing_noisy_audio(clean_path: str, noisy_path: str, expected_snr: float) -> Optional[dict]:
    """Verify SNR of existing noisy audio files"""
    if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
        print(f"  ERROR: Files not found - Clean: {os.path.exists(clean_path)}, Noisy: {os.path.exists(noisy_path)}")
        return None
    try:
        clean_audio, sr1 = sf.read(clean_path, dtype='float32')
        noisy_audio, sr2 = sf.read(noisy_path, dtype='float32')
        if sr1 != sr2: print(f"  WARNING: Sample rate mismatch - Clean: {sr1}, Noisy: {sr2}")
        if clean_audio.ndim == 2: clean_audio = clean_audio[:, 0]
        if noisy_audio.ndim == 2: noisy_audio = noisy_audio[:, 0]
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio, noisy_audio = clean_audio[:min_len], noisy_audio[:min_len]
        rms_clean = calculate_rms(clean_audio)
        noise = noisy_audio - clean_audio
        rms_noise = calculate_rms(noise)
        if rms_noise > 0 and rms_clean > 0:
            actual_snr = 20 * np.log10(rms_clean / rms_noise)
            verification_data = {'expected_snr': expected_snr, 'actual_snr': actual_snr, 'snr_error': abs(actual_snr - expected_snr), 'rms_clean': rms_clean, 'rms_noise': rms_noise, 'max_clean': np.max(np.abs(clean_audio)), 'max_noisy': np.max(np.abs(noisy_audio)), 'sample_rate': sr1, 'length': len(clean_audio)}
            print(f"  Expected: {expected_snr}dB, Actual: {actual_snr:.2f}dB, Error: {verification_data['snr_error']:.2f}dB")
            return verification_data
        else:
            print(f"  ERROR: Zero RMS - Clean: {rms_clean}, Noise: {rms_noise}")
            return None
    except Exception as e:
        print(f"  ERROR reading audio files: {e}")
        return None

def test_synthetic_audio(sample_rate: int = 16000, duration: float = 1.0):
    """Test noise generation with synthetic sine wave"""
    print("\n" + "="*60 + "\nTESTING NOISE GENERATION WITH SYNTHETIC AUDIO\n" + "="*60)
    frequency = 440; t = np.linspace(0, duration, int(sample_rate * duration))
    clean_signal = 0.1 * np.sin(2 * np.pi * frequency * t)
    test_snrs = [20, 10, 0, -10, -20]; results = []
    for snr_db in test_snrs:
        print(f"\nTesting SNR: {snr_db}dB")
        noisy_signal, debug_info = add_noise_with_snr_debug(clean_signal, snr_db)
        output_dir = Path('debug_audio_output'); output_dir.mkdir(exist_ok=True)
        clean_file = output_dir / 'test_clean.wav'; noisy_file = output_dir / f'test_snr_{snr_db}db.wav'
        sf.write(clean_file, clean_signal, sample_rate); sf.write(noisy_file, noisy_signal, sample_rate)
        results.append({'target_snr': snr_db, 'actual_snr': debug_info.get('actual_snr', 'N/A'), 'debug_info': debug_info})
        print(f"  Files saved: {clean_file}, {noisy_file}")
    return results

def verify_dataset_noise(dataset_path: str, num_samples: int = 5):
    """Verify noise in actual dataset files"""
    print("\n" + "="*60 + "\nVERIFYING ACTUAL DATASET NOISE LEVELS\n" + "="*60)
    dataset_path = Path(dataset_path)
    jsonl_files = list(dataset_path.glob('*_noisy_standardized.jsonl'))
    if not jsonl_files:
        print(f"ERROR: No noisy JSONL file found in {dataset_path}"); return
    jsonl_file = jsonl_files[0]; print(f"Using JSONL file: {jsonl_file}")
    with open(jsonl_file, 'r') as f: data = [json.loads(line) for line in f]
    by_snr = {}
    for item in data:
        snr = item.get('snr_db', 'unknown')
        if snr not in by_snr: by_snr[snr] = []
        by_snr[snr].append(item)
    print(f"Found {len(data)} noisy samples across {len(by_snr)} SNR levels: {list(by_snr.keys())}")
    verification_results = []
    for snr_level in sorted(by_snr.keys()):
        if snr_level == 'unknown': continue
        samples = by_snr[snr_level][:num_samples]
        print(f"\nVerifying SNR {snr_level}dB ({len(samples)} samples):")
        snr_results = []
        for i, sample in enumerate(samples):
            print(f"  Sample {i+1}: {Path(sample['audio_path']).name}")
            
            # --- THE CRITICAL FIX ---
            # The paths in the JSONL are relative to the project root. We must
            # construct the full path before passing it to the verification function.
            project_root = Path(__file__).resolve().parents[1]
            original_full_path = project_root / sample['original_audio_path']
            noisy_full_path = project_root / sample['audio_path']
            # --- END OF FIX ---

            result = verify_existing_noisy_audio(str(original_full_path), str(noisy_full_path), snr_level)
            if result: snr_results.append(result)
        if snr_results:
            avg_error = np.mean([r['snr_error'] for r in snr_results])
            avg_actual = np.mean([r['actual_snr'] for r in snr_results])
            print(f"  Average actual SNR: {avg_actual:.2f}dB (avg error: {avg_error:.2f}dB)")
            verification_results.append({'target_snr': snr_level, 'samples': snr_results, 'avg_actual_snr': avg_actual, 'avg_error': avg_error})
    return verification_results

def create_listening_test_files(dataset_path: str, output_dir: str = 'listening_test'):
    """Create a small set of files for human listening verification"""
    print("\n" + "="*60 + "\nCREATING LISTENING TEST FILES\n" + "="*60)
    dataset_path = Path(dataset_path); output_path = Path(output_dir); output_path.mkdir(exist_ok=True)
    jsonl_files = list(dataset_path.glob('*_noisy_standardized.jsonl'))
    if not jsonl_files: print(f"ERROR: No noisy JSONL file found in {dataset_path}"); return
    with open(jsonl_files[0], 'r') as f: data = [json.loads(line) for line in f]
    by_snr = {}
    for item in data:
        snr = item.get('snr_db')
        if snr is not None and snr not in by_snr: by_snr[snr] = item
    print(f"Creating listening test files for SNR levels: {sorted(by_snr.keys())}")
    for snr, sample in by_snr.items():
        original_path = Path(sample['original_audio_path'])
        if original_path.exists():
            clean_output = output_path / f"clean_{original_path.stem}.wav"
            os.system(f"cp '{original_path}' '{clean_output}'")
        noisy_path = Path(sample['audio_path'])
        if noisy_path.exists():
            noisy_output = output_path / f"snr_{snr}db_{original_path.stem}.wav"
            os.system(f"cp '{noisy_path}' '{noisy_output}'")
    print(f"Listening test files created in: {output_path}")
    print("Listen to these files to verify the noise levels sound appropriate to human ears")

def run_comprehensive_debug(dataset_path: str, run_synthetic: bool = True, num_samples: int = 3):
    """Run all debugging tests"""
    print("COMPREHENSIVE AUDIO NOISE DEBUGGING\n" + "="*80)
    results = {'synthetic_test': None, 'dataset_verification': None, 'summary': {}}
    if run_synthetic: results['synthetic_test'] = test_synthetic_audio()
    if dataset_path and os.path.exists(dataset_path):
        results['dataset_verification'] = verify_dataset_noise(dataset_path, num_samples)
        create_listening_test_files(dataset_path)
    else:
        print(f"\nWARNING: Dataset path not found: {dataset_path}")
    print("\n" + "="*60 + "\nDEBUGGING SUMMARY\n" + "="*60)
    if results['synthetic_test']:
        print("\nSynthetic Audio Test Results:")
        for result in results['synthetic_test']:
            target, actual = result['target_snr'], result.get('actual_snr', 'N/A')
            if actual != 'N/A': print(f"  {target:3d}dB → {actual:6.2f}dB (error: {abs(actual - target):4.2f}dB)")
    if results['dataset_verification']:
        print("\nDataset Verification Results:")
        for result in results['dataset_verification']:
            target, actual, error = result['target_snr'], result['avg_actual_snr'], result['avg_error']
            print(f"  {target:3d}dB → {actual:6.2f}dB (avg error: {error:4.2f}dB)")
    print(f"\nCheck the 'debug_audio_output' and 'listening_test' folders for audio files")
    print("Listen to the files to verify they sound appropriately noisy to human ears")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug audio noise generation and verify SNR levels")
    parser.add_argument('--dataset-path', type=str, help="Path to a *specific* noisy dataset directory (e.g., 'data/mmar_noisy').")
    parser.add_argument('--no-synthetic', action='store_true', help="Skip synthetic audio test")
    parser.add_argument('--num-samples', type=int, default=3, help="Number of samples to verify per SNR level")
    args = parser.parse_args()
    run_comprehensive_debug(dataset_path=args.dataset_path, run_synthetic=not args.no_synthetic, num_samples=args.num_samples)