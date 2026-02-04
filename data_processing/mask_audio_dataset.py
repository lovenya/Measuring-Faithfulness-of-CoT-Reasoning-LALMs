# data_processing/mask_audio_dataset.py

"""
Generate masked audio datasets at various percentile levels.

This script creates pre-processed masked audio files for the audio masking experiments.
It supports two masking types (silence, noise) and three position modes (random, start, end).

Usage:
    python mask_audio_dataset.py --source data/mmar --output data/mmar_masked \
        --mask-type silence --mode random --levels 10 20 30 40 50 60 70 80 90 100

    # Process all sakura subdatasets:
    python mask_audio_dataset.py --source data/sakura/animal --output data/sakura/animal_masked \
        --mask-type noise --mode start --levels 10 20 30

    # Quick test with limited samples:
    python mask_audio_dataset.py --source data/mmar --output data/mmar_masked_test \
        --mask-type silence --mode random --levels 10 50 --num-samples 5 --verbose
"""

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate the Root Mean Square of an audio signal."""
    return np.sqrt(np.mean(audio**2))


def mask_audio_with_silence(audio: np.ndarray, start_sample: int, end_sample: int) -> np.ndarray:
    """Replace audio segment with silence (zeros)."""
    masked_audio = audio.copy()
    if audio.ndim == 2:
        # Stereo
        masked_audio[start_sample:end_sample, :] = 0.0
    else:
        # Mono
        masked_audio[start_sample:end_sample] = 0.0
    return masked_audio


def mask_audio_with_noise(audio: np.ndarray, start_sample: int, end_sample: int, snr_db: float = 0) -> np.ndarray:
    """
    Replace audio segment with white noise.
    The noise level is calibrated to match the RMS of the surrounding audio.
    """
    masked_audio = audio.copy()
    segment_length = end_sample - start_sample
    
    if audio.ndim == 2:
        # Stereo: process each channel
        for ch in range(audio.shape[1]):
            # Calculate RMS from the non-masked portion
            non_masked = np.concatenate([audio[:start_sample, ch], audio[end_sample:, ch]])
            if len(non_masked) > 0:
                rms = calculate_rms(non_masked)
            else:
                rms = 0.1  # Default if entire audio is masked
            
            # Generate noise at target RMS (adjusted by SNR)
            target_rms = rms / (10**(snr_db / 20))
            noise = np.random.normal(0, target_rms, segment_length)
            masked_audio[start_sample:end_sample, ch] = noise
    else:
        # Mono
        non_masked = np.concatenate([audio[:start_sample], audio[end_sample:]])
        if len(non_masked) > 0:
            rms = calculate_rms(non_masked)
        else:
            rms = 0.1
        
        target_rms = rms / (10**(snr_db / 20))
        noise = np.random.normal(0, target_rms, segment_length)
        masked_audio[start_sample:end_sample] = noise
    
    return masked_audio


def get_mask_range(audio_length: int, percent: float, mode: str, rng: random.Random) -> tuple:
    """
    Calculate start and end sample indices for masking.
    
    Args:
        audio_length: Total number of samples in audio
        percent: Percentage of audio to mask (0-100)
        mode: 'random', 'start', or 'end'
        rng: Random number generator for reproducibility
    
    Returns:
        (start_sample, end_sample)
    """
    mask_length = int(audio_length * percent / 100)
    
    if mask_length == 0:
        return (0, 0)
    
    if mode == 'start':
        return (0, mask_length)
    elif mode == 'end':
        return (audio_length - mask_length, audio_length)
    elif mode == 'random':
        max_start = audio_length - mask_length
        if max_start <= 0:
            return (0, audio_length)
        start = rng.randint(0, max_start)
        return (start, start + mask_length)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def process_dataset(
    source_dir: str,
    output_dir: str,
    mask_type: str,
    mode: str,
    levels: list,
    num_samples: int = None,
    seed: int = 42,
    snr_db: float = 0,
    verbose: bool = False
):
    """
    Process an entire dataset directory, creating masked versions of audio files.
    
    Creates a separate subdirectory for each percentile level.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Find the standardized JSONL file
    jsonl_files = list(source_path.glob('*_standardized.jsonl'))
    if not jsonl_files:
        print(f"ERROR: No '*_standardized.jsonl' file found in '{source_dir}'. Skipping.")
        return
    
    jsonl_file = jsonl_files[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {source_path.name}")
    print(f"  Source JSONL: {jsonl_file.name}")
    print(f"  Mask type: {mask_type}")
    print(f"  Mode: {mode}")
    print(f"  Levels: {levels}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    # Load original data
    original_data = [json.loads(line) for line in open(jsonl_file, 'r')]
    
    if num_samples is not None and num_samples > 0:
        original_data = original_data[:num_samples]
        print(f"  Limited to {num_samples} samples for testing")
    
    # Initialize RNG for reproducibility
    rng = random.Random(seed)
    
    # Create output subdirectory structure: {mask_type}_{mode}/{level}/
    variant_dir = output_path / f"{mask_type}_{mode}"
    
    for level in levels:
        level_dir = variant_dir / str(level)
        audio_dir = level_dir / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        new_jsonl_data = []
        
        for i, item in enumerate(original_data):
            original_audio_path = Path(item['audio_path'])
            
            if not original_audio_path.exists():
                if verbose:
                    print(f"  WARNING: Audio file not found: {original_audio_path}")
                continue
            
            # Load audio
            audio, sample_rate = sf.read(original_audio_path, dtype='float32')
            audio_length = len(audio) if audio.ndim == 1 else audio.shape[0]
            
            # Get mask range
            start, end = get_mask_range(audio_length, level, mode, rng)
            
            # Apply masking
            if mask_type == 'silence':
                masked_audio = mask_audio_with_silence(audio, start, end)
            elif mask_type == 'noise':
                masked_audio = mask_audio_with_noise(audio, start, end, snr_db)
            else:
                raise ValueError(f"Unknown mask type: {mask_type}")
            
            # Save masked audio
            masked_filename = f"{original_audio_path.stem}_masked_{level}pct.wav"
            masked_audio_path = audio_dir / masked_filename
            sf.write(masked_audio_path, masked_audio, sample_rate)
            
            # Create new JSONL entry
            new_item = item.copy()
            new_item['audio_path'] = str(masked_audio_path)
            new_item['original_audio_path'] = str(original_audio_path)
            new_item['mask_type'] = mask_type
            new_item['mask_mode'] = mode
            new_item['mask_percent'] = level
            new_item['mask_start_sample'] = start
            new_item['mask_end_sample'] = end
            new_jsonl_data.append(new_item)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"    Level {level}%: Processed {i+1}/{len(original_data)} samples")
        
        # Write JSONL file for this level
        jsonl_output_path = level_dir / f"{source_path.name}_masked_{level}pct_standardized.jsonl"
        with open(jsonl_output_path, 'w') as f:
            for entry in new_jsonl_data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"  Level {level}%: Generated {len(new_jsonl_data)} masked samples -> {level_dir}")
    
    print(f"\nCompleted processing for {source_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate masked audio datasets at various percentile levels.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help="Path to the source dataset directory (e.g., 'data/mmar' or 'data/sakura/animal')."
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Path to the output directory for masked datasets."
    )
    parser.add_argument(
        '--mask-type',
        type=str,
        required=True,
        choices=['silence', 'noise'],
        help="Type of masking: 'silence' replaces with zeros, 'noise' replaces with white noise."
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['random', 'start', 'end'],
        help="Position mode for masking."
    )
    parser.add_argument(
        '--levels',
        nargs='+',
        type=int,
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="Percentile levels to generate (default: 10 20 30 ... 100)."
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        '--snr',
        type=float,
        default=0,
        help="SNR in dB for noise masking (default: 0, meaning noise at same level as audio)."
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable detailed progress logging."
    )
    
    args = parser.parse_args()
    
    process_dataset(
        source_dir=args.source,
        output_dir=args.output,
        mask_type=args.mask_type,
        mode=args.mode,
        levels=args.levels,
        num_samples=args.num_samples,
        seed=args.seed,
        snr_db=args.snr,
        verbose=args.verbose
    )
    
    print("\nDataset masking complete.")


if __name__ == "__main__":
    main()
