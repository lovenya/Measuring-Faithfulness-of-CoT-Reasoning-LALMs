# data_processing/mask_audio_dataset.py

"""
Generate masked audio datasets at various percentile levels.

This script creates pre-processed masked audio files for the audio masking experiments.
It supports two masking types (silence, noise) and three position modes (random, start, end).

Features:
- Multi-processing for faster processing (--workers)
- Restartable: skips already processed files
- Verbose progress tracking

Usage:
    python mask_audio_dataset.py --source data/mmar --output data/mmar_masked \
        --mask-type silence --mode random --levels 10 20 30 40 50 60 70 80 90 100

    # Use 8 CPU cores for parallel processing:
    python mask_audio_dataset.py --source data/mmar --output data/mmar_masked \
        --mask-type silence --mode random --workers 8

    # Quick test with limited samples:
    python mask_audio_dataset.py --source data/mmar --output data/mmar_masked_test \
        --mask-type silence --mode random --levels 10 50 --num-samples 5 --verbose
"""

import os
import json
import argparse
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

import numpy as np
import soundfile as sf


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate the Root Mean Square of an audio signal."""
    return np.sqrt(np.mean(audio**2))


def mask_audio_with_silence(audio: np.ndarray, start_sample: int, end_sample: int) -> np.ndarray:
    """Replace audio segment with silence (zeros)."""
    masked_audio = audio.copy()
    if audio.ndim == 2:
        masked_audio[start_sample:end_sample, :] = 0.0
    else:
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
        for ch in range(audio.shape[1]):
            non_masked = np.concatenate([audio[:start_sample, ch], audio[end_sample:, ch]])
            rms = calculate_rms(non_masked) if len(non_masked) > 0 else 0.1
            target_rms = rms / (10**(snr_db / 20))
            noise = np.random.normal(0, target_rms, segment_length)
            masked_audio[start_sample:end_sample, ch] = noise
    else:
        non_masked = np.concatenate([audio[:start_sample], audio[end_sample:]])
        rms = calculate_rms(non_masked) if len(non_masked) > 0 else 0.1
        target_rms = rms / (10**(snr_db / 20))
        noise = np.random.normal(0, target_rms, segment_length)
        masked_audio[start_sample:end_sample] = noise
    
    return masked_audio


def get_mask_range(audio_length: int, percent: float, mode: str, seed: int, item_id: str) -> tuple:
    """
    Calculate start and end sample indices for masking.
    Uses item_id + seed to generate reproducible random positions per sample.
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
        # Create reproducible random position per sample
        rng = random.Random(f"{seed}_{item_id}_{percent}")
        start = rng.randint(0, max_start)
        return (start, start + mask_length)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def process_single_sample(item: dict, level: int, mask_type: str, mode: str, 
                          audio_dir: Path, seed: int, snr_db: float) -> dict:
    """
    Process a single audio sample - designed to be called in parallel.
    Returns the new JSONL entry or None if skipped/failed.
    """
    original_audio_path = Path(item['audio_path'])
    item_id = item.get('id', original_audio_path.stem)
    
    # Output path
    masked_filename = f"{original_audio_path.stem}_masked_{level}pct.wav"
    masked_audio_path = audio_dir / masked_filename
    
    # RESTARTABILITY: Skip if already exists
    if masked_audio_path.exists():
        # Return existing entry (reconstruct from existing file)
        new_item = item.copy()
        new_item['audio_path'] = str(masked_audio_path)
        new_item['original_audio_path'] = str(original_audio_path)
        new_item['mask_type'] = mask_type
        new_item['mask_mode'] = mode
        new_item['mask_percent'] = level
        new_item['skipped'] = True  # Mark as skipped for counting
        return new_item
    
    if not original_audio_path.exists():
        return None
    
    try:
        # Load audio
        audio, sample_rate = sf.read(original_audio_path, dtype='float32')
        audio_length = len(audio) if audio.ndim == 1 else audio.shape[0]
        
        # Get mask range (reproducible per sample)
        start, end = get_mask_range(audio_length, level, mode, seed, item_id)
        
        # Apply masking
        if mask_type == 'silence':
            masked_audio = mask_audio_with_silence(audio, start, end)
        elif mask_type == 'noise':
            masked_audio = mask_audio_with_noise(audio, start, end, snr_db)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        # Save masked audio
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
        return new_item
        
    except Exception as e:
        print(f"  ERROR processing {original_audio_path}: {e}")
        return None


def process_dataset(
    source_dir: str,
    output_dir: str,
    mask_type: str,
    mode: str,
    levels: list,
    num_samples: int = None,
    seed: int = 42,
    snr_db: float = 0,
    num_workers: int = 1,
    verbose: bool = False
):
    """
    Process an entire dataset directory, creating masked versions of audio files.
    Supports parallel processing and is restartable.
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
    print(f"  Workers: {num_workers}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    # Load original data
    original_data = [json.loads(line) for line in open(jsonl_file, 'r')]
    
    if num_samples is not None and num_samples > 0:
        original_data = original_data[:num_samples]
        print(f"  Limited to {num_samples} samples for testing")
    
    # Create output subdirectory structure: {mask_type}_{mode}/{level}/
    variant_dir = output_path / f"{mask_type}_{mode}"
    
    for level in levels:
        level_dir = variant_dir / str(level)
        audio_dir = level_dir / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if JSONL already complete (quick restartability check)
        jsonl_output_path = level_dir / f"{source_path.name}_masked_{level}pct_standardized.jsonl"
        if jsonl_output_path.exists():
            existing_count = sum(1 for _ in open(jsonl_output_path))
            if existing_count >= len(original_data):
                print(f"  Level {level}%: Already complete ({existing_count} samples) - SKIPPING")
                continue
        
        # Process samples (parallel or sequential)
        new_jsonl_data = []
        skipped_count = 0
        processed_count = 0
        
        if num_workers > 1:
            # Parallel processing
            process_func = partial(
                process_single_sample,
                level=level,
                mask_type=mask_type,
                mode=mode,
                audio_dir=audio_dir,
                seed=seed,
                snr_db=snr_db
            )
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_func, item): i for i, item in enumerate(original_data)}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        if result.get('skipped'):
                            skipped_count += 1
                            del result['skipped']
                        else:
                            processed_count += 1
                        new_jsonl_data.append(result)
                    
                    if verbose and (processed_count + skipped_count) % 50 == 0:
                        print(f"    Level {level}%: {processed_count + skipped_count}/{len(original_data)} "
                              f"(new: {processed_count}, skipped: {skipped_count})")
        else:
            # Sequential processing
            for i, item in enumerate(original_data):
                result = process_single_sample(
                    item, level, mask_type, mode, audio_dir, seed, snr_db
                )
                if result:
                    if result.get('skipped'):
                        skipped_count += 1
                        del result['skipped']
                    else:
                        processed_count += 1
                    new_jsonl_data.append(result)
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"    Level {level}%: {i+1}/{len(original_data)} "
                          f"(new: {processed_count}, skipped: {skipped_count})")
        
        # Write JSONL file for this level
        with open(jsonl_output_path, 'w') as f:
            for entry in new_jsonl_data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"  Level {level}%: {len(new_jsonl_data)} samples "
              f"(new: {processed_count}, skipped: {skipped_count}) -> {level_dir}")
    
    print(f"\nCompleted processing for {source_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate masked audio datasets at various percentile levels.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--source', type=str, required=True,
        help="Path to the source dataset directory.")
    parser.add_argument('--output', type=str, required=True,
        help="Path to the output directory for masked datasets.")
    parser.add_argument('--mask-type', type=str, required=True, choices=['silence', 'noise'],
        help="Type of masking: 'silence' or 'noise'.")
    parser.add_argument('--mode', type=str, required=True, choices=['random', 'start', 'end'],
        help="Position mode for masking.")
    parser.add_argument('--levels', nargs='+', type=int, default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="Percentile levels to generate (default: 10 20 30 ... 100).")
    parser.add_argument('--num-samples', type=int, default=None,
        help="Limit number of samples to process (for testing).")
    parser.add_argument('--seed', type=int, default=42,
        help="Random seed for reproducibility (default: 42).")
    parser.add_argument('--snr', type=float, default=0,
        help="SNR in dB for noise masking (default: 0).")
    parser.add_argument('--workers', type=int, default=1,
        help="Number of parallel workers (default: 1, use 8+ for speed).")
    parser.add_argument('--verbose', action='store_true',
        help="Enable detailed progress logging.")
    
    args = parser.parse_args()
    
    # Auto-detect CPU count if workers > available
    max_workers = min(args.workers, multiprocessing.cpu_count())
    if args.workers > max_workers:
        print(f"Note: Requested {args.workers} workers, using {max_workers} (available CPUs)")
    
    process_dataset(
        source_dir=args.source,
        output_dir=args.output,
        mask_type=args.mask_type,
        mode=args.mode,
        levels=args.levels,
        num_samples=args.num_samples,
        seed=args.seed,
        snr_db=args.snr,
        num_workers=max_workers,
        verbose=args.verbose
    )
    
    print("\nDataset masking complete.")


if __name__ == "__main__":
    main()
