# data_processing/generate_noisy_audio.py

"""
Generate noisy versions of audio datasets at specified SNR levels.

For each audio file in a dataset, generates noisy versions by overlaying
white Gaussian noise calibrated to achieve a target SNR (Signal-to-Noise Ratio).

Supports:
  - MMAR dataset: data/mmar/ -> data/mmar_noisy/
  - Sakura datasets: data/sakura/{track}/ -> data/sakura_noisy/{track}/
  - All 4 Sakura tracks at once via --source data/sakura

Output structure:
  data/{dataset}_noisy/
    audio/
      {original_stem}_snr_{level}db.wav
    {dataset}_noisy_standardized.jsonl

Usage:
    python data_processing/generate_noisy_audio.py --source data/mmar --output data/mmar_noisy
    python data_processing/generate_noisy_audio.py --source data/sakura --output data/sakura_noisy
    python data_processing/generate_noisy_audio.py --source data/sakura/animal --output data/sakura_noisy/animal
"""

import os
import json
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import functools


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate Root Mean Square of an audio signal."""
    return np.sqrt(np.mean(audio ** 2))


def add_noise_at_snr(clean_audio: np.ndarray, snr_db: int) -> np.ndarray:
    """
    Overlay white Gaussian noise on clean audio to achieve target SNR.
    
    Handles both mono (1D) and stereo (2D) audio.
    SNR = 20 * log10(RMS_signal / RMS_noise)
    """
    if clean_audio.ndim == 2:
        # Stereo: process each channel independently
        return np.stack(
            [add_noise_at_snr(clean_audio[:, ch], snr_db) for ch in range(clean_audio.shape[1])],
            axis=1
        )

    # Mono processing
    rms_clean = calculate_rms(clean_audio)
    if rms_clean == 0:
        return clean_audio  # Silent audio, nothing to do

    # Calculate required noise RMS for target SNR
    rms_noise_target = rms_clean / (10 ** (snr_db / 20))

    # Generate and scale white noise
    noise = np.random.normal(0, 1, len(clean_audio))
    rms_noise_current = calculate_rms(noise)
    if rms_noise_current == 0:
        return clean_audio

    scaled_noise = noise * (rms_noise_target / rms_noise_current)
    return clean_audio + scaled_noise


def _process_audio_snr_pair(args_tuple):
    """
    Top-level worker function (must be at module level to be picklable for Pool).
    Generates one noisy audio file for a given (audio_path, snr_db, output_dir).
    Returns (audio_path_str, snr_db, noisy_filepath_str) on success, or None on failure.
    Skips if the output file already exists (restart-safe).
    """
    audio_path_str, snr_db, output_audio_dir_str = args_tuple
    audio_path = Path(audio_path_str)
    output_audio_dir = Path(output_audio_dir_str)

    noisy_filename = f"{audio_path.stem}_snr_{snr_db}db.wav"
    noisy_filepath = output_audio_dir / noisy_filename

    # Already done — restart-safe
    if noisy_filepath.exists():
        return (audio_path_str, snr_db, str(noisy_filepath))

    if not audio_path.exists():
        return None

    try:
        clean_audio, sample_rate = sf.read(str(audio_path), dtype='float32')
        noisy_audio = add_noise_at_snr(clean_audio, snr_db)
        sf.write(str(noisy_filepath), noisy_audio, sample_rate)
        return (audio_path_str, snr_db, str(noisy_filepath))
    except Exception as e:
        print(f"  WARNING: Failed {audio_path.name} @ {snr_db}dB: {e}")
        return None


def process_single_dataset(source_dir: str, output_dir: str, snr_levels: list[int], num_workers: int = 1):
    """
    Process a single dataset directory: generate noisy audio at all SNR levels.
    
    Args:
        source_dir:  Path to source dataset (e.g., data/mmar or data/sakura/animal)
        output_dir:  Path to output directory
        snr_levels:  List of SNR levels in dB
        num_workers: Number of parallel worker processes (default: 1)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Find the standardized JSONL file
    jsonl_candidates = list(source_path.glob('*_standardized.jsonl'))
    if not jsonl_candidates:
        print(f"ERROR: No '*_standardized.jsonl' file found in '{source_dir}'. Skipping.")
        return
    jsonl_file = jsonl_candidates[0]

    print(f"\n{'='*60}")
    print(f"Processing: {source_path.name}")
    print(f"  Source JSONL: {jsonl_file.name}")
    print(f"  Output dir:  {output_path}")
    print(f"  SNR levels:  {snr_levels} dB")
    print(f"  Workers:     {num_workers}")
    print(f"{'='*60}")

    # Create output directories
    output_audio_dir = output_path / 'audio'
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    # Load source data
    with open(jsonl_file, 'r') as f:
        original_entries = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(original_entries)} entries")

    # Find unique audio files
    unique_audio_paths = list(dict.fromkeys(
        entry['audio_path'] for entry in original_entries
    ))
    print(f"  Unique audio files: {len(unique_audio_paths)}")

    # Build all (audio_path, snr_db, output_dir) work items
    work_items = [
        (audio_path_str, snr_db, str(output_audio_dir))
        for audio_path_str in unique_audio_paths
        for snr_db in snr_levels
    ]
    total = len(work_items)
    print(f"  Total work items:   {total} ({len(unique_audio_paths)} files × {len(snr_levels)} SNR levels)")

    # Run in parallel or single-threaded
    generated_files = {}  # (original_path, snr) -> noisy_path
    print(f"\n  Generating noisy audio...")

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            for result in tqdm(
                pool.imap(_process_audio_snr_pair, work_items, chunksize=4),
                total=total, desc="  Noisy files", unit="file"
            ):
                if result:
                    generated_files[(result[0], result[1])] = result[2]
    else:
        for item in tqdm(work_items, desc="  Noisy files", unit="file"):
            result = _process_audio_snr_pair(item)
            if result:
                generated_files[(result[0], result[1])] = result[2]

    # Build output JSONL
    new_entries = []
    for entry in original_entries:
        for snr_db in snr_levels:
            noisy_path = generated_files.get((entry['audio_path'], snr_db))
            if not noisy_path:
                continue

            new_entry = entry.copy()
            new_entry['snr_db'] = snr_db
            new_entry['original_audio_path'] = entry['audio_path']
            new_entry['audio_path'] = noisy_path
            new_entries.append(new_entry)

    # Write output JSONL
    dataset_name = source_path.name
    output_jsonl = output_path / f"{dataset_name}_noisy_standardized.jsonl"
    with open(output_jsonl, 'w') as f:
        for entry in new_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\n  ✓ Generated {len(generated_files)} noisy audio files")
    print(f"  ✓ Created {len(new_entries)} JSONL entries")
    print(f"  ✓ Output JSONL: {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate noisy audio datasets at specified SNR levels.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--source', type=str, required=True,
                        help="Source dataset directory.\n"
                             "  Single: data/mmar or data/sakura/animal\n"
                             "  All Sakura: data/sakura (auto-detects all tracks)")
    parser.add_argument('--output', type=str, required=True,
                        help="Output directory for noisy dataset.\n"
                             "  Single: data/mmar_noisy or data/sakura_noisy/animal\n"
                             "  All Sakura: data/sakura_noisy")
    parser.add_argument('--snr-levels', nargs='+', type=int,
                        default=[20, 10, 5, 0, -5, -10],
                        help="SNR levels in dB (default: 20 10 5 0 -5 -10)")
    parser.add_argument('--num-workers', type=int, default=1,
                        help="Number of parallel worker processes (default: 1).\n"
                             "Set to match --cpus-per-task in your sbatch/salloc, e.g. 8.")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)

    if source_path.name == 'sakura':
        # Process all Sakura tracks
        tracks = [d for d in sorted(source_path.iterdir()) 
                  if d.is_dir() and not d.name.endswith('_masked')]
        print(f"Detected Sakura mode: found {len(tracks)} tracks: {[t.name for t in tracks]}")
        for track_dir in tracks:
            output_track_dir = output_path / track_dir.name
            process_single_dataset(str(track_dir), str(output_track_dir), args.snr_levels, args.num_workers)
    else:
        process_single_dataset(args.source, args.output, args.snr_levels, args.num_workers)

    print(f"\n{'='*60}")
    print("All datasets processed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
