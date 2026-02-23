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


def process_single_dataset(source_dir: str, output_dir: str, snr_levels: list[int]):
    """
    Process a single dataset directory: generate noisy audio at all SNR levels.
    
    Args:
        source_dir: Path to source dataset (e.g., data/mmar or data/sakura/animal)
        output_dir: Path to output directory (e.g., data/mmar_noisy or data/sakura_noisy/animal)
        snr_levels: List of SNR levels in dB (e.g., [20, 10, 5, 0, -5, -10])
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
    print(f"{'='*60}")

    # Create output directories
    output_audio_dir = output_path / 'audio'
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    # Load source data
    with open(jsonl_file, 'r') as f:
        original_entries = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(original_entries)} entries")

    # Find unique audio files (Sakura has 500 audios but 1000 entries due to hop types)
    unique_audio_paths = {}
    for entry in original_entries:
        audio_path = entry['audio_path']
        if audio_path not in unique_audio_paths:
            unique_audio_paths[audio_path] = True
    print(f"  Unique audio files: {len(unique_audio_paths)}")

    # Generate noisy audio for each unique audio file
    print(f"\n  Generating noisy audio...")
    generated_files = {}  # (original_path, snr) -> noisy_path
    
    for audio_path_str in tqdm(unique_audio_paths.keys(), desc="  Audio files", unit="file"):
        audio_path = Path(audio_path_str)
        
        if not audio_path.exists():
            print(f"  WARNING: Audio not found: {audio_path}. Skipping.")
            continue

        # Load audio once
        clean_audio, sample_rate = sf.read(str(audio_path), dtype='float32')

        for snr_db in snr_levels:
            # Generate noisy version
            noisy_audio = add_noise_at_snr(clean_audio, snr_db)

            # Save with SNR-tagged filename
            noisy_filename = f"{audio_path.stem}_snr_{snr_db}db.wav"
            noisy_filepath = output_audio_dir / noisy_filename
            sf.write(str(noisy_filepath), noisy_audio, sample_rate)

            generated_files[(audio_path_str, snr_db)] = str(noisy_filepath)

    # Build output JSONL — one entry per (original_entry, snr_level)
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
            process_single_dataset(str(track_dir), str(output_track_dir), args.snr_levels)
    else:
        process_single_dataset(args.source, args.output, args.snr_levels)

    print(f"\n{'='*60}")
    print("All datasets processed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
