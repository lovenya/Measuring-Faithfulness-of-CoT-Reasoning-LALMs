# data_processing/add_noise_to_dataset.py

import os
import numpy as np
import soundfile as sf
import json
import argparse
from pathlib import Path

def calculate_rms(audio):
    """Calculates the Root Mean Square of an audio signal."""
    return np.sqrt(np.mean(audio**2))

# --- UPDATED FUNCTION TO HANDLE STEREO AUDIO ---
def add_noise_with_snr(clean_audio, snr_db):
    """
    Adds white noise to a clean audio signal to achieve a specific SNR.
    This version is robust to both mono and stereo audio.
    """
    # Check if the audio is stereo
    if clean_audio.ndim == 2:
        # Handle stereo: process each channel independently
        noisy_channels = []
        for i in range(clean_audio.shape[1]):
            channel = clean_audio[:, i]
            noisy_channel = add_noise_with_snr(channel, snr_db) # Recursive call for mono channel
            noisy_channels.append(noisy_channel)
        # Stack the noisy channels back into a stereo signal
        return np.stack(noisy_channels, axis=1)

    # --- Original logic for MONO audio ---
    rms_clean = calculate_rms(clean_audio)
    if rms_clean == 0:
        return clean_audio

    rms_noise = rms_clean / (10**(snr_db / 20))
    noise = np.random.normal(0, 1, len(clean_audio))
    rms_current_noise = calculate_rms(noise)
    
    # Handle case where noise is silent to avoid division by zero
    if rms_current_noise == 0:
        return clean_audio
        
    scaled_noise = noise * (rms_noise / rms_current_noise)
    noisy_audio = clean_audio + scaled_noise
    return noisy_audio
# --- END OF UPDATE ---


def process_dataset(source_dir: str, output_dir: str, snr_levels: list):
    """
    Processes an entire dataset directory, creating noisy versions of audio
    files and a new standardized JSONL file.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    try:
        jsonl_file = next(source_path.glob('*_standardized.jsonl'))
    except StopIteration:
        print(f"ERROR: No '*_standardized.jsonl' file found in '{source_dir}'. Skipping.")
        return

    print(f"\nProcessing dataset: {source_path.name}")
    print(f"  - Source JSONL: {jsonl_file.name}")
    print(f"  - Output directory: {output_path}")

    output_audio_path = output_path / 'audio'
    output_audio_path.mkdir(parents=True, exist_ok=True)

    new_jsonl_data = []
    original_data = [json.loads(line) for line in open(jsonl_file, 'r')]

    for i, item in enumerate(original_data):
        clean_audio_path = Path(item['audio_path'])
        
        if not clean_audio_path.exists():
            print(f"  - WARNING: Source audio file not found, skipping: {clean_audio_path}")
            continue
            
        print(f"  - Processing audio {i+1}/{len(original_data)}: {clean_audio_path.name}")

        clean_audio, sample_rate = sf.read(clean_audio_path, dtype='float32')

        for snr in snr_levels:
            noisy_audio = add_noise_with_snr(clean_audio, snr)
            
            noisy_filename = f"{clean_audio_path.stem}_snr_{snr}db.wav"
            output_file_path = output_audio_path / noisy_filename
            
            sf.write(output_file_path, noisy_audio, sample_rate)

            new_item = item.copy()
            new_item['audio_path'] = str(output_file_path)
            new_item['snr_db'] = snr
            new_item['original_audio_path'] = str(clean_audio_path)
            new_item['id'] = f"{item['id']}_snr_{snr}"
            new_jsonl_data.append(new_item)

    output_jsonl_path = output_path / f"{source_path.name}_noisy_standardized.jsonl"
    with open(output_jsonl_path, 'w') as f:
        for entry in new_jsonl_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"  - Successfully generated {len(new_jsonl_data)} noisy audio samples.")
    print(f"  - New JSONL file created at: {output_jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a new dataset with noisy audio at specified SNR levels.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--source', 
        type=str, 
        required=True,
        help="Path to the source dataset directory (e.g., 'data/sakura/animal')."
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help="Path to the output directory for the new noisy dataset (e.g., 'data/sakura_noisy/animal')."
    )
    parser.add_argument(
        '--snr-levels', 
        nargs='+', 
        type=int, 
        default=[20, 10, 5, 0, -5, -10],
        help="A list of SNR levels (in dB) to generate. Example: --snr-levels 20 10 0 -10"
    )
    args = parser.parse_args()

    # --- Logic to handle multiple source directories ---
    # This allows you to process all sakura tracks with one command.
    source_base = Path(args.source)
    output_base = Path(args.output)
    
    if source_base.name == 'sakura':
        # If the source is the main 'sakura' folder, find all sub-tracks.
        sub_dirs = [d for d in source_base.iterdir() if d.is_dir()]
        for sub_dir in sub_dirs:
            output_sub_dir = output_base / sub_dir.name
            process_dataset(str(sub_dir), str(output_sub_dir), args.snr_levels)
    else:
        # Otherwise, process the single specified directory.
        process_dataset(args.source, args.output, args.snr_levels)

    print("\nDataset processing complete.")