# MMAC/data_fetch_and_normalisation/download_and_normalize_mmar.py

import argparse
import json
import tarfile
import requests
import shutil
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Import the centralized path from our config file
from config import MMAR_DATASET_PATH

def download_file(url: str, destination: Path):
    """Downloads a file with a progress bar."""
    print(f"Downloading from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        print(f"Successfully downloaded {destination.name}.")
    except Exception as e:
        print(f"FATAL: Failed to download {url}. Error: {e}")
        raise

def normalize_mmar(dataset_name: str, split: str, output_dir: Path):
    """
    Downloads the MMAR dataset archive, extracts audio, and creates a 
    standardized JSONL manifest.
    """
    print(f"MMAC Data Normalization: Starting process for '{dataset_name}' [{split} split].")
    
    # 1. Setup Directory Structure from Config
    output_dir.mkdir(parents=True, exist_ok=True)
    final_audio_dir = output_dir / "audio"
    final_audio_dir.mkdir(exist_ok=True)
    
    # Temporary directory for download and extraction
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    jsonl_path = output_dir / "mmar_standardized.jsonl"
    meta_path = output_dir / "MMAR-meta.json"

    print(f"Output directory set to: {output_dir.resolve()}")

    # --- PHASE 1: DOWNLOAD ARCHIVE ---
    archive_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/mmar-audio.tar.gz"
    local_archive_path = temp_dir / "mmar-audio.tar.gz"
    download_file(archive_url, local_archive_path)

    # --- PHASE 2: EXTRACT ARCHIVE ---
    print(f"Extracting audio archive '{local_archive_path.name}'... This may take a while.")
    extracted_audio_root = temp_dir / "extracted_audio"
    with tarfile.open(local_archive_path, "r:gz") as tar:
        tar.extractall(path=extracted_audio_root)
    print("Extraction complete.")

    # --- PHASE 3: NORMALIZE DATA ---
    try:
        dataset_meta = load_dataset(dataset_name, split=split)
        print("Successfully loaded dataset metadata from Hugging Face.")
    except Exception as e:
        print(f"Error: Failed to load dataset metadata '{dataset_name}'.")
        print(e)
        return

    skipped_count = 0
    total_samples = len(dataset_meta)
    original_metadata = []

    with open(jsonl_path, 'w', encoding='utf-8') as f_jsonl:
        for idx, sample in enumerate(tqdm(dataset_meta, desc="Normalizing MMAR")):
            original_metadata.append(sample)

            try:
                answer, choices = sample['answer'], sample['choices']
                answer_key = choices.index(answer)
            except (ValueError, KeyError, TypeError) as e:
                skipped_count += 1
                print(f"\nWARNING: Skipping sample (ID: {sample.get('id')}) due to answer/choice mismatch. Error: {e}")
                continue

            # Define new audio path and copy/rename the file
            new_audio_filename = f"mmar_audio_{idx}.wav"
            final_audio_path = final_audio_dir / new_audio_filename
            
            # The original path is relative inside the archive, e.g., "./audio/file.wav"
            # We need to construct the full path to the *extracted* file.
            original_relative_path = Path(sample['audio_path'])
            # The extracted folder might have a top-level dir, let's be robust
            # A common pattern is './audio/...' so we find the file in the extracted root.
            source_audio_path = extracted_audio_root / original_relative_path.relative_to(original_relative_path.anchor)

            if source_audio_path.exists():
                shutil.copy(source_audio_path, final_audio_path)
            else:
                skipped_count += 1
                print(f"\nWARNING: Source audio file not found after extraction: {source_audio_path}")
                continue

            normalized_sample = {
                "id": f"mmar_{idx}",
                "original_id": sample['id'],
                "audio_path": str(final_audio_path), # Store the absolute path for clarity
                "question": sample['question'],
                "choices": choices,
                "answer": answer,
                "answer_key": answer_key,
                "modality": sample.get('modality'),
                "category": sample.get('category'),
                "sub-category": sample.get('sub-category'),
                "language": sample.get('language'),
                "source": sample.get('source'),
                "url": sample.get('url'),
                "timestamp": sample.get('timestamp')
            }
            f_jsonl.write(json.dumps(normalized_sample) + '\n')

    with open(meta_path, 'w', encoding='utf-8') as f_meta:
        json.dump(original_metadata, f_meta, indent=2)

    # --- CLEANUP ---
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("Cleanup complete.")

    print("\n--- Normalization Complete ---")
    print(f"Total samples in source: {total_samples}")
    print(f"Successfully processed: {total_samples - skipped_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Standardized JSONL saved to: {jsonl_path.resolve()}")
    print(f"Audio files saved in: {final_audio_dir.resolve()}")
    print("----------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMAC: Download and normalize the MMAR dataset.")
    parser.add_argument("--dataset_name", type=str, default="BoJack/MMAR", help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to process.")
    
    args = parser.parse_args()
    
    # Use the path from the central config file
    normalize_mmar(
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=MMAR_DATASET_PATH
    )