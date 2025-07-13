import os
import json
import soundfile as sf
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import shutil

# --- Configuration ---
DATASETS_CONFIG = {
    "mmar": {
        "hf_id": "BoJack/MMAR",
        "hf_cache": "./data/mmar/hf_cache",
        "standardized_json": "./data/mmar/mmar_test_standardized.jsonl",
        "audio_output": "./data/mmar/audio",
        "split": "test"
    },
    "sakura_animal": {
        "hf_id": "SLLM-multi-hop/AnimalQA", 
        "hf_cache": "./data/sakura/animal/hf_cache",
        "standardized_json": "./data/sakura/animal/sakura_animal_test_standardized.jsonl",
        "audio_output": "./data/sakura/animal/audio",
        "split": "test"
    },
    "sakura_emotion": {
        "hf_id": "SLLM-multi-hop/EmotionQA",
        "hf_cache": "./data/sakura/emotion/hf_cache", 
        "standardized_json": "./data/sakura/emotion/sakura_emotion_test_standardized.jsonl",
        "audio_output": "./data/sakura/emotion/audio",
        "split": "test"
    },
    "sakura_gender": {
        "hf_id": "SLLM-multi-hop/GenderQA",
        "hf_cache": "./data/sakura/gender/hf_cache",
        "standardized_json": "./data/sakura/gender/sakura_gender_test_standardized.jsonl", 
        "audio_output": "./data/sakura/gender/audio",
        "split": "test"
    },
    "sakura_language": {
        "hf_id": "SLLM-multi-hop/LanguageQA",
        "hf_cache": "./data/sakura/language/hf_cache",
        "standardized_json": "./data/sakura/language/sakura_language_test_standardized.jsonl",
        "audio_output": "./data/sakura/language/audio", 
        "split": "test"
    }
}

def extract_audio_from_dataset(dataset_name, config):
    """
    Extract all audio files from a cached HuggingFace dataset.
    
    Args:
        dataset_name: Name identifier for the dataset
        config: Configuration dictionary for the dataset
    """
    print(f"\n--- Processing {dataset_name.upper()} ---")
    
    # Create output directory
    audio_output_dir = Path(config["audio_output"])
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from cache
    print(f"Loading dataset from cache: {config['hf_cache']}")
    try:
        dataset = load_dataset(
            config["hf_id"], 
            split=config["split"], 
            cache_dir=config["hf_cache"]
        )
        print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"ERROR: Failed to load dataset {dataset_name}")
        print(f"Error details: {e}")
        return [], 0
    
    # Extract audio files
    print("Extracting audio files...")
    extracted_files = []
    failed_extractions = 0
    
    for i, sample in enumerate(tqdm(dataset, desc=f"Extracting {dataset_name} audio")):
        try:
            # Get audio data
            audio_data = sample.get('audio')
            sample_id = sample.get('id', f"{dataset_name}_{i}")
            
            if not audio_data or 'array' not in audio_data or 'sampling_rate' not in audio_data:
                print(f"Warning: No valid audio data for sample {sample_id}")
                failed_extractions += 1
                continue
            
            # Define output filename - ensure .wav extension
            audio_filename = f"{sample_id}.wav"
            audio_filepath = audio_output_dir / audio_filename
            
            # Extract and save audio as .wav format
            audio_array = audio_data['array']
            sampling_rate = audio_data['sampling_rate']
            
            # Save as .wav with original sample rate (no resampling)
            sf.write(str(audio_filepath), audio_array, sampling_rate, format='WAV')
            
            # Store info for JSON update
            extracted_files.append({
                'original_id': sample_id,
                'new_audio_path': str(audio_filepath.relative_to(Path("./data"))),  # Relative path from data/
                'absolute_path': str(audio_filepath.resolve()),
                'sample_rate': sampling_rate,
                'duration': len(audio_array) / sampling_rate
            })
            
        except Exception as e:
            print(f"Error extracting audio for sample {sample_id}: {e}")
            failed_extractions += 1
            continue
    
    print(f"Extraction complete for {dataset_name}:")
    print(f"  Successfully extracted: {len(extracted_files)} files")
    print(f"  Failed extractions: {failed_extractions}")
    print(f"  Audio files saved to: {audio_output_dir.resolve()}")
    
    return extracted_files, failed_extractions

def update_standardized_json(dataset_name, config, extracted_files):
    """
    Update the standardized JSON file to point to extracted audio files.
    
    Args:
        dataset_name: Name identifier for the dataset
        config: Configuration dictionary for the dataset
        extracted_files: List of extracted file info
    """
    json_path = Path(config["standardized_json"])
    
    if not json_path.exists():
        print(f"Warning: Standardized JSON not found at {json_path}")
        return
    
    # Create mapping from ID to new audio path
    id_to_audio_path = {item['original_id']: item['new_audio_path'] for item in extracted_files}
    
    # Read existing JSON
    updated_samples = []
    with open(json_path, 'r') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line.strip())
                sample_id = sample.get('id')
                
                if sample_id in id_to_audio_path:
                    # Update audio path
                    sample['audio_path'] = id_to_audio_path[sample_id]
                    updated_samples.append(sample)
                else:
                    print(f"Warning: No extracted audio found for sample {sample_id}")
                    # Keep original entry but mark as problematic
                    sample['audio_path_status'] = 'extraction_failed'
                    updated_samples.append(sample)
    
    # Write updated JSON
    backup_path = json_path.with_suffix('.jsonl.backup')
    shutil.copy2(json_path, backup_path)
    print(f"Backup created: {backup_path}")
    
    with open(json_path, 'w') as f:
        for sample in updated_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Updated standardized JSON: {json_path}")
    print(f"  Updated {len([s for s in updated_samples if 'audio_path_status' not in s])} samples")

def verify_extracted_audio(extracted_files):
    """
    Verify that extracted audio files are valid and provide dataset statistics.
    
    Args:
        extracted_files: List of extracted file info
    """
    print("\n--- Audio Verification & Statistics ---")
    
    valid_files = 0
    invalid_files = 0
    total_duration = 0
    sample_rates = {}
    duration_stats = []
    
    for file_info in tqdm(extracted_files, desc="Verifying audio files"):
        try:
            filepath = Path(file_info['absolute_path'])
            
            # Check if file exists
            if not filepath.exists():
                print(f"Missing file: {filepath}")
                invalid_files += 1
                continue
            
            # Try to read the file to ensure it's valid
            data, samplerate = sf.read(str(filepath))
            
            # Basic validation checks
            if len(data) == 0:
                print(f"Empty audio file: {filepath}")
                invalid_files += 1
                continue
            
            if samplerate <= 0:
                print(f"Invalid sample rate {samplerate} for file: {filepath}")
                invalid_files += 1
                continue
            
            # File is valid
            valid_files += 1
            duration = len(data) / samplerate
            total_duration += duration
            duration_stats.append(duration)
            
            # Track sample rate distribution
            sample_rates[samplerate] = sample_rates.get(samplerate, 0) + 1
            
        except Exception as e:
            print(f"Invalid audio file {filepath}: {e}")
            invalid_files += 1
    
    # Print verification results
    print(f"Verification Results:")
    print(f"  ✓ Valid files: {valid_files}")
    print(f"  ✗ Invalid files: {invalid_files}")
    print(f"  Total audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    if duration_stats:
        print(f"\nDuration Statistics:")
        print(f"  Average duration: {sum(duration_stats)/len(duration_stats):.2f} seconds")
        print(f"  Min duration: {min(duration_stats):.2f} seconds")
        print(f"  Max duration: {max(duration_stats):.2f} seconds")
    
    if sample_rates:
        print(f"\nSample Rate Distribution:")
        for sr, count in sorted(sample_rates.items()):
            print(f"  {sr} Hz: {count} files")
    
    # Alert if there are issues
    if invalid_files > 0:
        print(f"\n⚠️  WARNING: {invalid_files} files failed verification!")
        print("   Check the error messages above and consider re-running extraction.")
    else:
        print(f"\n✓ All {valid_files} audio files verified successfully!")
    
    return valid_files, invalid_files

def main():
    """
    Main function to extract audio from all configured datasets.
    """
    print("=== Audio Extraction Script ===")
    print("This script will extract all audio files from cached HuggingFace datasets")
    print("and update your standardized JSON files accordingly.")
    
    all_extracted_files = []
    total_failed = 0
    
    # Process each dataset
    for dataset_name, config in DATASETS_CONFIG.items():
        extracted_files, failed_count = extract_audio_from_dataset(dataset_name, config)
        
        if extracted_files:
            # Update corresponding JSON file
            update_standardized_json(dataset_name, config, extracted_files)
            all_extracted_files.extend(extracted_files)
        
        total_failed += failed_count
    
    # Overall verification
    if all_extracted_files:
        valid_count, invalid_count = verify_extracted_audio(all_extracted_files)
        
        # Final status check
        if invalid_count > 0:
            print(f"\n⚠️  ATTENTION: {invalid_count} files failed verification!")
            print("Please check the errors above before proceeding with experiments.")
        else:
            print(f"\n✅ SUCCESS: All {valid_count} audio files extracted and verified!")
    
    # Summary
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Total files extracted: {len(all_extracted_files)}")
    print(f"Total failed extractions: {total_failed}")
    print(f"Datasets processed: {list(DATASETS_CONFIG.keys())}")
    
    if all_extracted_files and total_failed == 0:
        print("\n🎉 All datasets processed successfully!")
        print("Your audio files are now ready for inference experiments.")
    
    print("\nNext steps:")
    print("1. Verify a few extracted audio files manually (optional)")
    print("2. Update your inference scripts to use the new audio paths")
    print("3. Run your faithfulness experiments!")
    print("4. Consider backing up the extracted audio files")

if __name__ == "__main__":
    main()