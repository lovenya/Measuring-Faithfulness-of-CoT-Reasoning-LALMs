import os
import json
import soundfile as sf # Library for reading/writing audio files
from pathlib import Path # For robust path handling
from datasets import load_dataset # Hugging Face datasets library
from tqdm import tqdm # Progress bar
import shutil # For file operations like copying

# --- Configuration ---
# Define all the datasets and their paths/settings
DATASETS_CONFIG = {
    "mmar": {
        "hf_id": "BoJack/MMAR",
        "hf_cache": "./data/mmar/hf_cache", # Path to the Hugging Face cache for this dataset
        "standardized_json": "./data/mmar/mmar_test_standardized.jsonl", # Path to the standardized JSONL file
        "audio_output": "./data/mmar/audio", # Directory where extracted WAV files will be saved
        "split": "test"
    },
    # "sakura_animal": {
    #     "hf_id": "SLLM-multi-hop/AnimalQA",
    #     "hf_cache": "./data/sakura/animal/hf_cache",
    #     "standardized_json": "./data/sakura/animal/sakura_animal_test_standardized.jsonl",
    #     "audio_output": "./data/sakura/animal/audio",
    #     "split": "test"
    # },
    # "sakura_emotion": {
    #     "hf_id": "SLLM-multi-hop/EmotionQA",
    #     "hf_cache": "./data/sakura/emotion/hf_cache",
    #     "standardized_json": "./data/sakura/emotion/sakura_emotion_test_standardized.jsonl",
    #     "audio_output": "./data/sakura/emotion/audio",
    #     "split": "test"
    # },
    # "sakura_gender": {
    #     "hf_id": "SLLM-multi-hop/GenderQA",
    #     "hf_cache": "./data/sakura/gender/hf_cache",
    #     "standardized_json": "./data/sakura/gender/sakura_gender_test_standardized.jsonl",
    #     "audio_output": "./data/sakura/gender/audio",
    #     "split": "test"
    # },
    # "sakura_language": {
    #     "hf_id": "SLLM-multi-hop/LanguageQA",
    #     "hf_cache": "./data/sakura/language/hf_cache",
    #     "standardized_json": "./data/sakura/language/sakura_language_test_standardized.jsonl",
    #     "audio_output": "./data/sakura/language/audio",
    #     "split": "test"
    # }
}

# --- Core Extraction Function ---
def extract_audio_from_dataset(dataset_name: str, config: dict):
    """
    Extract all audio files from a cached HuggingFace dataset and save them as WAV.
    
    Args:
        dataset_name: Name identifier for the dataset (used for logging).
        config: Configuration dictionary for the dataset (from DATASETS_CONFIG).
    
    Returns:
        A tuple containing:
        - extracted_files: List of dictionaries with info about successfully extracted files.
        - failed_extractions: Count of samples where audio extraction failed.
    """
    print(f"\n--- Processing {dataset_name.upper()} ---")
    
    # Create the dedicated output directory for the extracted audio files
    audio_output_dir = Path(config["audio_output"])
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from the *local cache* specified in the config.
    # This script assumes the dataset has already been downloaded and cached
    # by the previous download_and_preprocess scripts.
    print(f"Attempting to load dataset from cache: {config['hf_cache']}")
    try:
        dataset = load_dataset(
            config["hf_id"], # Use the HF ID to correctly identify the dataset type
            split=config["split"],
            cache_dir=config["hf_cache"] # CRITICAL: Load from our local cache
        )
        print(f"Dataset loaded successfully from cache. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"ERROR: Failed to load dataset {dataset_name} from cache.")
        print(f"Please ensure you have run the download_and_preprocess script for {dataset_name} first.")
        print(f"Error details: {e}")
        return [], 0 # Return empty list and 0 failures if dataset loading fails
    
    # Extract audio files
    print("Extracting audio files...")
    extracted_files = []
    failed_extractions = 0
    
    # Iterate through each sample in the loaded dataset
    for i, sample in enumerate(tqdm(dataset, desc=f"Extracting {dataset_name} audio")):
        try:
            # Get audio data from the 'audio' feature provided by the datasets library
            audio_data = sample.get('audio')
            # Generate a stable sample ID for the output filename
            # Prioritize 'id' if available (like MMAR), otherwise use 'file' stem (like SAKURA)
            sample_id = sample.get('id') or Path(sample.get('file', f"sample_{i}")).stem 
            
            # Basic check for valid audio data structure
            if not audio_data or 'array' not in audio_data or 'sampling_rate' not in audio_data:
                print(f"Warning: No valid audio data structure for sample {sample_id}. Skipping.")
                failed_extractions += 1
                continue
            
            # Define the output filename using the sample ID and ensuring .wav extension
            audio_filename = f"{sample_id}.wav"
            audio_filepath = audio_output_dir / audio_filename
            
            # Get the actual audio data (numpy array) and its sampling rate
            audio_array = audio_data['array']
            sampling_rate = audio_data['sampling_rate']
            
            # Robustness check: Ensure array is not empty
            if len(audio_array) == 0:
                 print(f"Warning: Empty audio array for sample {sample_id}. Skipping.")
                 failed_extractions += 1
                 continue

            # Use soundfile to write the audio data as a WAV file.
            # We save it with the original sample rate from the dataset.
            sf.write(str(audio_filepath), audio_array, sampling_rate, format='WAV')
            
            # Store info about the extracted file.
            # new_audio_path is relative to the project root/data directory for portability.
            extracted_files.append({
                'original_id': sample_id,
                'new_audio_path': str(audio_filepath.relative_to(Path("./data"))), # Relative path for use in JSONL
                'absolute_path': str(audio_filepath.resolve()), # Absolute path for verification
                'sample_rate': sampling_rate,
                'duration': len(audio_array) / sampling_rate # Calculate duration
            })
            
        except Exception as e:
            # Catch any other unexpected errors during extraction for this sample
            print(f"Error extracting audio for sample {sample_id}: {e}")
            failed_extractions += 1
            continue
    
    # Summary statistics for the current dataset's extraction
    print(f"Extraction complete for {dataset_name}:")
    print(f"  Successfully extracted: {len(extracted_files)} files")
    print(f"  Failed extractions: {failed_extractions}")
    if len(extracted_files) > 0:
        print(f"  Audio files saved to: {audio_output_dir.resolve()}")
    
    return extracted_files, failed_extractions

# --- Function to Update Standardized JSONL ---
def update_standardized_json(dataset_name: str, config: dict, extracted_files: list):
    """
    Update the standardized JSONL file to point to the newly extracted audio files.
    
    Args:
        dataset_name: Name identifier for the dataset (for logging).
        config: Configuration dictionary for the dataset.
        extracted_files: List of dictionaries with info about successfully extracted files.
                         (Output of extract_audio_from_dataset).
    """
    json_path = Path(config["standardized_json"])
    
    # Check if the standardized JSONL file exists (created by previous script)
    if not json_path.exists():
        print(f"Warning: Standardized JSONL file not found at {json_path}. Cannot update audio paths.")
        return
    
    # Create a quick lookup map from original ID to the new audio path
    id_to_new_audio_path = {item['original_id']: item['new_audio_path'] for item in extracted_files}
    
    # Read the existing standardized JSONL file line by line
    updated_samples = []
    samples_updated_count = 0
    samples_skipped_count = 0

    print(f"Updating standardized JSONL file: {json_path}")
    with open(json_path, 'r') as f:
        for line in tqdm(f, desc=f"Updating {dataset_name} JSONL"):
            if line.strip(): # Skip empty lines
                try:
                    sample = json.loads(line.strip())
                    sample_id = sample.get('id')
                    
                    if sample_id and sample_id in id_to_new_audio_path:
                        # If we successfully extracted audio for this ID, update the path
                        sample['audio_path'] = id_to_new_audio_path[sample_id]
                        updated_samples.append(sample)
                        samples_updated_count += 1
                    else:
                        # If extraction failed or ID is missing, keep the original entry
                        # and optionally add a status for debugging
                        if sample_id: # Only warn if ID was present but extraction failed
                             print(f"Warning: Could not find extracted audio path for sample ID {sample_id} in {dataset_name} JSONL. Keeping original path or leaving as is.")
                             # Optional: sample['audio_path_status'] = 'extracted_audio_missing'
                        updated_samples.append(sample)
                        samples_skipped_count += 1

                except json.JSONDecodeError:
                     print(f"Warning: Skipping line due to JSON decoding error: {line.strip()}")
                     samples_skipped_count += 1
                except Exception as e:
                     print(f"Warning: Error processing line for update: {line.strip()}. Error: {e}")
                     samples_skipped_count += 1

    # Write the updated data back to the JSONL file.
    # Create a backup first as a safety measure.
    backup_path = json_path.with_suffix('.jsonl.backup')
    try:
        shutil.copy2(json_path, backup_path)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Warning: Failed to create backup of {json_path}. Error: {e}")

    try:
        with open(json_path, 'w') as f:
            for sample in updated_samples:
                f.write(json.dumps(sample) + '\n')
        print(f"Updated standardized JSONL file: {json_path}")
        print(f"  Successfully updated audio paths for {samples_updated_count} samples.")
        print(f"  Skipped or failed to find paths for {samples_skipped_count} samples (check warnings).")

    except Exception as e:
         print(f"FATAL ERROR: Failed to write updated JSONL file {json_path}. Error: {e}")
         print(f"The backup file {backup_path} might be your only remaining copy.")


# --- Function to Verify Extracted Audio Files ---
def verify_extracted_audio(all_extracted_files_info: list):
    """
    Verify that extracted audio files are valid and provide dataset statistics.
    Designed to be run after extraction from ALL datasets is complete.
    
    Args:
        all_extracted_files_info: List of dictionaries with info about all successfully 
                                  extracted files across all datasets.
                                  (Combined output of extract_audio_from_dataset calls).
    """
    print("\n--- Audio Verification & Statistics ---")
    
    valid_files = 0
    invalid_files = 0 # Files that exist but fail validation (e.g., empty, corrupt)
    missing_files = 0 # Files that were expected but don't exist on disk
    total_duration = 0
    sample_rates = {} # To count distribution of sample rates
    duration_stats = [] # List of durations for min/max/avg calculation
    
    if not all_extracted_files_info:
        print("No files were extracted across all datasets to verify.")
        return 0, 0, 0 # valid, invalid, missing

    # Verify each file in the combined list
    for file_info in tqdm(all_extracted_files_info, desc="Verifying extracted audio files"):
        # Use the absolute path for verification
        filepath = Path(file_info.get('absolute_path')) 
        original_id = file_info.get('original_id', 'N/A')
        
        # Check 1: Does the file exist?
        if not filepath or not filepath.exists():
            print(f"Warning: Missing audio file on disk for ID {original_id}. Path: {filepath}")
            missing_files += 1
            continue # Skip to the next file

        try:
            # Check 2: Try to read the file to ensure it's a valid audio file
            # Use soundfile.info for a quicker check than sf.read for validation
            info = sf.info(str(filepath))
            
            # Check 3: Basic validation checks from info
            if info.frames == 0:
                print(f"Warning: Empty audio file (0 frames) for ID {original_id}. Path: {filepath}")
                invalid_files += 1
                continue

            if info.samplerate <= 0:
                 print(f"Warning: Invalid sample rate ({info.samplerate}) for ID {original_id}. Path: {filepath}")
                 invalid_files += 1
                 continue
            
            # If all checks pass, the file is valid
            valid_files += 1
            duration = info.duration # Use info.duration which is often more efficient
            total_duration += duration
            duration_stats.append(duration)
            
            # Track sample rate distribution
            sample_rates[info.samplerate] = sample_rates.get(info.samplerate, 0) + 1
            
        except Exception as e:
            # Catch any errors during soundfile.info or other checks
            print(f"Warning: File appears corrupt or invalid for ID {original_id}. Path: {filepath}. Error: {e}")
            invalid_files += 1
    
    # --- Print Verification Results ---
    print("\nVerification Summary:")
    print(f"  Total files processed for verification: {len(all_extracted_files_info)}")
    print(f"  ✓ Valid files: {valid_files}")
    print(f"  ✗ Invalid (Corrupt/Empty) files: {invalid_files}")
    print(f"  ? Missing files (not found): {missing_files}") # Highlight missing files

    # Calculate and print duration statistics if there are valid files
    if duration_stats:
        print(f"\nValid Audio Duration Statistics:")
        # Ensure accurate duration calculation
        avg_duration = sum(duration_stats) / len(duration_stats) if duration_stats else 0
        min_duration = min(duration_stats) if duration_stats else 0
        max_duration = max(duration_stats) if duration_stats else 0
        print(f"  Average duration: {avg_duration:.2f} seconds")
        print(f"  Min duration: {min_duration:.2f} seconds")
        print(f"  Max duration: {max_duration:.2f} seconds")
        print(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)") # Total for valid files

    # Print sample rate distribution if there are valid files
    if sample_rates:
        print(f"\nValid Audio Sample Rate Distribution:")
        for sr, count in sorted(sample_rates.items()):
            print(f"  {sr} Hz: {count} files")
    
    # Final overall status alert
    if invalid_files > 0 or missing_files > 0:
        print(f"\n⚠️  WARNING: Encountered issues with {invalid_files + missing_files} audio files!")
        print("   Please check the 'Warning' messages above before running compute jobs.")
    else:
        print(f"\n✅ All {valid_files} extracted audio files passed verification.")
    
    return valid_files, invalid_files, missing_files

# --- Main Execution Logic ---
def main():
    """
    Main function to extract and verify audio from all configured datasets.
    """
    print("=== Audio Extraction and Verification Script ===")
    print("This script extracts audio files from cached HuggingFace datasets,")
    print("saves them to simple directories, updates standardized JSONL files,")
    print("and verifies the extracted audio integrity.")
    
    all_extracted_files_info = [] # Collect info from all datasets
    total_failed_extractions = 0 # Sum of extraction failures across all datasets
    
    # Process each dataset defined in the configuration
    for dataset_name, config in DATASETS_CONFIG.items():
        # Run the audio extraction process for the current dataset
        extracted_files_info_dataset, failed_count_dataset = extract_audio_from_dataset(dataset_name, config)
        
        # If extraction was successful for any files in this dataset:
        if extracted_files_info_dataset:
            # Update the corresponding standardized JSONL file
            update_standardized_json(dataset_name, config, extracted_files_info_dataset)
            # Add the info about these extracted files to our overall list
            all_extracted_files_info.extend(extracted_files_info_dataset)
        
        # Keep track of total failures across all datasets
        total_failed_extractions += failed_count_dataset
    
    # --- Overall Verification Step ---
    # Verify all extracted files from all datasets at once
    if all_extracted_files_info:
        valid_count, invalid_count, missing_count = verify_extracted_audio(all_extracted_files_info)
        
        # Final conclusion based on overall verification results
        if invalid_count > 0 or missing_count > 0:
            print(f"\n⚠️  SUMMARY: {invalid_count + missing_count} audio files have issues.")
            print("   Review the logs and verification report.")
        else:
            print(f"\n✅ Overall SUCCESS: All extracted audio files are valid and ready.")
    else:
        print("\nNo audio files were extracted from any configured dataset.")
        print("Please check the dataset loading steps and cache directories.")
        
    # Final script completion message
    print("\n=== Script Finished ===")
    print("Review the output log for details and any warnings/errors.")
    print("Proceed to experiment runs using the updated JSONL files pointing to the new audio directories.")


# Entry point when the script is run directly
if __name__ == "__main__":
    # Ensure we are running from the project root by checking for a known dir
    # This is a basic safety check to ensure relative paths work correctly
    if not Path('./data').exists():
         print("ERROR: Please run this script from the root directory of your project (where the 'data' folder is located).")
    else:
        main()