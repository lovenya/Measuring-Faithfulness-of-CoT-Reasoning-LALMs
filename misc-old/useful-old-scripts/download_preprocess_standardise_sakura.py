import os
import json
import soundfile as sf
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import shutil
import re

# --- Configuration ---
SAKURA_DATASETS_CONFIG = {
    "animal": {
        "hf_id": "SLLM-multi-hop/AnimalQA",
        "hf_cache": "./data/sakura/animal/hf_cache",
        "standardized_json": "./data/sakura/animal/sakura_animal_test_standardized.jsonl",
        "audio_output": "./data/sakura/animal/audio",
        "split": "test",
        "track": "animal"
    },
    "emotion": {
        "hf_id": "SLLM-multi-hop/EmotionQA",
        "hf_cache": "./data/sakura/emotion/hf_cache",
        "standardized_json": "./data/sakura/emotion/sakura_emotion_test_standardized.jsonl",
        "audio_output": "./data/sakura/emotion/audio",
        "split": "test",
        "track": "emotion"
    },
    "gender": {
        "hf_id": "SLLM-multi-hop/GenderQA",
        "hf_cache": "./data/sakura/gender/hf_cache",
        "standardized_json": "./data/sakura/gender/sakura_gender_test_standardized.jsonl",
        "audio_output": "./data/sakura/gender/audio",
        "split": "test",
        "track": "gender"
    },
    "language": {
        "hf_id": "SLLM-multi-hop/LanguageQA",
        "hf_cache": "./data/sakura/language/hf_cache",
        "standardized_json": "./data/sakura/language/sakura_language_test_standardized.jsonl",
        "audio_output": "./data/sakura/language/audio",
        "split": "test",
        "track": "language"
    }
}

def extract_choices_from_instruction(instruction: str):
    """
    Extract multiple choice options from the instruction text.
    Returns a list of choice texts without the letter prefixes.
    
    Example:
    Input: "Which animal? (a) sheep (b) cat (c) rooster (d) crow"
    Output: ["sheep", "cat", "rooster", "crow"]
    """
    choices = []
    
    # Pattern 1: (a) option (b) option (c) option (d) option
    pattern1 = r'\([a-d]\)\s*([^()]+?)(?=\s*\([a-d]\)|\s*$)'
    matches1 = re.findall(pattern1, instruction)
    
    if matches1:
        choices = [match.strip() for match in matches1]
        return choices
    
    # Pattern 2: a) option b) option c) option d) option
    pattern2 = r'[a-d]\)\s*([^a-d)]+?)(?=\s*[a-d]\)|\s*$)'
    matches2 = re.findall(pattern2, instruction)
    
    if matches2:
        choices = [match.strip() for match in matches2]
        return choices
    
    # Pattern 3: Look for numbered options 1. 2. 3. 4.
    pattern3 = r'[1-4]\.\s*([^1-4.]+?)(?=\s*[1-4]\.|\s*$)'
    matches3 = re.findall(pattern3, instruction)
    
    if matches3:
        choices = [match.strip() for match in matches3]
        return choices
    
    # If no pattern matches, return empty list
    return []

def extract_answer_text(answer: str):
    """
    Extract the text part from an answer that includes the letter prefix.
    
    Example:
    Input: "(b) cat"
    Output: "cat"
    """
    # Pattern to match (letter) text or letter) text
    pattern = r'^\([a-d]\)\s*(.+)$|^[a-d]\)\s*(.+)$'
    match = re.match(pattern, answer.strip())
    
    if match:
        # Return whichever group matched (one will be None)
        return (match.group(1) or match.group(2)).strip()
    
    # If no pattern matches, return the original answer
    return answer.strip()

def download_and_preprocess_sakura_dataset(track_name: str, config: dict):
    """
    Download and preprocess a single Sakura dataset track.
    
    Args:
        track_name: Name of the track (animal, emotion, gender, language)
        config: Configuration dictionary for the dataset
    
    Returns:
        Tuple of (success_count, total_processed, extracted_audio_info)
    """
    print(f"\n--- Processing SAKURA {track_name.upper()} ---")
    
    # Create necessary directories
    output_dir = Path(config["standardized_json"]).parent
    audio_output_dir = Path(config["audio_output"])
    hf_cache_dir = Path(config["hf_cache"])
    
    print(f"Creating output directories...")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and load the dataset
    print(f"Loading dataset '{config['hf_id']}' (split: '{config['split']}')...")
    print(f"Audio files will be cached in: {hf_cache_dir.resolve()}")
    
    try:
        dataset = load_dataset(
            config["hf_id"],
            split=config["split"],
            cache_dir=str(hf_cache_dir)
        )
        print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"ERROR: Failed to load dataset {track_name}")
        print(f"Error details: {e}")
        return 0, 0, []
    
    # Process and standardize the data
    print("Processing and standardizing data...")
    standardized_data = []
    extracted_audio_info = []
    success_count = 0
    
    for i, item in enumerate(tqdm(dataset, desc=f"Processing {track_name}")):
        try:
            # Generate consistent audio filename
            audio_filename = f"sakura_{track_name}_audio_{i}.wav"
            audio_filepath = audio_output_dir / audio_filename
            
            # Extract and save audio
            audio_data = item.get('audio')
            if not audio_data or 'array' not in audio_data or 'sampling_rate' not in audio_data:
                print(f"Warning: No valid audio data for sample {i}. Skipping.")
                continue
                
            audio_array = audio_data['array']
            sampling_rate = audio_data['sampling_rate']
            
            if len(audio_array) == 0:
                print(f"Warning: Empty audio array for sample {i}. Skipping.")
                continue
            
            # Save audio file
            sf.write(str(audio_filepath), audio_array, sampling_rate, format='WAV')
            
            # Store audio extraction info
            extracted_audio_info.append({
                'original_index': i,
                'audio_filename': audio_filename,
                'audio_path': str(audio_filepath.relative_to(Path("./data"))),
                'absolute_path': str(audio_filepath.resolve()),
                'sample_rate': sampling_rate,
                'duration': len(audio_array) / sampling_rate
            })
            
            # Process single-hop question
            single_instruction = item.get('single_instruction', '')
            single_answer = item.get('single_answer', '')
            
            if single_instruction and single_answer:
                # Extract choices from single_instruction
                single_choices = extract_choices_from_instruction(single_instruction)
                if single_choices:
                    # Extract the text part from the answer
                    single_answer_text = extract_answer_text(single_answer)
                    
                    try:
                        single_answer_key = single_choices.index(single_answer_text)
                        
                        single_sample = {
                            "id": f"sakura_{track_name}_{i}_single",
                            "audio_path": str(audio_filepath.relative_to(Path("./data"))),
                            "question": single_instruction,
                            "choices": single_choices,
                            "answer": single_answer_text,
                            "answer_key": single_answer_key,
                            "hop_type": "single",
                            "track": track_name,
                            "modality": "audio",
                            "language": "en",
                            "source": "sakura"
                        }
                        standardized_data.append(single_sample)
                        success_count += 1
                        
                    except ValueError:
                        print(f"Warning: Single answer '{single_answer_text}' not found in choices {single_choices} for sample {i}")
                        print(f"  Original answer: '{single_answer}'")
                        print(f"  Instruction: {single_instruction[:100]}...")
                else:
                    print(f"Warning: Could not extract choices from single instruction for sample {i}")
                    print(f"  Instruction: {single_instruction[:100]}...")
            
            # Process multi-hop question
            multi_instruction = item.get('multi_instruction', '')
            multi_answer = item.get('multi_answer', '')
            
            if multi_instruction and multi_answer:
                # Extract choices from multi_instruction
                multi_choices = extract_choices_from_instruction(multi_instruction)
                if multi_choices:
                    # Extract the text part from the answer
                    multi_answer_text = extract_answer_text(multi_answer)
                    
                    try:
                        multi_answer_key = multi_choices.index(multi_answer_text)
                        
                        multi_sample = {
                            "id": f"sakura_{track_name}_{i}_multi",
                            "audio_path": str(audio_filepath.relative_to(Path("./data"))),
                            "question": multi_instruction,
                            "choices": multi_choices,
                            "answer": multi_answer_text,
                            "answer_key": multi_answer_key,
                            "hop_type": "multi",
                            "track": track_name,
                            "modality": "audio",
                            "language": "en",
                            "source": "sakura"
                        }
                        standardized_data.append(multi_sample)
                        success_count += 1
                        
                    except ValueError:
                        print(f"Warning: Multi answer '{multi_answer_text}' not found in choices {multi_choices} for sample {i}")
                        print(f"  Original answer: '{multi_answer}'")
                        print(f"  Instruction: {multi_instruction[:100]}...")
                else:
                    print(f"Warning: Could not extract choices from multi instruction for sample {i}")
                    print(f"  Instruction: {multi_instruction[:100]}...")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Save standardized data
    print(f"Writing standardized data to: {config['standardized_json']}")
    with open(config["standardized_json"], 'w') as f:
        for entry in standardized_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Successfully processed {success_count} data points from {len(dataset)} audio samples.")
    print(f"Audio files saved to: {audio_output_dir.resolve()}")
    
    return success_count, len(dataset), extracted_audio_info

def verify_sakura_audio_files(all_extracted_info):
    """
    Verify that all extracted Sakura audio files are valid.
    
    Args:
        all_extracted_info: List of all extracted audio info across all tracks
    """
    print("\n--- Sakura Audio Verification ---")
    
    valid_files = 0
    invalid_files = 0
    missing_files = 0
    total_duration = 0
    
    if not all_extracted_info:
        print("No audio files to verify.")
        return
    
    for file_info in tqdm(all_extracted_info, desc="Verifying Sakura audio files"):
        filepath = Path(file_info['absolute_path'])
        
        if not filepath.exists():
            print(f"Warning: Missing audio file: {filepath}")
            missing_files += 1
            continue
        
        try:
            info = sf.info(str(filepath))
            if info.frames == 0:
                print(f"Warning: Empty audio file: {filepath}")
                invalid_files += 1
                continue
            
            valid_files += 1
            total_duration += info.duration
            
        except Exception as e:
            print(f"Warning: Invalid audio file {filepath}: {e}")
            invalid_files += 1
    
    print(f"\nSakura Audio Verification Summary:")
    print(f"  ✓ Valid files: {valid_files}")
    print(f"  ✗ Invalid files: {invalid_files}")
    print(f"  ? Missing files: {missing_files}")
    print(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    if invalid_files > 0 or missing_files > 0:
        print(f"\n⚠️  WARNING: {invalid_files + missing_files} audio files have issues!")
    else:
        print(f"\n✅ All {valid_files} Sakura audio files are valid.")

def main():
    """
    Main function to process all Sakura datasets.
    """
    print("=== Sakura Dataset Preprocessing Script ===")
    print("This script downloads and preprocesses all 4 Sakura dataset tracks:")
    print("- AnimalQA, EmotionQA, GenderQA, LanguageQA")
    print("- Creates standardized JSONL files with proper ID mapping")
    print("- Extracts audio files with consistent naming")
    print("- Generates separate entries for single-hop and multi-hop questions")
    
    # Check if we're in the right directory
    if not Path('./data').exists():
        print("ERROR: Please run this script from the root directory of your project (where the 'data' folder is located).")
        return
    
    all_extracted_info = []
    total_success = 0
    total_processed = 0
    
    # Process each Sakura dataset track
    for track_name, config in SAKURA_DATASETS_CONFIG.items():
        try:
            success_count, processed_count, extracted_info = download_and_preprocess_sakura_dataset(
                track_name, config
            )
            total_success += success_count
            total_processed += processed_count
            all_extracted_info.extend(extracted_info)
            
        except Exception as e:
            print(f"ERROR: Failed to process {track_name} track: {e}")
            continue
    
    # Verify all extracted audio files
    if all_extracted_info:
        verify_sakura_audio_files(all_extracted_info)
    
    # Final summary
    print(f"\n=== Final Summary ===")
    print(f"Total data points created: {total_success}")
    print(f"Total audio samples processed: {total_processed}")
    print(f"Total audio files extracted: {len(all_extracted_info)}")
    
    if total_success > 0:
        print(f"\n✅ SUCCESS: All Sakura datasets have been processed and are ready for use.")
        print("You can now run your training/evaluation scripts using the standardized JSONL files.")
    else:
        print(f"\n❌ FAILED: No data was successfully processed.")
        print("Please check the error messages above and ensure you have proper internet access.")

if __name__ == "__main__":
    main()