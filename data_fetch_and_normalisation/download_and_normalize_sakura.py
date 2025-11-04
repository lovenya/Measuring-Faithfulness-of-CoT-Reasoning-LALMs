# MMAC/data_fetch_and_normalisation/download_and_normalize_sakura.py

"""
Downloads and normalizes the four SAKURA dataset tracks from Hugging Face.

This script performs several key transformations:
1.  For each track, it loads the source dataset.
2.  It splits each source record into two distinct data points: one for the
    'single-hop' instruction and one for the 'multi-hop' instruction.
3.  The final output is a standardized JSONL file and a corresponding 'audio/'
    folder for each track, with paths managed by a central config.py.
4.  A raw JSON is also a part of output. It contains the original, pre-normalization metadata.
    The idea was to have a backup, just in case.
"""

import json
import re
import soundfile as sf
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Centralized path configuration for the SAKURA dataset's output directory.
from config import SAKURA_DATASET_PATH

# Configuration mapping for the different SAKURA tracks.
SAKURA_TRACKS_CONFIG = {
    "animal": {"hf_id": "SLLM-multi-hop/AnimalQA"},
    "emotion": {"hf_id": "SLLM-multi-hop/EmotionQA"},
    "gender": {"hf_id": "SLLM-multi-hop/GenderQA"},
    "language": {"hf_id": "SLLM-multi-hop/LanguageQA"},
}


def extract_choices_from_instruction(instruction: str) -> list:
    """
    Parses an instruction string to find and extract multiple-choice options.
    Example: "... (a) choice1 (b) choice2" -> ["choice1", "choice2"]
    """
    # This regex looks for patterns like (a) text, (b) text...
    pattern = r'\([a-zA-Z]\)\s*([^()]+?)(?=\s*\([a-zA-Z]\)|\s*$)'
    matches = re.findall(pattern, instruction)
    if matches:
        return [match.strip() for match in matches]
    return []


def extract_answer_text(answer: str) -> str:
    """
    Strips the leading letter and parentheses from an answer string.
    Example: "(b) cat" -> "cat"
    """
    pattern = r'^\([a-zA-Z]\)\s*(.+)$'
    match = re.match(pattern, answer.strip())
    if match:
        return match.group(1).strip()
    return answer.strip()


def normalize_sakura_track(track_name: str, config: dict, base_output_dir: Path):
    """Downloads and normalizes a single SAKURA dataset track."""
    print(f"\n--- Processing SAKURA Track: {track_name.upper()} ---")

    # Set up the directory structure for this specific track.
    track_dir = base_output_dir / track_name
    audio_dir = track_dir / "audio"
    track_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    
    jsonl_path = track_dir / f"sakura_{track_name}_standardized.jsonl"
    meta_path = track_dir / f"SAKURA-{track_name}-meta.json"

    print(f"Output directory: {track_dir.resolve()}")

    # Load the dataset metadata and audio decoders from Hugging Face.
    try:
        dataset = load_dataset(config['hf_id'], split="test", trust_remote_code=True)
        print(f"Successfully loaded '{config['hf_id']}'. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"FATAL: Failed to load dataset for track '{track_name}'. Error: {e}")
        return

    skipped_samples = 0
    original_metadata = []

    with open(jsonl_path, 'w', encoding='utf-8') as f_jsonl:
        for idx, sample in enumerate(tqdm(dataset, desc=f"Normalizing {track_name}")):
            
            # For the metadata backup, we need a JSON-serializable version of the sample.
            # The raw 'audio' field is a special decoder object that can't be saved to JSON,
            # so we create a copy of the sample and remove it before appending.
            serializable_sample = dict(sample)
            serializable_sample.pop('audio', None)
            original_metadata.append(serializable_sample)

            # Process and save the audio file for the current sample.
            try:
                # The 'datasets' library uses lazy loading. Accessing a key like 'array'
                # triggers the actual audio decoding from disk.
                audio_data = sample['audio']
                audio_array = audio_data['array']
                sampling_rate = audio_data['sampling_rate']

                audio_filename = f"sakura_{track_name}_audio_{idx}.wav"
                audio_filepath = audio_dir / audio_filename
                sf.write(audio_filepath, audio_array, sampling_rate)

            except (TypeError, KeyError, AttributeError) as e:
                print(f"\nWarning: Failed to process audio for sample {idx}. Error: {e}. Skipping.")
                skipped_samples += 1
                continue

            # Define the two "hops" (single and multi) to process for each sample.
            hop_types_to_process = [
                ("single", "single_instruction", "single_answer"),
                ("multi", "multi_instruction", "multi_answer"),
            ]

            for hop_type, instruction_key, answer_key_str in hop_types_to_process:
                instruction = sample.get(instruction_key, "")
                answer_str = sample.get(answer_key_str, "")

                if not instruction or not answer_str:
                    continue

                choices = extract_choices_from_instruction(instruction)
                if not choices:
                    print(f"\nWarning: Could not extract choices for '{hop_type}' hop in sample {idx}. Skipping this hop.")
                    continue

                clean_answer = extract_answer_text(answer_str)
                
                try:
                    answer_idx = choices.index(clean_answer)
                except ValueError:
                    print(f"\nWarning: Answer '{clean_answer}' not found in choices {choices} for '{hop_type}' hop in sample {idx}. Skipping this hop.")
                    continue

                # Construct and write the final, standardized data point.
                normalized_sample = {
                    "id": f"sakura_{track_name}_{idx}_{hop_type}",
                    "audio_path": str(audio_filepath),
                    "question": instruction,
                    "choices": choices,
                    "answer": clean_answer,
                    "answer_key": answer_idx,
                    "hop_type": hop_type,
                    "track": track_name,
                }
                f_jsonl.write(json.dumps(normalized_sample) + '\n')

    # Save the original metadata (without audio objects) for traceability.
    with open(meta_path, 'w', encoding='utf-8') as f_meta:
        json.dump(original_metadata, f_meta, indent=2)

    print(f"--- {track_name.upper()} Complete ---")
    print(f"Total source samples: {len(dataset)}")
    print(f"Skipped source samples: {skipped_samples}")
    print(f"Standardized JSONL saved to: {jsonl_path.resolve()}")
    print(f"Audio files saved in: {audio_dir.resolve()}")
    print("--------------------------")


def main():
    """Main function to iterate through and process all configured SAKURA tracks."""
    print("=== MMAC: SAKURA Dataset Normalization Script ===")
    
    # The base output directory is imported from our central config.
    base_output_dir = SAKURA_DATASET_PATH
    print(f"Using configured base output directory: {base_output_dir.resolve()}")

    for track_name, config in SAKURA_TRACKS_CONFIG.items():
        normalize_sakura_track(track_name, config, base_output_dir)

    print("\n✅ All SAKURA tracks processed.")


if __name__ == "__main__":
    main()