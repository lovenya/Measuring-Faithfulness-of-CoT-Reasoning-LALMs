import os
import re
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from datasets import Audio


# --- Configuration ---
SAKURA_TRACKS = ["Emotion", "Animal", "Language", "Gender"]
BASE_DATASET_ID = "SLLM-multi-hop"
DATASET_SPLIT = "test"
base_output_dir = Path("./data/sakura")


Audio.set_default_backend("soundfile")

# --- Core Parsing Logic ---
def parse_instruction_and_answer(instruction: str, answer_str: str):
    """
    Parses a raw instruction string to extract the question, choices, and answer key.

    Args:
        instruction: The raw string containing the question and choices, 
                     e.g., "What is the emotion? (a) sad (b) fear..."
        answer_str: The string containing the correct answer, e.g., "(a) sad"

    Returns:
        A dictionary {'question': str, 'choices': list, 'answer_key': int}
        or None if parsing fails.
    """
    try:
        # Regex to find all choices, e.g., "(a) some text".
        # It uses a positive lookahead `(?=...)` to correctly handle the last choice.
        choice_pattern = re.compile(r'\(([a-z])\)\s*(.*?)(?=\s*\([a-z]\)\s*|$)', re.DOTALL)
        matches = choice_pattern.findall(instruction)

        if not matches:
            return None

        # The question is the text before the first choice begins.
        first_match_start_index = instruction.find(f"({matches[0][0]})")
        question = instruction[:first_match_start_index].strip()

        # Create the list of choice texts and corresponding letters
        letters = [match[0] for match in matches]
        choices = [match[1].strip() for match in matches]

        # Extract the correct letter from the answer string, e.g., 'a' from "(a) sad"
        correct_letter_match = re.search(r'\(([a-z])\)', answer_str)
        if not correct_letter_match:
            return None
        correct_letter = correct_letter_match.group(1)

        # Find the index of the correct letter to get the answer_key
        answer_key = letters.index(correct_letter)

        return {"question": question, "choices": choices, "answer_key": answer_key}

    except (ValueError, IndexError, AttributeError):
        # Catches errors from `.index()` if letter not found or regex failures.
        return None

# --- Script Execution ---
def main():
    """
    Downloads and processes all four tracks of the SAKURA dataset,
    parsing them into a standardized format with distinct choices and answer keys.
    """
    print("--- SAKURA Dataset Preprocessing Script (with Parsing) ---")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for track in SAKURA_TRACKS:
        print(f"\n{'='*20} Processing Track: {track} {'='*20}")

        dataset_id = f"{BASE_DATASET_ID}/{track}QA"
        track_output_dir = base_output_dir / track.lower()
        output_jsonl_path = track_output_dir / f"sakura_{track.lower()}_test_standardized.jsonl"
        hf_cache_dir = track_output_dir / "hf_cache"

        track_output_dir.mkdir(parents=True, exist_ok=True)
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading dataset '{dataset_id}'...")
        try:
            dataset = load_dataset(dataset_id, split=DATASET_SPLIT, cache_dir=str(hf_cache_dir))
            
            
            
            print("Processing and standardizing data...")
            
            
        except Exception as e:
            print(f"\n--- ERROR on track {track} ---: Failed to load dataset. Skipping. Details: {e}")
            continue
        
        print("Processing and standardizing data...")
        standardized_data = []
        parsing_failures = 0
        
        for item in tqdm(dataset, desc=f"Standardizing {track} samples"):
            audio_path = item.get('audio', {}).get('path')
            if not audio_path:
                continue
            
            base_id = Path(item.get('file', 'unknown_file')).stem

            # --- Process Single-Hop Question ---
            single_parsed = parse_instruction_and_answer(item['single_instruction'], item['single_answer'])
            if single_parsed:
                standardized_data.append({
                    "id": f"{base_id}_single",
                    "audio_path": audio_path,
                    "track": track,
                    "hop_type": "single",
                    **single_parsed
                })
            else:
                parsing_failures += 1
                print(f"Warning: Failed to parse single-hop for ID {base_id}")

            # --- Process Multi-Hop Question ---
            multi_parsed = parse_instruction_and_answer(item['multi_instruction'], item['multi_answer'])
            if multi_parsed:
                standardized_data.append({
                    "id": f"{base_id}_multi",
                    "audio_path": audio_path,
                    "track": track,
                    "hop_type": "multi",
                    **multi_parsed
                })
            else:
                parsing_failures += 1
                print(f"Warning: Failed to parse multi-hop for ID {base_id}")

        print(f"Successfully standardized {len(standardized_data)} records.")
        if parsing_failures > 0:
            print(f"Encountered and skipped {parsing_failures} records due to parsing issues.")

        print(f"Writing standardized data to: {output_jsonl_path}")
        with open(output_jsonl_path, 'w') as f:
            for entry in standardized_data:
                f.write(json.dumps(entry) + '\n')

    print("\n--- All SAKURA Tracks Processed ---")
    print(f"Standardized data saved in subdirectories under: {base_output_dir.resolve()}")

if __name__ == "__main__":
    main()