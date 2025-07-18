import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
# This is the Hugging Face Hub ID for the dataset
DATASET_ID = "BoJack/MMAR"
# Define the split we want to process
DATASET_SPLIT = "test"

# Define output paths within our project structure
# We use Path for robust path handling
output_dir = Path("./data/mmar")
output_jsonl_path = output_dir / "mmar_test_standardized.jsonl"

# Define a specific cache directory within our project for offline dependency management.
# This ensures the downloaded audio files are stored locally with the project.
hf_cache_dir = output_dir / "hf_cache"

# --- Script Execution ---
def main():
    """
    Downloads the MMAR dataset, standardizes its format, and saves it as a 
    JSONL file for offline use by experiment scripts.
    - Converts the 'answer' string to an integer 'answer_key'.
    - Manages file paths for a self-contained project structure.
    """
    print("--- MMAR Dataset Preprocessing Script ---")

    # 1. Create necessary directories
    print(f"Creating output directories at: {output_dir.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    print("Directories created successfully.")

    # 2. Download and load the dataset from Hugging Face
    # Rationale: By specifying cache_dir, we ensure that all downloaded data,
    # including the audio files, is stored in our project folder, making it
    # fully self-contained and ready for offline compute node access.
    print(f"Loading dataset '{DATASET_ID}' (split: '{DATASET_SPLIT}')...")
    print(f"Audio files will be cached in: {hf_cache_dir.resolve()}")
    try:
        dataset = load_dataset(
            DATASET_ID,
            split=DATASET_SPLIT,
            cache_dir=str(hf_cache_dir)
        )
    except Exception as e:
        print("\n--- ERROR ---")
        print(f"Failed to load dataset from Hugging Face.")
        print("Please ensure you are logged in via 'huggingface-cli login' and have internet access.")
        print(f"Error details: {e}")
        return

    print("Dataset loaded successfully.")

    # 3. Process and standardize the data
    print("Processing and standardizing data...")
    standardized_data = []
    for item in tqdm(dataset, desc="Standardizing samples"):
        try:
            # Extract the necessary fields
            choices = item.get('choices')
            answer_str = item.get('answer')

            # Robustness Check: Ensure choices and answer exist and are valid
            if not choices or not isinstance(choices, list) or answer_str is None:
                print(f"Warning: Skipping item with invalid 'choices' or 'answer'. ID: {item.get('id')}")
                continue

            # Transformation: Convert the answer string to an integer index (answer_key)
            # This is a critical step for our analysis framework.
            try:
                answer_key = choices.index(answer_str)
            except ValueError:
                # This handles cases where the provided answer is not in the choices list.
                print(f"Warning: Answer '{answer_str}' not found in choices for ID {item.get('id')}. Skipping.")
                continue

            # The audio path provided by the dataset is what we need.
            # The `datasets` library downloads it to the cache_dir we specified.
            audio_path = item.get('audio_path')
            if not audio_path:
                print(f"Warning: Skipping item with missing 'audio_path'. ID: {item.get('id')}")
                continue

            # Create the standardized dictionary
            standardized_sample = {
                "id": item.get('id'),
                "audio_path": audio_path, # This path is relative to the HF cache
                "question": item.get('question'),
                "choices": choices,
                "answer_key": answer_key
            }
            standardized_data.append(standardized_sample)

        except Exception as e:
            print(f"An unexpected error occurred while processing item {item.get('id')}: {e}")
            continue
            
    print(f"Successfully standardized {len(standardized_data)} samples.")

    # 4. Save the standardized data to a JSONL file
    print(f"Writing standardized data to: {output_jsonl_path}")
    with open(output_jsonl_path, 'w') as f:
        for entry in standardized_data:
            f.write(json.dumps(entry) + '\n')

    print("\n--- Preprocessing Complete ---")
    print(f"Standardized data saved to: {output_jsonl_path.resolve()}")
    print("The project is now ready for running offline experiments with the MMAR dataset.")

if __name__ == "__main__":
    main()