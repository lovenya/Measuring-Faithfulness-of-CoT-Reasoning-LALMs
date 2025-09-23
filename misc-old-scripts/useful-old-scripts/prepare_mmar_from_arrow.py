import json
from pathlib import Path
from datasets import load_dataset  # Correctly use load_dataset
from tqdm import tqdm

# --- Configuration ---
# Point to the top-level dataset directory within the cache
# This directory should contain the versioned folders (e.g., 'default/0.0.0/...')
MMAR_CACHE_DIR = "./data/mmar/hf_cache/BoJack___mmar"

# The output path remains the same
OUTPUT_JSONL_PATH = Path("./data/mmar/mmar_test_standardized.jsonl")

# --- Script Execution ---
def main():
    print("--- MMAR Standardization Script (from local cache) ---")
    
    source_path = Path(MMAR_CACHE_DIR)
    # Check for the existence of the cache directory itself
    if not source_path.exists():
        print(f"ERROR: Could not find the cache directory at '{source_path.resolve()}'.")
        print("Please ensure MMAR_CACHE_DIR points to the correct dataset cache location.")
        return

    # 1. Load the dataset using load_dataset and select the 'test' split
    print(f"Loading dataset from: {source_path.resolve()}")
    # Use load_dataset on the cache directory, then select the 'test' split
    try:
        dataset = load_dataset(str(source_path))['test']
    except Exception as e:
        print(f"ERROR: Failed to load dataset from local cache '{source_path.resolve()}'.")
        print("This could be due to a version incompatibility between your 'datasets' library and the cached '.arrow' files.")
        print("Please ensure your 'datasets' and 'pyarrow' libraries are compatible with the cached data format.")
        print(f"Error details: {e}")
        return

    print("Dataset loaded successfully.")

    # ... (rest of the script is unchanged and correct) ...
    print("Processing and standardizing data...")
    standardized_data = []
    for item in tqdm(dataset, desc="Standardizing MMAR samples"):
        try:
            choices = item.get('choices')
            answer_str = item.get('answer')
            if not choices or not isinstance(choices, list) or answer_str is None:
                continue
            answer_key = choices.index(answer_str)
            standardized_sample = {
                "id": item.get('id'),
                "audio_path": item.get('audio_path'),
                "question": item.get('question'),
                "choices": choices,
                "answer_key": answer_key
            }
            standardized_data.append(standardized_sample)
        except ValueError:
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing item {item.get('id')}: {e}")
            continue
    print(f"Successfully standardized {len(standardized_data)} samples.")
    print(f"Writing standardized data to: {OUTPUT_JSONL_PATH.resolve()}")
    with open(OUTPUT_JSONL_PATH, 'w') as f:
        for entry in standardized_data:
            f.write(json.dumps(entry) + '\n')
    print("\n--- MMAR Standardization Complete ---")

if __name__ == "__main__":
    main()