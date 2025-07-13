import json
from pathlib import Path
from datasets import load_dataset
import os

def diagnose_dataset_ids(dataset_name, hf_id, cache_dir, standardized_json_path):
    """
    Diagnose ID mismatches between extracted audio and standardized JSON.
    """
    print(f"\n=== DIAGNOSING {dataset_name.upper()} ===")
    
    # 1. Try to load the dataset
    try:
        print(f"Loading dataset from cache: {cache_dir}")
        dataset = load_dataset(hf_id, split="test", cache_dir=cache_dir)
        print(f"✓ Dataset loaded successfully. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # 2. Check actual dataset IDs
    actual_ids = []
    for i, sample in enumerate(dataset):
        sample_id = sample.get('id', f"sample_{i}")
        actual_ids.append(sample_id)
        if i < 5:  # Show first 5 as examples
            print(f"  Sample {i}: ID = '{sample_id}'")
    
    print(f"Total actual dataset samples: {len(actual_ids)}")
    
    # 3. Check standardized JSON IDs
    json_path = Path(standardized_json_path)
    if json_path.exists():
        print(f"\nReading standardized JSON: {json_path}")
        json_ids = []
        with open(json_path, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    sample = json.loads(line.strip())
                    json_id = sample.get('id')
                    json_ids.append(json_id)
                    if i < 5:  # Show first 5 as examples
                        print(f"  JSON sample {i}: ID = '{json_id}'")
        
        print(f"Total JSON samples: {len(json_ids)}")
        
        # 4. Compare IDs
        actual_id_set = set(actual_ids)
        json_id_set = set(json_ids)
        
        matching_ids = actual_id_set & json_id_set
        only_in_dataset = actual_id_set - json_id_set
        only_in_json = json_id_set - actual_id_set
        
        print(f"\n--- ID COMPARISON ---")
        print(f"Matching IDs: {len(matching_ids)}")
        print(f"Only in dataset: {len(only_in_dataset)}")
        print(f"Only in JSON: {len(only_in_json)}")
        
        if only_in_dataset:
            print(f"Examples only in dataset: {list(only_in_dataset)[:5]}")
        if only_in_json:
            print(f"Examples only in JSON: {list(only_in_json)[:5]}")
            
    else:
        print(f"✗ Standardized JSON not found at: {json_path}")

def diagnose_cache_structure(cache_dir):
    """
    Examine the cache directory structure to understand what's there.
    """
    print(f"\n=== CACHE STRUCTURE: {cache_dir} ===")
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print("✗ Cache directory doesn't exist")
        return
    
    # Walk through cache structure
    for root, dirs, files in os.walk(cache_path):
        level = root.replace(str(cache_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # Show some files
        subindent = ' ' * 2 * (level + 1)
        for f in files[:3]:  # Show first 3 files
            print(f"{subindent}{f}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files)-3} more files")

def main():
    """
    Run diagnostics on all datasets.
    """
    datasets_to_check = [
        {
            "name": "mmar",
            "hf_id": "BoJack/MMAR",
            "cache_dir": "./data/mmar/hf_cache",
            "json_path": "./data/mmar/mmar_test_standardized.jsonl"
        },
        {
            "name": "sakura_animal", 
            "hf_id": "SLLM-multi-hop/AnimalQA",
            "cache_dir": "./data/sakura/animal/hf_cache",
            "json_path": "./data/sakura/animal/sakura_animal_test_standardized.jsonl"
        }
    ]
    
    print("=== AUDIO DATASET DIAGNOSTICS ===")
    print("This script will help identify ID mismatches and loading issues.")
    
    for dataset_config in datasets_to_check:
        # First check cache structure
        diagnose_cache_structure(dataset_config["cache_dir"])
        
        # Then diagnose IDs
        diagnose_dataset_ids(
            dataset_config["name"],
            dataset_config["hf_id"], 
            dataset_config["cache_dir"],
            dataset_config["json_path"]
        )
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()