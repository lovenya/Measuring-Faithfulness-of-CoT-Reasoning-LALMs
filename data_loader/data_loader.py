import json
import os


def load_dataset(jsonl_path: str) -> list:
    """
    Loads a dataset from a JSONL file.
    
    Args:
        jsonl_path (str): Path to the JSONL file.
        
    Returns:
        list: List of data samples with standardized format.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
    
    data_samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract required fields
                sample = {
                    'id': data['id'],
                    'audio_path': data['audio_path'],
                    'question': data['question'],
                    'choices': data['choices'],
                    'answer_key': data['answer_key']
                }
                
                # Add optional fields if they exist
                if 'answer' in data:
                    sample['answer'] = data['answer']
                if 'hop_type' in data:
                    sample['hop_type'] = data['hop_type']
                if 'track' in data:
                    sample['track'] = data['track']
                if 'modality' in data:
                    sample['modality'] = data['modality']
                if 'language' in data:
                    sample['language'] = data['language']
                if 'source' in data:
                    sample['source'] = data['source']
                
                data_samples.append(sample)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num} in {jsonl_path}: {e}")
                continue
            except KeyError as e:
                print(f"Warning: Missing required field {e} in line {line_num} of {jsonl_path}")
                continue
    
    print(f"Loaded {len(data_samples)} samples from {jsonl_path}")
    return data_samples


def get_dataset_info(data_samples: list) -> dict:
    """
    Get basic information about the loaded dataset.
    
    Args:
        data_samples (list): List of data samples.
        
    Returns:
        dict: Dictionary containing dataset statistics.
    """
    if not data_samples:
        return {"total_samples": 0}
    
    info = {
        "total_samples": len(data_samples),
        "sample_ids": [sample['id'] for sample in data_samples],
    }
    
    # Add track information if available
    if 'track' in data_samples[0]:
        tracks = set(sample.get('track', 'unknown') for sample in data_samples)
        info['tracks'] = list(tracks)
    
    # Add source information if available
    if 'source' in data_samples[0]:
        sources = set(sample.get('source', 'unknown') for sample in data_samples)
        info['sources'] = list(sources)
    
    return info


def format_choices_for_prompt(choices: list) -> str:
    """
    Format choices list into a string suitable for prompting.
    
    Args:
        choices (list): List of choice strings.
        
    Returns:
        str: Formatted choices string.
    """
    return "\n".join(choices)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_datasets = [
        "mmar_test_standardized.jsonl",
        "sakura_animal_test_standardized.jsonl"
        
    ]
    
    for dataset_path in sample_datasets:
        if os.path.exists(dataset_path):
            print(f"\n--- Testing {dataset_path} ---")
            try:
                data = load_dataset(dataset_path)
                info = get_dataset_info(data)
                print(f"Dataset info: {info}")
                
                if data:
                    print(f"First sample: {data[0]}")
                    print(f"Formatted choices: {format_choices_for_prompt(data[0]['choices'])}")
                    
            except Exception as e:
                print(f"Error loading {dataset_path}: {e}")
        else:
            print(f"\n{dataset_path} not found - skipping test")