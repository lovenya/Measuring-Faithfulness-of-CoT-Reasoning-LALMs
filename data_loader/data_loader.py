# data_loader/data_loader.py

import json
import os
import random # <--- CHANGE 1: IMPORTED FOR JUMBLING UTILITY

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


# def get_dataset_info(data_samples: list) -> dict:
#     """
#     Get basic information about the loaded dataset.
    
#     Args:
#         data_samples (list): List of data samples.
        
#     Returns:
#         dict: Dictionary containing dataset statistics.
#     """
#     if not data_samples:
#         return {"total_samples": 0}
    
#     info = {
#         "total_samples": len(data_samples),
#         "sample_ids": [sample['id'] for sample in data_samples],
#     }
    
#     # Add track information if available
#     if 'track' in data_samples[0]:
#         tracks = set(sample.get('track', 'unknown') for sample in data_samples)
#         info['tracks'] = list(tracks)
    
#     # Add source information if available
#     if 'source' in data_samples[0]:
#         sources = set(sample.get('source', 'unknown') for sample in data_samples)
#         info['sources'] = list(sources)
    
#     return info


# def format_choices_for_prompt(choices: list) -> str:
#     """
#     --- CHANGE 2 (REWRITTEN FUNCTION) ---
#     Formats a list of choices into a standardized, enumerated string
#     on the fly. This ensures a consistent format for the model.
#     Example: ["cat", "dog"] -> "(A) cat\n(B) dog"
    
#     Args:
#         choices (list): List of choice strings.
        
#     Returns:
#         str: Formatted and enumerated choices string.
#     """
#     if not choices:
#         return ""
    
#     formatted_choices = []
#     for i, choice in enumerate(choices):
#         # chr(ord('A') + i) generates A, B, C, ...
#         letter = chr(ord('A') + i)
#         formatted_choices.append(f"({letter}) {choice}")
        
#     return "\n".join(formatted_choices)


# def prepare_jumbled_choices(choices: list, answer_key: int) -> tuple[list, int]:
#     """
#     --- CHANGE 3 (NEW FUNCTION) ---
#     Shuffles the choices and determines the new answer key.
#     This is a utility for experiments testing choice order bias.

#     Args:
#         choices (list): The original list of choices.
#         answer_key (int): The index of the correct answer in the original list.

#     Returns:
#         tuple[list, int]: A tuple containing:
#             - The new, shuffled list of choices.
#             - The new integer index of the correct answer.
#     """
#     if not choices or answer_key >= len(choices):
#         # Return original values if input is invalid to avoid crashing
#         return choices, answer_key

#     # Store the correct answer text
#     correct_answer_text = choices[answer_key]

#     # Create a new list to shuffle, preserving the original
#     shuffled_choices = list(choices)
#     random.shuffle(shuffled_choices)

#     # Find the new index of the correct answer
#     new_answer_key = shuffled_choices.index(correct_answer_text)

#     return shuffled_choices, new_answer_key