import torch
import librosa
import json
import re
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# --- Configuration ---
# Paths to the LOCAL, PRE-DOWNLOADED model and data
LOCAL_MODEL_PATH = "./Qwen2-Audio-7B-Instruct"
MMAR_ARROW_PATH = "./data/mmar"  # Path to the directory containing the .arrow file
SAKURA_ARROW_PATH = "./data/sakura/animal" # Path for the Animal track

# --- This logic would belong in data_handling/data_loader.py ---
# It's included here for this self-contained verification script.

def parse_sakura_instruction(instruction: str, answer_str: str):
    """Parses a SAKURA instruction to extract question, choices, and answer key."""
    try:
        choice_pattern = re.compile(r'\(([a-z])\)\s*(.*?)(?=\s*\([a-z]\)\s*|$)', re.DOTALL)
        matches = choice_pattern.findall(instruction)
        if not matches: return None
        
        first_match_start_index = instruction.find(f"({matches[0][0]})")
        question = instruction[:first_match_start_index].strip()
        
        letters = [match[0] for match in matches]
        choices = [match[1].strip() for match in matches]
        
        correct_letter_match = re.search(r'\(([a-z])\)', answer_str)
        if not correct_letter_match: return None
        correct_letter = correct_letter_match.group(1)
        
        answer_key = letters.index(correct_letter)
        return {"question": question, "choices": choices, "answer_key": answer_key}
    except Exception:
        return None

def get_mmar_sample(arrow_path: str, index: int = 0):
    """Loads a single standardized sample from the MMAR .arrow file."""
    print(f"\n--- Loading MMAR sample from: {arrow_path} ---")
    dataset = load_from_disk(arrow_path)
    sample = dataset[index]
    
    # Perform the answer-to-key transformation on the fly
    answer_key = sample['choices'].index(sample['answer'])
    
    return {
        "audio_path": sample['audio_path'],
        "question": sample['question'],
        "choices": sample['choices'],
        "answer_key": answer_key
    }

def get_sakura_sample(arrow_path: str, index: int = 0):
    """Loads a single standardized sample from the SAKURA .arrow file."""
    print(f"\n--- Loading SAKURA sample from: {arrow_path} ---")
    dataset = load_from_disk(arrow_path)
    sample = dataset[index]
    
    # Use the parser for the multi-hop question as a test case
    parsed_data = parse_sakura_instruction(sample['multi_instruction'], sample['multi_answer'])
    
    return {
        "audio_path": sample['audio']['path'],
        **parsed_data
    }

# --- This logic would belong in core/llm_audio_utils.py ---
# It's included here for this self-contained verification script.

def run_inference(model, processor, device, sample_data):
    """Runs a single inference and prints the result."""
    audio_path = sample_data['audio_path']
    question = sample_data['question']
    choices = sample_data['choices']
    
    # Construct the prompt
    options_str = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
    prompt_text = f"Based on the provided audio, answer the following question.\n\nQuestion: {question}\n{options_str}\n\nPlease respond with only the letter of the correct choice in parentheses."

    print(f"Audio File: {Path(audio_path).name}")
    print(f"Prompt being sent to model:\n---\n{prompt_text}\n---")

    # Prepare model inputs
    audio_array, _ = librosa.load(audio_path, sr=16000)
    conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": audio_path}, {"type": "text", "text": prompt_text}]}]
    templated_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[templated_text], audios=[audio_array], return_tensors="pt").to(device)

    # Generate response
    generate_ids = model.generate(**inputs, max_new_tokens=10)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Simple parsing of the final response
    final_answer = response.split("user")[-1].strip()

    print(f"\nModel Full Response: {response}")
    print(f"Parsed Answer: {final_answer}")
    print("-" * 50)


# --- Main Verification Execution ---
def main():
    print("--- Verification Script for Local Arrow Datasets ---")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This script must be run on a GPU node.")
    device = "cuda"
    
    print("--> Loading model and processor...")
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH, torch_dtype="auto", device_map="auto")
    print("--> Model loaded.")

    # Test MMAR
    try:
        mmar_sample = get_mmar_sample(MMAR_ARROW_PATH)
        run_inference(model, processor, device, mmar_sample)
    except Exception as e:
        print(f"!!! FAILED to process MMAR sample: {e}")

    # Test SAKURA
    try:
        sakura_sample = get_sakura_sample(SAKURA_ARROW_PATH)
        run_inference(model, processor, device, sakura_sample)
    except Exception as e:
        print(f"!!! FAILED to process SAKURA sample: {e}")

if __name__ == "__main__":
    main()