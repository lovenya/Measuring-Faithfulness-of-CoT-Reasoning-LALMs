# test_inference_job.py

import torch
import librosa
import json
from pathlib import Path
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import sys # To print errors

# --- Configuration: Define Paths ---
# Path to the local Qwen2-Audio model directory on the HPC
LOCAL_MODEL_PATH = "./Qwen2-Audio-7B-Instruct" 

# Paths to the standardized JSONL files created during data preparation
# IMPORTANT: These paths are relative to where your job script will be run (likely your project root)
MMAR_JSONL_PATH = "./data/mmar/mmar_test_standardized.jsonl"
SAKURA_EMOTION_JSONL_PATH = "./data/sakura/emotion/sakura_emotion_test_standardized.jsonl"

# --- Function to Load Model and Processor ---
def load_qwen_audio_model_and_processor(model_path: str):
    """Loads the Qwen-Audio model and its associated processor."""
    print(f"Loading model and processor from {model_path}...")
    try:
        # Use device_map="auto" to automatically load onto the available GPU
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto", # Load in bfloat16/float16 if supported, saves VRAM
            device_map="auto"
        )
        # use_fast=False is recommended for Qwen2-Audio processor
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        print("Model and processor loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"ERROR: Failed to load model or processor from {model_path}.", file=sys.stderr)
        raise # Re-raise the exception to stop the script

# --- Function to Load a Sample from JSONL ---
def load_sample_from_jsonl(jsonl_path: str, sample_index: int = 0):
    """Loads a specific sample (by index) from a standardized JSONL file."""
    print(f"Loading sample {sample_index} from {jsonl_path}...")
    try:
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if i == sample_index:
                    sample = json.loads(line.strip())
                    print(f"Sample {sample_index} loaded successfully.")
                    # print(f"Sample content: {sample}") # Uncomment for debugging
                    return sample
            print(f"ERROR: Sample index {sample_index} out of bounds in {jsonl_path}.", file=sys.stderr)
            return None # Return None if index is out of bounds
    except FileNotFoundError:
        print(f"ERROR: Standardized JSONL file not found at {jsonl_path}.", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Failed to decode JSON from line {i+1} in {jsonl_path}.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading sample {sample_index} from {jsonl_path}.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        return None


# --- Function to Run Inference on a Single Sample ---
def run_inference_on_sample(model, processor, sample: dict):
    """Runs inference on one standardized data sample."""
    print(f"\n--- Running Inference for Sample ID: {sample.get('id', 'N/A')} ---")
    
    audio_path = sample.get('audio_path')
    question = sample.get('question')
    choices = sample.get('choices')
    
    if not all([audio_path, question, choices is not None]): # Choices can be an empty list, check explicitly
        print(f"ERROR: Sample missing audio_path, question, or choices: {sample}", file=sys.stderr)
        return None

    # 1. Load and Resample Audio
    print(f"Loading audio from {audio_path}...")
    try:
        # CRITICAL: Ensure sample rate matches model expectation (16kHz for Qwen-Audio)
        audio_array, sampling_rate = librosa.load(str(Path(audio_path)), sr=16000)
        if sampling_rate != 16000:
             print(f"Warning: Audio originally sampled at {sampling_rate} Hz, resampled to 16000 Hz.")
        print("Audio loaded and resampled successfully.")
        # print(f"Audio shape: {audio_array.shape}, Sample Rate: {sampling_rate}") # Uncomment for debugging
    except FileNotFoundError:
        print(f"ERROR: Audio file not found at {audio_path}. Ensure path is correct on HPC.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Failed to load or process audio from {audio_path}.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        return None

    # 2. Prepare Text Prompt (Question + Choices)
    # Construct the prompt exactly as expected by the model for a multiple-choice question.
    prompt_text = f"{question}\n"
    if choices:
        for i, choice in enumerate(choices):
            # Use ASCII letters A, B, C... for the options
            prompt_text += f"({chr(65 + i)}) {choice}\n"
    
    # Add instruction for structured output
    prompt_text += "\nPlease respond with only the letter of the correct choice in parentheses. For example: (A)."

    print("Constructed Prompt:")
    print(prompt_text)


    # 3. Prepare Model Inputs (Multimodal)
    print("Preparing multimodal inputs for the model...")
    try:
        # Qwen-Audio expects a specific chat template and multimodal processor call
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": str(Path(audio_path))}, # Using path as required by template
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # Apply the chat template to the text part
        text_input = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process both text and audio inputs
        inputs = processor(text=[text_input], audios=[audio_array], return_tensors="pt").to(model.device)
        print("Inputs prepared.")
        # print(f"Input keys: {inputs.keys()}") # Uncomment for debugging
        # print(f"Input input_ids shape: {inputs['input_ids'].shape}") # Uncomment for debugging

    except Exception as e:
        print(f"ERROR: Failed to prepare model inputs for sample {sample.get('id', 'N/A')}.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        return None


    # 4. Generate the Response
    print("Generating response from model...")
    try:
        # Set a reasonable max_new_tokens, higher than minimum expected output (like (A))
        # to allow for potential freeform explanations if the model doesn't strictly follow the prompt.
        generate_ids = model.generate(**inputs, max_new_tokens=64) 
        print("Response generated.")
    except Exception as e:
        print(f"ERROR: Failed during model generation for sample {sample.get('id', 'N/A')}.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        return None

    # 5. Decode and Print the Result
    print("Decoding response...")
    try:
        # Decode the generated tokens, skipping special tokens
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("Response decoded.")
        
        print("\n" + "="*50)
        print(f"Sample ID: {sample.get('id', 'N/A')}")
        print(f"Ground Truth Answer Key: {sample.get('answer_key', 'N/A')} (Index)")
        if choices and isinstance(sample.get('answer_key'), int):
             try:
                 gt_choice = choices[sample['answer_key']]
                 print(f"Ground Truth Choice: {gt_choice}")
             except IndexError:
                 print(f"Warning: Invalid answer_key {sample['answer_key']} for choices list of length {len(choices)}.", file=sys.stderr)
        print(f"\nModel Response: {response.strip()}") # Use strip to remove leading/trailing whitespace
        print("="*50)
        
        return response.strip() # Return the cleaned response

    except Exception as e:
        print(f"ERROR: Failed to decode model response for sample {sample.get('id', 'N/A')}.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        return None


# --- Main Execution Flow ---
if __name__ == "__main__":
    print("--- Starting Test Inference Job Script ---")

    # 1. Check CUDA availability
    if not torch.cuda.is_available():
        print("FATAL ERROR: CUDA is not available. This script requires a GPU.", file=sys.stderr)
        sys.exit(1) # Exit with an error code

    device = "cuda"
    print(f"CUDA device found: {torch.cuda.get_device_name(0)}")
    
    # 2. Load Model and Processor
    try:
        model, processor = load_qwen_audio_model_and_processor(LOCAL_MODEL_PATH)
    except Exception: # load function prints error, just catch to exit
        sys.exit(1)

    # 3. Run Inference on MMAR Sample
    mmar_sample = load_sample_from_jsonl(MMAR_JSONL_PATH, sample_index=0)
    if mmar_sample:
        run_inference_on_sample(model, processor, mmar_sample)
    else:
        print("Skipping MMAR inference due to sample loading failure.", file=sys.stderr)

    # 4. Run Inference on SAKURA Emotion Sample
    sakura_sample = load_sample_from_jsonl(SAKURA_EMOTION_JSONL_PATH, sample_index=0)
    if sakura_sample:
        run_inference_on_sample(model, processor, sakura_sample)
    else:
         print("Skipping SAKURA Emotion inference due to sample loading failure.", file=sys.stderr)

    print("\n--- Test Inference Job Script Finished ---")