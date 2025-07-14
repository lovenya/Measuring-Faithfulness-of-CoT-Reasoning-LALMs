# test_inference_job.py

import torch
import librosa
import json
from pathlib import Path
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import sys
import os

# --- Configuration: Define Paths ---
LOCAL_MODEL_PATH = "./Qwen2-Audio-7B-Instruct" 
MMAR_JSONL_PATH = "./data/mmar/mmar_test_standardized.jsonl"
SAKURA_EMOTION_JSONL_PATH = "./data/sakura/emotion/sakura_emotion_test_standardized.jsonl"

# --- Function to Load Model and Processor ---
def load_qwen_audio_model_and_processor(model_path: str):
    """Loads the Qwen-Audio model and its associated processor."""
    print(f"Loading model and processor from {model_path}...")
    try:
        # Check if CUDA is available and set device
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            print(f"Using device: {device}")
        else:
            print("CUDA not available, using CPU")
            device = "cpu"
        
        # Load processor first
        processor = AutoProcessor.from_pretrained(
            model_path, 
            use_fast=False,
            trust_remote_code=True
        )
        print("✓ Processor loaded successfully")
        
        # Load model
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        print("✓ Model loaded successfully")
        
        print(f"Model device: {next(model.parameters()).device}")
        return model, processor
    except Exception as e:
        print(f"ERROR: Failed to load model or processor from {model_path}.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        raise

# --- Function to Load a Sample from JSONL ---
def load_sample_from_jsonl(jsonl_path: str, sample_index: int = 0):
    """Loads a specific sample (by index) from a standardized JSONL file."""
    print(f"Loading sample {sample_index} from {jsonl_path}...")
    
    # Check if file exists
    if not os.path.exists(jsonl_path):
        print(f"ERROR: Standardized JSONL file not found at {jsonl_path}.", file=sys.stderr)
        return None
    
    try:
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if i == sample_index:
                    sample = json.loads(line.strip())
                    print(f"Sample {sample_index} loaded successfully.")
                    # Validate sample structure
                    if not all(key in sample for key in ['audio_path', 'question', 'choices']):
                        print(f"ERROR: Sample missing required keys: {sample}", file=sys.stderr)
                        return None
                    return sample
            print(f"ERROR: Sample index {sample_index} out of bounds in {jsonl_path}.", file=sys.stderr)
            return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to decode JSON from line {i+1} in {jsonl_path}.", file=sys.stderr)
        print(f"JSON error: {e}", file=sys.stderr)
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
    
    if not all([audio_path, question, choices is not None]):
        print(f"ERROR: Sample missing audio_path, question, or choices: {sample}", file=sys.stderr)
        return None

    # 1. Load and Resample Audio
    print(f"Loading audio from {audio_path}...")
    
    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at {audio_path}. Ensure path is correct on HPC.", file=sys.stderr)
        return None
    
    try:
        # Load audio at 16kHz for Qwen-Audio
        audio_array, sampling_rate = librosa.load(str(Path(audio_path)), sr=16000)
        print(f"Audio loaded successfully. Shape: {audio_array.shape}, Sample Rate: {sampling_rate}")
    except Exception as e:
        print(f"ERROR: Failed to load or process audio from {audio_path}.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        return None

    # 2. Prepare Text Prompt (Question + Choices)
    prompt_text = f"{question}\n"
    if choices:
        for i, choice in enumerate(choices):
            prompt_text += f"({chr(65 + i)}) {choice}\n"
    
    prompt_text += "\nPlease respond with only the letter of the correct choice in parentheses. For example: (A)."

    print("Constructed Prompt:")
    print(prompt_text)

    # 3. Prepare Model Inputs (Multimodal)
    print("Preparing multimodal inputs for the model...")
    try:
        # Qwen-Audio conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": str(Path(audio_path))},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # Apply chat template
        text_input = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(text=[text_input], audios=[audio_array], return_tensors="pt")
        
        # Move to model device
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print("✓ Inputs prepared successfully")
        print(f"Input shape: {inputs['input_ids'].shape}")

    except Exception as e:
        print(f"ERROR: Failed to prepare model inputs for sample {sample.get('id', 'N/A')}.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        return None

    # 4. Generate the Response
    print("Generating response from model...")
    try:
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs, 
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        print("✓ Response generated successfully")
    except Exception as e:
        print(f"ERROR: Failed during model generation for sample {sample.get('id', 'N/A')}.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        return None

    # 5. Decode and Print the Result
    print("Decoding response...")
    try:
        # Extract only the newly generated tokens
        input_token_length = inputs['input_ids'].shape[1]
        new_tokens = generate_ids[:, input_token_length:]
        
        # Decode response
        response = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print("\n" + "="*60)
        print(f"Sample ID: {sample.get('id', 'N/A')}")
        print(f"Ground Truth Answer Key: {sample.get('answer_key', 'N/A')} (Index)")
        
        if choices and isinstance(sample.get('answer_key'), int):
            try:
                gt_choice = choices[sample['answer_key']]
                gt_letter = chr(65 + sample['answer_key'])
                print(f"Ground Truth Choice: ({gt_letter}) {gt_choice}")
            except IndexError:
                print(f"Warning: Invalid answer_key {sample['answer_key']} for choices list of length {len(choices)}.", file=sys.stderr)
        
        print(f"\nModel Response: {response.strip()}")
        print("="*60)
        
        return response.strip()

    except Exception as e:
        print(f"ERROR: Failed to decode model response for sample {sample.get('id', 'N/A')}.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        return None

# --- Main Execution Flow ---
if __name__ == "__main__":
    print("--- Starting Test Inference Job Script ---")

    # 1. Check system info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available. This will be very slow on CPU.")

    # 2. Load Model and Processor
    try:
        model, processor = load_qwen_audio_model_and_processor(LOCAL_MODEL_PATH)
    except Exception:
        print("FATAL: Could not load model. Exiting.")
        sys.exit(1)

    # 3. Run Inference on MMAR Sample
    print("\n" + "="*70)
    print("TESTING MMAR DATASET")
    print("="*70)
    
    mmar_sample = load_sample_from_jsonl(MMAR_JSONL_PATH, sample_index=0)
    if mmar_sample:
        mmar_response = run_inference_on_sample(model, processor, mmar_sample)
        if mmar_response:
            print(f"\n✓ MMAR inference completed successfully.")
        else:
            print("✗ MMAR inference failed.", file=sys.stderr)
    else:
        print("✗ Skipping MMAR inference due to sample loading failure.", file=sys.stderr)

    # 4. Run Inference on SAKURA Emotion Sample
    print("\n" + "="*70)
    print("TESTING SAKURA EMOTION DATASET")
    print("="*70)
    
    sakura_sample = load_sample_from_jsonl(SAKURA_EMOTION_JSONL_PATH, sample_index=0)
    if sakura_sample:
        sakura_response = run_inference_on_sample(model, processor, sakura_sample)
        if sakura_response:
            print(f"\n✓ SAKURA Emotion inference completed successfully.")
        else:
            print("✗ SAKURA Emotion inference failed.", file=sys.stderr)
    else:
        print("✗ Skipping SAKURA Emotion inference due to sample loading failure.", file=sys.stderr)

    print("\n--- Test Inference Job Script Finished ---")