# run_inference.py

import torch
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# --- 1. Define Paths and Check for GPU ---
# Use the local model folder we created with git-lfs
local_model_path = "./Qwen2-Audio-7B-Instruct" 
# Use the local audio file we just downloaded
local_audio_path = "./sample_data/harvard.wav"

print("--- Qwen2-Audio Inference Test ---")
print(f"Model Path: {local_model_path}")
print(f"Audio Path: {local_audio_path}")

try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This script must be run on a GPU node.")
    device = "cuda"
    print(f"CUDA device found: {torch.cuda.get_device_name(0)}")

    # --- 2. Load Model and Processor from Local Files ---
    print("--> Loading processor and model...")
    processor = AutoProcessor.from_pretrained(local_model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    print("--> Model and processor loaded successfully.")

    # --- 3. Prepare Inputs (Audio + Text Prompt) ---
    print("--> Preparing inputs for the model...")
    
    # Load the audio file
    audio_array, sampling_rate = librosa.load(local_audio_path, sr=16000)

    # This is the structured conversation format the model expects.
    # We provide the audio and then ask a question about it.
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": local_audio_path}, # NOTE: We use the path here
                {"type": "text", "text": "What does the person say?"},
            ],
        }
    ]

    # The processor converts the conversation into tokens and processes the audio.
    # We pass the loaded audio array directly.
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text], audios=[audio_array], return_tensors="pt").to(device)
    print("--> Inputs prepared.")

    # --- 4. Generate the Response ---
    print("--> Generating response from model...")
    generate_ids = model.generate(**inputs, max_new_tokens=512)
    
    # --- 5. Decode and Print the Result ---
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print("\n" + "="*50)
    print("INFERENCE COMPLETE")
    print(f"\nModel Prompt: What does the person say?")
    print(f"\nModel Response: {response}")
    print("="*50)

except Exception as e:
    print(f"\nINFERENCE FAILED")
    print(f"An error occurred: {e}")