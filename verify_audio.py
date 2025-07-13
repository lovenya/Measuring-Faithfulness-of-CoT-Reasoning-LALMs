# verify_audio.py (Corrected Version)
from datasets import load_dataset, Audio
import soundfile as sf
from pathlib import Path

# --- Configuration ---
# The Hugging Face ID of the dataset
DATASET_ID = "SLLM-multi-hop/AnimalQA"
# The path to the cache directory we transferred from local
CACHE_PATH = "./data/sakura/animal/hf_cache"
# Index of the sample we want to extract
SAMPLE_INDEX = 0
# Where to save the extracted audio file
OUTPUT_FILENAME = "extracted_sample_animal_0.wav"

print("--- Audio Verification Script (Corrected) ---")

try:
    # 1. Load the dataset using its ID.
    #    It will find the data in the specified cache_dir and load instantly.
    print(f"Loading dataset '{DATASET_ID}' from cache: {CACHE_PATH}")
    dataset = load_dataset(DATASET_ID, split="test", cache_dir=CACHE_PATH)
    print("Dataset loaded successfully from local cache.")

    # 2. Access the specific audio sample
    print(f"Extracting audio from sample at index {SAMPLE_INDEX}...")
    audio_data = dataset[SAMPLE_INDEX]['audio']
    
    if audio_data and 'array' in audio_data and 'sampling_rate' in audio_data:
        audio_array = audio_data['array']
        sampling_rate = audio_data['sampling_rate']
        
        # 3. Save the extracted audio data to a new .wav file
        print(f"Saving extracted audio to: {OUTPUT_FILENAME}")
        sf.write(OUTPUT_FILENAME, audio_array, sampling_rate)
        
        print("\nSUCCESS!")
        print(f"Audio for sample {SAMPLE_INDEX} has been extracted and saved to '{OUTPUT_FILENAME}'.")
        print("You can now copy this file back to your local machine to listen to it.")
    else:
        print(f"\nERROR: Sample {SAMPLE_INDEX} does not contain valid audio data.")
        print(f"Data found: {audio_data}")

except Exception as e:
    print(f"\nAn error occurred")
    print(f"Details: {e}")