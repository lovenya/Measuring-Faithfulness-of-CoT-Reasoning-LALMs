# ssml_test.py

import os
import torch
from TTS.api import TTS
from pathlib import Path
import soundfile as sf

# --- NEW IMPORTS ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.shared_configs import BaseTTSConfig
# --- THE CRITICAL FIX: Use the correct import path ---
from TTS.tts.layers.xtts.tokenizer import CharacterConfig 
# --- END OF FIX ---
from torch.serialization import safe_globals

# --- Configuration (unchanged) ---
TTS_MODEL_PATH = "tts_models/XTTS-v2/"
REFERENCE_SPEAKER_PATH = "tts_models/reference_speaker.wav"
OUTPUT_DIR = "tts_test_outputs"

def get_audio_duration(filepath: str) -> float:
    # ... (this function is unchanged)
    try:
        with sf.SoundFile(filepath) as f:
            return len(f) / f.samplerate
    except Exception as e:
        print(f"Could not measure duration of {filepath}: {e}")
        return 0.0

def main():
    """
    Runs a series of tests to validate SSML <break> tag functionality for generating silence.
    """
    # --- 1. Setup ---
    print("Initializing Coqui TTS model from local files...")
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # The trusted classes list is now correct.
        trusted_classes = [
            XttsConfig, 
            XttsAudioConfig,
            BaseDatasetConfig,
            BaseTTSConfig,
            CharacterConfig
        ]
        
        with safe_globals(trusted_classes):
            tts = TTS(model_path=TTS_MODEL_PATH, config_path=os.path.join(TTS_MODEL_PATH, "config.json")).to(device)
        
        print(f"TTS model loaded successfully onto device: {device}")
    except Exception as e:
        print(f"FATAL: Failed to load TTS model. Error: {e}")
        print("Please ensure your tts-env is activated and the model path is correct.")
        return

    # ... (The rest of the script is unchanged) ...
    print("\n--- Running Test 1: Calibrated Silence (Two-Pass Method) ---")
    text_to_replace = "The cat sat"
    remaining_text = "on the mat."
    temp_audio_path = os.path.join(OUTPUT_DIR, "temp_word_segment.wav")
    print(f"  - Generating audio for the phrase: '{text_to_replace}'")
    tts.tts_to_file(text=text_to_replace, speaker_wav=REFERENCE_SPEAKER_PATH, language="en", file_path=temp_audio_path)
    target_silence_duration_ms = int(get_audio_duration(temp_audio_path) * 1000)
    print(f"  - Measured duration of phrase: {target_silence_duration_ms} ms")
    ssml_string = f'<speak><break time="{target_silence_duration_ms}ms"/> {remaining_text}</speak>'
    print(f"  - Constructed SSML string: {ssml_string}")
    final_output_path = os.path.join(OUTPUT_DIR, "calibrated_silence_test.wav")
    print(f"  - Generating final audio with silence...")
    tts.tts_to_file(text=ssml_string, speaker_wav=REFERENCE_SPEAKER_PATH, language="en", file_path=final_output_path, ssml_on=True)
    print(f"  - SUCCESS! Final audio saved to: {final_output_path}")
    print("\n--- Running Test 2: Simple Hard-coded Silence ---")
    simple_ssml = '<speak>This is the first part. <break time="2000ms"/> This is the second part.</speak>'
    simple_output_path = os.path.join(OUTPUT_DIR, "simple_silence_test.wav")
    print(f"  - Generating audio for: '{simple_ssml}'")
    tts.tts_to_file(text=simple_ssml, speaker_wav=REFERENCE_SPEAKER_PATH, language="en", file_path=simple_output_path, ssml_on=True)
    print(f"  - SUCCESS! Simple test audio saved to: {simple_output_path}")
    print("\n--- SSML Test Complete ---")
    print(f"Please download and listen to the files in the '{OUTPUT_DIR}' directory to verify the results.")

if __name__ == "__main__":
    main()