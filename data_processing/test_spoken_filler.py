# data_processing/test_spoken_filler.py

import os
import torch
import torchaudio
import time
import numpy as np
from pydub import AudioSegment

# --- Standard Setup (from our previous successful scripts) ---
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils import io
except ImportError:
    print("FATAL ERROR: TTS library not found.")
    print("Please ensure you are running this script in the dedicated 'tts-env' Conda environment.")
    exit(1)

original_load_fsspec = io.load_fsspec
def patched_load_fsspec(path, map_location=None, **kwargs):
    kwargs['weights_only'] = False
    return original_load_fsspec(path, map_location, **kwargs)
io.load_fsspec = patched_load_fsspec

def setup_tts_model(model_dir: str) -> Xtts:
    """ Loads the local, offline Coqui XTTS-v2 model into memory. """
    print("--- Setting up Coqui TTS Model ---")
    config_path = os.path.join(model_dir, "config.json")
    
    # --- THE CRITICAL FIX ---
    # In newer versions of the TTS library, the configuration must be loaded in two steps.
    # First, create the config object.
    config = XttsConfig()
    # Second, load the JSON data into that object.
    config.load_json(config_path)
    # --- END OF FIX ---

    # Now, we pass the correctly populated config object to the model initializer.
    model = Xtts.init_from_config(config)
    
    model.load_checkpoint(model.config, checkpoint_dir=model_dir, use_deepspeed=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"--- TTS Model setup complete on device: {device} ---")
    return model

# --- The rest of the script (convert_filler_text_to_audio and __main__) is unchanged and correct. ---
def convert_filler_text_to_audio(tts_model: Xtts, text: str, speaker_wav: str) -> AudioSegment:
    """
    Converts a string containing words and '...' filler tokens into a single
    audio segment with speech and corresponding silence.
    """
    parts = text.split(' ')
    final_audio = AudioSegment.empty()
    current_text_chunk = ""
    for part in parts:
        if part == "...":
            if current_text_chunk:
                print(f"  - Generating speech for: '{current_text_chunk.strip()}'")
                outputs = tts_model.synthesize(
                    current_text_chunk.strip(), tts_model.config, speaker_wav=speaker_wav,
                    language="en", enable_text_splitting=True
                )
                numpy_audio = outputs['wav']
                raw_audio_data = (numpy_audio * 32767).astype(np.int16).tobytes()
                speech_segment = AudioSegment(data=raw_audio_data, sample_width=2, frame_rate=24000, channels=1)
                final_audio += speech_segment
                current_text_chunk = ""
            print("  - Generating 40ms of silence for '...'")
            silence_segment = AudioSegment.silent(duration=40)
            final_audio += silence_segment
        else:
            current_text_chunk += part + " "
    if current_text_chunk:
        print(f"  - Generating speech for final chunk: '{current_text_chunk.strip()}'")
        outputs = tts_model.synthesize(
            current_text_chunk.strip(), tts_model.config, speaker_wav=speaker_wav,
            language="en", enable_text_splitting=True
        )
        numpy_audio = outputs['wav']
        raw_audio_data = (numpy_audio * 32767).astype(np.int16).tobytes()
        speech_segment = AudioSegment(data=raw_audio_data, sample_width=2, frame_rate=24000, channels=1)
        final_audio += speech_segment
    return final_audio


if __name__ == "__main__":
    TTS_MODEL_DIR = './tts_models/XTTS-v2'
    SPEAKER_WAV = './tts_models/reference_speaker.wav'
    OUTPUT_DIR = './tts_filler_test_output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_sentences = [
        ("This is a simple test.", "test_1_no_filler.wav"),
        ("... ... ... This test starts with silence.", "test_2_start_filler.wav"),
        ("This test ends with silence. ... ... ...", "test_3_end_filler.wav"),
        ("This ... test has ... filler in the ... middle.", "test_4_middle_filler.wav")
    ]
    print("Loading TTS model for the test...")
    tts_model = setup_tts_model(TTS_MODEL_DIR)
    for i, (text, filename) in enumerate(test_sentences):
        print(f"\n--- Processing test sentence {i+1}/{len(test_sentences)} ---")
        print(f"Input text: '{text}'")
        start_time = time.time()
        generated_audio = convert_filler_text_to_audio(tts_model, text, SPEAKER_WAV)
        end_time = time.time()
        output_path = os.path.join(OUTPUT_DIR, filename)
        print(f"  - Saving final combined audio to: {output_path}")
        generated_audio.export(output_path, format="wav")
        print(f"  - Done. Generation took {end_time - start_time:.2f} seconds.")
    print(f"\n--- Test complete. Please check the '{OUTPUT_DIR}' directory and listen to the .wav files. ---")