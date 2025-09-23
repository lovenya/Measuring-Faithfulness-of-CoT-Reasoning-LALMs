# data_processing/create_silent_audio.py

# standalone script to make a silent audio for AF3 and SALMONN's infernece to work

import numpy as np
import soundfile as sf
import os

def create_silent_wav(output_path: str, duration_s: int = 1, sample_rate: int = 16000):
    """
    Generates a silent, mono WAV file.

    This is a crucial asset for our research framework. It allows us to perform
    text-only inference with multi-modal models (like Audio Flamingo) that
    always expect an audio input. By providing a silent track, we ensure that
    the model's text-manipulation capabilities are tested in a scientifically
    pure manner, without any influence from a real audio stimulus.

    Args:
        output_path (str): The path where the silent WAV file will be saved.
        duration_s (int): The duration of the silent audio in seconds.
        sample_rate (int): The sample rate for the audio file. 16000 Hz is a
                           standard for many speech and audio models.
    """
    print(f"--- Generating Silent Audio File ---")
    
    # Ensure the directory for the output file exists.
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)

    # Create an array of zeros. The length is duration * sample_rate.
    # The data type is float32, a standard for high-quality audio processing.
    num_samples = int(duration_s * sample_rate)
    silent_audio = np.zeros(num_samples, dtype=np.float32)

    # Use the soundfile library to write the NumPy array to a WAV file.
    # We specify mono (channels=1) and the standard 'PCM_16' subtype.
    try:
        sf.write(output_path, silent_audio, sample_rate, subtype='PCM_16')
        print(f"Successfully created silent WAV file at: {output_path}")
        print(f"  - Duration: {duration_s} second(s)")
        print(f"  - Sample Rate: {sample_rate} Hz")
    except Exception as e:
        print(f"ERROR: Failed to create silent audio file.")
        print(f"  - Details: {e}")

if __name__ == "__main__":
    # We define the standard output path here. This aligns with the path
    # we have set in our main config.py file.
    DEFAULT_OUTPUT_PATH = "./assets/silent.wav"
    create_silent_wav(DEFAULT_OUTPUT_PATH)