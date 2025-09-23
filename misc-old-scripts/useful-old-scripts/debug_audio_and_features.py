# debug_audio_and_features.py
import soundfile as sf
import numpy as np
import librosa
import torch
from transformers import WhisperFeatureExtractor
import sys

wav = sys.argv[1] if len(sys.argv) > 1 else "data/mmar/audio/mmar_audio_1.wav"
print("Checking:", wav)

# read with soundfile
audio, sr = sf.read(wav, dtype="float32")
print("Original sr:", sr, "shape:", audio.shape, "dtype:", audio.dtype)
if audio.ndim == 2:
    print("Stereo detected, channels:", audio.shape[1])
    audio = audio[:, 0]
print("After mono shape:", audio.shape)

# resample to 16000
target_sr = 16000
if sr != target_sr:
    audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    sr = target_sr
    print("Resampled to", sr)

print("Post-resample length (samples):", len(audio), "duration(s):", len(audio)/sr)
print("Audio stats: min", float(np.min(audio)), "max", float(np.max(audio)), "mean", float(np.mean(audio)))
# detect near-silence
rms = float(np.sqrt((audio**2).mean()))
print("RMS:", rms)
if rms < 1e-5:
    print("WARNING: audio is effectively silent (very low RMS).")

# feature extractor check
wav_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")  # or use the exact path in your cfg
features = wav_processor([audio], sampling_rate=sr, return_tensors="pt")["input_features"]
print("Features tensor shape:", features.shape, "dtype:", features.dtype)
print("Features stats: min", float(features.min()), "max", float(features.max()), "mean", float(features.mean()))
