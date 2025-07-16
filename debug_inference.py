#!/usr/bin/env python3
"""
Full Diagnostics for Qwen2-Audio-7B-Instruct (offline):

1) Test both `audio=` and `audios=` keywords
2) Check for zero-length or silent audio arrays
3) Verify sample-rate matches processor
4) Run a “minimal” local-file example
5) Inspect raw inputs tensor shapes

— Fixes:
  • Always pass sampling_rate to the processor call
  • Use `audio=` instead of deprecated `audios=`
  • Use max_new_tokens instead of max_length for .generate()
"""

import os
import torch
import librosa
import numpy as np
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# ——— Helpers ——————————————————————————————————————————————————

def check_sample_rate(processor, target_sr):
    model_sr = processor.feature_extractor.sampling_rate
    print(f"[3] Processor expects sampling_rate = {model_sr}, "
          f"librosa.load using target_sr = {target_sr}")
    if model_sr != target_sr:
        print("⚠️  WARNING: sample rates differ!")

def check_audio_array(wav, path):
    length = wav.shape[0]
    rms = float(np.sqrt(np.mean(wav**2)))
    print(f"[2] Loaded {length} samples from {path}, RMS amplitude = {rms:.6f}")
    if length == 0:
        print("⚠️  ERROR: zero-length waveform!")
    if rms < 1e-4:
        print("⚠️  WARNING: audio is nearly silent!")

def inspect_inputs(inputs, label):
    print(f"\n[5] INPUT TENSORS ({label}) — shape listing:")
    for k, v in inputs.items():
        print(f"    {k:20s} {tuple(v.shape)}")
    print()

def minimal_local_example(model, processor, wav, sr):
    print("\n[4] Running minimal local‑file example:")
    # build a trivial conversation to test audio ingestion
    conv = [
       {'role': 'system', 'content': 'You are a helpful assistant.'},
       {"role": "user", "content": [
           {"type": "audio", "array": wav},
           {"type": "text",  "text": "What kind of sound is this?"}
       ]}
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    # featurize (pass sampling_rate explicitly!)
    inputs = processor(
        text=text,
        audio=[wav],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inspect_inputs(inputs, "local minimal example")
    # generate (use max_new_tokens)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=64)
    out = processor.batch_decode(
        ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]
    print("    Local minimal output:", out, "\n")

def test_kwarg_variants(processor, wav, sr):
    print("[1] Testing keyword-arg variants:")
    for kw in ("audio", "audios"):
        try:
            kwargs = {kw: [wav]}
            inputs = processor(
                text=["<|AUDIO|> test"],
                **kwargs,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
            print(f"    • processor(..., {kw}=[wav], ...) accepted → keys: {list(inputs.keys())}")
        except Exception as e:
            print(f"    • processor with `{kw}=` raised: {e}")

# ——— Main routine ——————————————————————————————————————————————

def main():
    LOCAL_MODEL_PATH = "./Qwen2-Audio-7B-Instruct"
    print(f"Loading processor & model from '{LOCAL_MODEL_PATH}'...")
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    model     = Qwen2AudioForConditionalGeneration.from_pretrained(
        LOCAL_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True
    )
    print(f"Loaded on device: {model.device}\n")

    # 3) sample-rate check
    TARGET_SR = 16000
    check_sample_rate(processor, TARGET_SR)

    # prepare local file
    local_path = "data/sakura/emotion/audio/sakura_emotion_audio_7.wav"
    if not os.path.exists(local_path):
        print(f"Local file not found: {local_path}")
        return

    # 2) load and check array
    wav, sr = librosa.load(local_path, sr=TARGET_SR)
    check_audio_array(wav, local_path)

    # 1) test audio= vs audios=
    test_kwarg_variants(processor, wav, sr)

    # 4) minimal local example
    minimal_local_example(model, processor, wav, sr)

    # 5) inspect actual inputs and run one generate
    print("\n[5] Building inputs with the CORRECT kwarg (audio=)...")
    conv = [
        {"role":"system", "content":"You are a helpful assistant."},
        {"role":"user",   "content":[
            {"type":"audio", "array":wav},
            {"type":"text",  "text":"What animal is this?"}
        ]}
    ]
    
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text,
        audio=[wav],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inspect_inputs(inputs, "Local file final")

    print("[5] Generating on local file (final)...")
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=50)
    raw = processor.batch_decode(
        ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )[0]
    print("    Raw model output:", repr(raw))

if __name__ == "__main__":
    main()
