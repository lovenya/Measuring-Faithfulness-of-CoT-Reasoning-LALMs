#!/usr/bin/env python3
"""
Transcribe WAV files with a locally downloaded HuggingFace-style Whisper model.
This version ensures input_features dtype matches model dtype and supplies an attention_mask.
"""
import os
import argparse
import glob
from pathlib import Path

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_audio(path, target_sr=16000):
    speech, sr = torchaudio.load(path)
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    speech = speech.squeeze(0).numpy()
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        speech = resampler(torch.from_numpy(speech)).numpy()
    return speech, target_sr

def forced_decoder_prompt_length(forced_decoder_ids):
    if not forced_decoder_ids:
        return 0
    try:
        if any(isinstance(x, (list, tuple)) for x in forced_decoder_ids):
            return sum(len(x) for x in forced_decoder_ids)
        return len(forced_decoder_ids)
    except Exception:
        return 0

def transcribe_file(model, processor, audio_path, device, requested_max_new_tokens=448, num_beams=5):
    speech_array, sampling_rate = load_audio(audio_path, target_sr=16000)

    # produce input_features (batch, seq_len, feat_dim)
    try:
        inputs = processor.feature_extractor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features
    except Exception:
        processed = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = processed["input_features"]

    # move to device
    input_features = input_features.to(device)

    # --- NEW: cast input_features to model dtype to avoid float32 <-> float16 mismatch ---
    # determine model dtype (use first param)
    model_param = next(model.parameters())
    model_dtype = model_param.dtype
    if input_features.dtype != model_dtype:
        # cast input features to model dtype (e.g., float16)
        input_features = input_features.to(model_dtype)

    # --- NEW: create attention_mask if not provided (ones) ---
    # attention_mask shape should be (batch, seq_len) for encoder
    attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.long, device=device)

    # optional forced decoder ids
    forced_decoder_ids = None
    try:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    except Exception:
        forced_decoder_ids = None

    # safe max_new_tokens computation (same as before)
    max_target_positions = getattr(model.config, "max_target_positions", 448)
    decoder_prompt_len = forced_decoder_prompt_length(forced_decoder_ids)
    available_for_generation = max_target_positions - decoder_prompt_len
    if available_for_generation <= 1:
        raise RuntimeError(
            f"Not enough room for generation: model max_target_positions={max_target_positions}, "
            f"decoder_prompt_len={decoder_prompt_len}."
        )
    safe_max_new_tokens = min(requested_max_new_tokens, max(1, available_for_generation - 1))

    gen_kwargs = dict(max_new_tokens=safe_max_new_tokens, num_beams=num_beams)
    if forced_decoder_ids is not None:
        gen_kwargs["forced_decoder_ids"] = forced_decoder_ids

    # pass attention_mask explicitly and use input_features= to avoid deprecation warning
    with torch.no_grad():
        generated_ids = model.generate(input_features=input_features,
                                       attention_mask=attention_mask,
                                       **gen_kwargs)

    try:
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception:
        transcription = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    meta = {
        "max_target_positions": max_target_positions,
        "decoder_prompt_len": decoder_prompt_len,
        "used_max_new_tokens": safe_max_new_tokens,
        "model_dtype": str(model_dtype),
        "input_features_dtype": str(input_features.dtype)
    }

    return transcription.strip(), meta

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    model_dir = Path(args.model_dir).resolve()
    assert model_dir.exists(), f"Model dir not found: {model_dir}"

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)

    print("Loading model (this can take some time)...")
    load_kwargs = {"local_files_only": True}
    if device.type == "cuda":
        # prefer dtype parameter for newer HF versions
        try:
            load_kwargs["dtype"] = torch.float16
        except Exception:
            load_kwargs["torch_dtype"] = torch.float16

    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir), **load_kwargs)
    model.to(device)
    model.eval()

    input_path = Path(args.input_path)
    if input_path.is_dir():
        wavs = sorted(glob.glob(str(input_path / "*.wav")))
        if not wavs:
            raise SystemExit(f"No .wav files found in {input_path}")
    else:
        wavs = [str(input_path)]

    os.makedirs(args.output_dir, exist_ok=True)

    for wav in wavs:
        print(f"\n-> Transcribing: {wav}")
        try:
            text, meta = transcribe_file(model, processor, wav, device,
                                         requested_max_new_tokens=args.max_new_tokens,
                                         num_beams=args.num_beams)
        except Exception as e:
            print("Error during transcription:", repr(e))
            raise
        print(f"(model.max_target_positions={meta['max_target_positions']}, "
              f"decoder_prompt_len={meta['decoder_prompt_len']}, "
              f"used_max_new_tokens={meta['used_max_new_tokens']}, "
              f"model_dtype={meta['model_dtype']}, input_features_dtype={meta['input_features_dtype']})")

        out_name = Path(args.output_dir) / (Path(wav).stem + ".txt")
        out_name.write_text(text, encoding="utf-8")
        print(f"Saved -> {out_name}")
        print("Preview:", text[:300].replace("\n", " "))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./transcripts")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_new_tokens", type=int, default=448)
    p.add_argument("--num_beams", type=int, default=5)
    args = p.parse_args()
    main(args)
