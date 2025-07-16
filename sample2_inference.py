#!/usr/bin/env python3
"""
Qwen2-Audio-Instruct-7B Inference Script (raw output)
For SAKURA and MMAR datasets on Compute Canada cluster
"""

import json
import random
import torch
import librosa
import numpy as np
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from pathlib import Path
import argparse
import sys
import os

class AudioInferenceEngine:
    def __init__(self, model_path="./Qwen2-Audio-7B-Instruct", device="auto"):
        """Initialize the audio inference engine"""
        self.device = self._setup_device(device)
        print(f"Using device: {self.device}")
        
        print(f"Loading model and processor from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            local_files_only=True
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        print("Model loaded successfully!")
    
    def _setup_device(self, device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
    
    def load_audio(self, audio_path, target_sr=16000):
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
            print(f"Loaded {audio.shape[0]} samples (sr={sr}) from {audio_path}")

            return audio
        except Exception as e:
            print(f"Error loading audio from {audio_path}: {e}")
            return None
    
    def format_prompt(self, question, choices):
        # keep same formatting as before, including the <|AUDIO|> token
        formatted_choices = [f"({chr(97+i)}) {c}" for i,c in enumerate(choices)]
        return f"<|AUDIO|> {question} " + " ".join(formatted_choices)
    
    def run_inference(self, audio_path, question, choices):
        audio = self.load_audio(audio_path)
        if audio is None:
            return None, "Failed to load audio"
        
        prompt = self.format_prompt(question, choices)
        try:
            inputs = self.processor(
                text=[prompt],
                audios=[audio],
                return_tensors="pt",
                padding=True,
                sampling_rate=16000
            )
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            
            with torch.no_grad():
                for k,v in inputs.items():
                    print(k, v.shape)

                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # **Raw decode**: decode everything after the prompt without skipping SPECIAL tokens
            raw = self.processor.batch_decode(
                generate_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=False,     # keep any special tokens
                clean_up_tokenization_spaces=False
            )[0]
            
            return raw, None
        
        except Exception as e:
            return None, f"Inference error: {e}"

def load_dataset_sample(dataset_path):
    try:
        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]
        return random.choice(data)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def fix_audio_path(audio_path):
    if audio_path.startswith('data/data/'):
        return audio_path.replace('data/data/', 'data/')
    elif not audio_path.startswith('data/'):
        return os.path.join('data', audio_path)
    return audio_path

def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio Raw Inference Script")
    parser.add_argument("--model", default="./Qwen2-Audio-7B-Instruct", help="Local model path")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--dataset", choices=["sakura", "mmar", "both"], default="both")
    args = parser.parse_args()

    engine = AudioInferenceEngine(args.model, args.device)

    datasets = {}
    if args.dataset in ["sakura", "both"]:
        datasets.update({
            "sakura_animal":   "data/sakura/animal/sakura_animal_test_standardized.jsonl",
            "sakura_emotion":  "data/sakura/emotion/sakura_emotion_test_standardized.jsonl",
            "sakura_gender":   "data/sakura/gender/sakura_gender_test_standardized.jsonl",
            "sakura_language": "data/sakura/language/sakura_language_test_standardized.jsonl",
        })
    if args.dataset in ["mmar", "both"]:
        datasets["mmar"] = "data/mmar/mmar_test_standardized.jsonl"

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"Warning: Dataset file not found: {path}")
            continue

        print(f"\n=== Testing on {name} ===")
        sample = load_dataset_sample(path)
        if not sample:
            continue

        audio_path = fix_audio_path(sample["audio_path"])
        print(f"Sample ID: {sample['id']}")
        print(f"Audio path: {audio_path}")
        print(f"Question: {sample['question']}")
        print(f"Choices: {sample['choices']}")
        print(f"(Ground truth: {sample['choices'][sample['answer_key']]})\n")

        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            continue

        raw_response, err = engine.run_inference(audio_path, sample["question"], sample["choices"])
        if err:
            print(f"Inference error: {err}")
        else:
            print("=== Raw model output ===")
            print(raw_response)
            print("========================")

if __name__ == "__main__":
    main()
