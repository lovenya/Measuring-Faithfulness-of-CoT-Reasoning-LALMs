#!/usr/bin/env python3
"""
Qwen2-Audio-Instruct-7B Inference Script (chat-template–corrected)
For SAKURA and MMAR datasets on Compute Canada cluster
"""

import json
import random
import torch
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import argparse
import os

class AudioInferenceEngine:
    def __init__(self, model_path="./Qwen2-Audio-7B-Instruct", device="auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device in ("auto","cuda") else "cpu")
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

    def load_audio(self, audio_path, target_sr=16000):
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
            print(f"Loaded {audio.shape[0]} samples (sr={sr}) from {audio_path}")
            return audio
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None

    def format_choices(self, choices):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        return " ".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))

    def create_conversation(self, question, choices, audio_path):
        full_q = f"{question} {self.format_choices(choices)}"
        return [
            {"role":"user","content":[
                {"type":"audio","audio_path":audio_path},
                {"type":"text","text": full_q}
            ]}
        ]

    def run_inference(self, sample):
        audio = self.load_audio(sample["audio_path"])
        if audio is None:
            return None, "Failed to load audio"
        # Build chat prompt
        conversation = self.create_conversation(sample["question"], sample["choices"], sample["audio_path"])
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        # Prepare inputs
        inputs = self.processor(
            text=text,
            audio=[audio],
            return_tensors="pt",
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Generate
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=50
            )
        # Decode after prompt
        start = inputs['input_ids'].shape[1]
        raw = self.processor.batch_decode(
            gen_ids[:, start:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        return raw, None


def load_random_sample(path):
    with open(path) as f:
        data = [json.loads(l) for l in f]
    return random.choice(data)


def fix_path(rel):
    return rel if rel.startswith('data/') else os.path.join('data', rel)


def resolve_answer(sample):
    key = sample.get('answer_key')
    try:
        idx = int(key)
    except:
        idx = ord(str(key).lower()) - ord('a')
    choices = sample.get('choices', [])
    return choices[idx] if 0 <= idx < len(choices) else None


def main():
    parser = argparse.ArgumentParser(description="Chat-template inference – corrected")
    parser.add_argument("--model", default="./Qwen2-Audio-7B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", choices=["sakura","mmar","both"], default="both")
    args = parser.parse_args()

    engine = AudioInferenceEngine(args.model, args.device)
    tasks = {}
    if args.dataset in ('sakura','both'):
        tasks.update({
            name: f"data/sakura/{typ}/" + name + "_test_standardized.jsonl"
            for typ,name in [('animal','sakura_animal'),('emotion','sakura_emotion'),('gender','sakura_gender'),('language','sakura_language')]
        })
    if args.dataset in ('mmar','both'):
        tasks['mmar'] = 'data/mmar/mmar_test_standardized.jsonl'

    for name,path in tasks.items():
        if not os.path.exists(path):
            print(f"Missing {path}")
            continue
        print(f"\n=== {name} ===")
        sample = load_random_sample(path)
        sample['audio_path'] = fix_path(sample['audio_path'])
        print(f"Q: {sample['question']}, Choices: {sample['choices']}")
        print(f"Ground truth: {resolve_answer(sample)}")
        resp, err = engine.run_inference(sample)
        if err: print("Error:", err)
        else: print("Model says:", resp)

if __name__ == '__main__':
    main()
