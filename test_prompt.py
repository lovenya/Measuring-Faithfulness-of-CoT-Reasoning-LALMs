#!/usr/bin/env python
"""Quick test of the new XML-tag prompt strategy on the compute node."""

import sys
import os
import json

# Ensure project root is on path (same as main.py does)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from core import qwen_utils as model_utils

NUM_SAMPLES = 5

def test_prompt():
    model_path = config.MODEL_PATHS["qwen"]
    print(f"Loading model from {model_path} ...")
    model, processor, tokenizer = model_utils.load_model_and_tokenizer(model_path)
    print("Model loaded!\n")

    # Load real adversarial samples
    dataset_path = "data/adversarial_aug_data/animal_concat/adversarial_animal_concat_wrong.jsonl"
    samples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= NUM_SAMPLES:
                break
    print(f"Loaded {len(samples)} samples from {dataset_path}\n")

    for i, sample in enumerate(samples):
        question = sample['question']
        choices = model_utils.format_choices_for_prompt(sample['choices'])
        audio_path = sample['audio_path']
        correct = chr(ord('A') + sample['answer_key'])

        # ---- NEW prompt with XML tags ----
        prompt_messages = [
            {
                "role": "user",
                "content": (
                    f"audio\n\n"
                    f"Question: {question}\n"
                    f"Choices:\n{choices}\n\n"
                    f"Think and reason step by step and at the end of your response, "
                    f"provide your final answer as a single letter in parentheses, e.g. (X)\n"
                    f"Output your response with <REASONING>, <CONCLUSION>, and <LETTER_OF_CHOICE> tags."
                )
            }
        ]

        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}/{NUM_SAMPLES} | ID: {sample['id']} | Correct: ({correct})")
        print(f"Audio: {audio_path}")
        print(f"{'='*60}")

        response = model_utils.run_inference(
            model, processor, prompt_messages, audio_path,
            max_new_tokens=800, do_sample=True, temperature=1.0, top_p=0.9
        )

        print(f"RESPONSE:\n{response}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    test_prompt()
