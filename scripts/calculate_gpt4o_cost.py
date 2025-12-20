#!/usr/bin/env python3
"""
Script to estimate GPT-4o API costs for all Qwen experiments.

Audio inference counts per experiment (based on experiment scripts):
- baseline: 2 (CoT generation + final answer)
- adding_mistakes: per sentence: 1 text-only (mistake gen) + 2 audio (continue + final)
- paraphrasing: per sentence: 1 text-only (paraphrase) + 1 audio (trial)
- early_answering: 1 audio (answer extraction at each prefix point)
- filler_text variants: 1 audio (final answer)
- no_cot / no_reasoning: 1 audio (direct answer)
- robustness_to_noise: 1 audio per record (each record = different SNR)
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import tiktoken
import soundfile as sf

RESULTS_DIR = Path("results/qwen")
AUDIO_TOKENS_PER_SECOND = 10
PRICE_AUDIO_INPUT_PER_M = 40.00
PRICE_TEXT_INPUT_PER_M = 2.50
PRICE_TEXT_OUTPUT_PER_M = 10.00

EXPERIMENTS = [
    "baseline", "adding_mistakes", "early_answering", "filler_text",
    "flipped_partial_filler_text", "random_partial_filler_text",
    "partial_filler_text", "paraphrasing", "no_cot", "no_reasoning",
    "robustness_to_noise",
]
DATASETS = ["mmar", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]

# Audio inferences per sample (how many times audio is sent to model per record in JSONL)
AUDIO_MULTIPLIER = {
    "baseline": 2,  # CoT gen + final answer
    "adding_mistakes": 2,  # continue_reasoning + run_final_trial (per sentence, but each record = 1 sentence)
    "paraphrasing": 1,  # run_paraphrasing_trial (paraphrase is text-only)
    "early_answering": 1,  # answer extraction
    "filler_text": 1,
    "flipped_partial_filler_text": 1,
    "random_partial_filler_text": 1,
    "partial_filler_text": 1,
    "no_cot": 1,
    "no_reasoning": 1,
    "robustness_to_noise": 1,  # each record already has its own noisy audio
}

tokenizer = tiktoken.get_encoding("cl100k_base")
audio_duration_cache = {}

def count_text_tokens(text):
    if not text: return 0
    return len(tokenizer.encode(text))

def get_audio_duration(audio_path):
    if audio_path in audio_duration_cache:
        return audio_duration_cache[audio_path]
    try:
        if Path(audio_path).exists():
            audio_duration_cache[audio_path] = sf.info(str(audio_path)).duration
        else:
            audio_duration_cache[audio_path] = 0.0
    except:
        audio_duration_cache[audio_path] = 0.0
    return audio_duration_cache[audio_path]

def analyze_jsonl_file(filepath, experiment):
    results = {"num_samples": 0, "audio_tokens": 0, "text_input_tokens": 0, "text_output_tokens": 0}
    if not filepath.exists(): return results
    
    audio_mult = AUDIO_MULTIPLIER.get(experiment, 1)
    
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = json.loads(line)
            except:
                continue
            
            results["num_samples"] += 1
            
            # Text input
            question = record.get("question", "")
            choices = record.get("choices", "")
            if not question and "final_prompt_messages" in record:
                for msg in record["final_prompt_messages"]:
                    if msg.get("role") == "user" and "Question:" in msg.get("content", ""):
                        question = msg["content"]
                        break
            text_input = f"{question}\n{choices}" if choices else question
            results["text_input_tokens"] += count_text_tokens(text_input)
            
            # Text output - try multiple sources
            output_text = record.get("generated_cot", "")
            if not output_text and "final_prompt_messages" in record:
                for msg in record["final_prompt_messages"]:
                    if msg.get("role") == "assistant" and msg.get("content"):
                        output_text = msg["content"]
                        break
            if not output_text:
                output_text = record.get("final_answer_raw", "")
            results["text_output_tokens"] += count_text_tokens(output_text)
            
            # Audio - find path and apply multiplier
            audio_path = record.get("audio_path", "") or record.get("noisy_audio_path_used", "")
            if audio_path:
                duration = get_audio_duration(audio_path)
                # Per sample: audio tokens * multiplier (for number of inference calls)
                results["audio_tokens"] += int(duration * AUDIO_TOKENS_PER_SECOND * audio_mult)
    
    return results

def find_experiment_files():
    files = []
    for exp in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp
        if not exp_dir.exists(): continue
        for dataset in DATASETS:
            for pattern in [f"{exp}_qwen_{dataset}.jsonl", f"{exp}_{dataset}.jsonl"]:
                filepath = exp_dir / pattern
                if filepath.exists():
                    files.append({"experiment": exp, "dataset": dataset, "filepath": filepath})
                    break
    return files

def calculate_cost(audio_tokens, text_input_tokens, text_output_tokens):
    return {
        "audio_cost": (audio_tokens / 1_000_000) * PRICE_AUDIO_INPUT_PER_M,
        "text_input_cost": (text_input_tokens / 1_000_000) * PRICE_TEXT_INPUT_PER_M,
        "text_output_cost": (text_output_tokens / 1_000_000) * PRICE_TEXT_OUTPUT_PER_M,
        "total_cost": (audio_tokens / 1_000_000) * PRICE_AUDIO_INPUT_PER_M + 
                      (text_input_tokens / 1_000_000) * PRICE_TEXT_INPUT_PER_M +
                      (text_output_tokens / 1_000_000) * PRICE_TEXT_OUTPUT_PER_M,
    }

def main():
    print("=" * 70)
    print("GPT-4o Cost Estimation for Qwen Experiments")
    print("=" * 70)
    print(f"\nAudio: {AUDIO_TOKENS_PER_SECOND} tokens/sec")
    print(f"Pricing: Audio ${PRICE_AUDIO_INPUT_PER_M}/1M, Text in ${PRICE_TEXT_INPUT_PER_M}/1M, Text out ${PRICE_TEXT_OUTPUT_PER_M}/1M\n")
    
    files = find_experiment_files()
    print(f"Found {len(files)} experiment files.\n")
    
    totals = {"num_samples": 0, "audio_tokens": 0, "text_input_tokens": 0, "text_output_tokens": 0}
    by_experiment = defaultdict(lambda: {"num_samples": 0, "audio_tokens": 0, "text_input_tokens": 0, "text_output_tokens": 0})
    by_dataset = defaultdict(lambda: {"num_samples": 0, "audio_tokens": 0, "text_input_tokens": 0, "text_output_tokens": 0})
    
    for file_info in files:
        exp, dataset, filepath = file_info["experiment"], file_info["dataset"], file_info["filepath"]
        print(f"Processing: {exp}/{dataset}...")
        results = analyze_jsonl_file(filepath, exp)
        
        for key in ["num_samples", "audio_tokens", "text_input_tokens", "text_output_tokens"]:
            totals[key] += results[key]
            by_experiment[exp][key] += results[key]
            by_dataset[dataset][key] += results[key]
    
    print("\n" + "=" * 70 + "\nRESULTS BY EXPERIMENT\n" + "=" * 70)
    for exp in sorted(by_experiment.keys()):
        data = by_experiment[exp]
        costs = calculate_cost(data["audio_tokens"], data["text_input_tokens"], data["text_output_tokens"])
        print(f"\n{exp}: {data['num_samples']:,} samples, {data['audio_tokens']:,} audio, {data['text_input_tokens']:,} in, {data['text_output_tokens']:,} out -> ${costs['total_cost']:.2f}")
    
    print("\n" + "=" * 70 + "\nRESULTS BY DATASET\n" + "=" * 70)
    for dataset in sorted(by_dataset.keys()):
        data = by_dataset[dataset]
        costs = calculate_cost(data["audio_tokens"], data["text_input_tokens"], data["text_output_tokens"])
        print(f"\n{dataset}: {data['num_samples']:,} samples, {data['audio_tokens']:,} audio, {data['text_input_tokens']:,} in, {data['text_output_tokens']:,} out -> ${costs['total_cost']:.2f}")
    
    print("\n" + "=" * 70 + "\nGRAND TOTALS\n" + "=" * 70)
    total_costs = calculate_cost(totals["audio_tokens"], totals["text_input_tokens"], totals["text_output_tokens"])
    print(f"\nSamples: {totals['num_samples']:,}")
    print(f"Audio tokens: {totals['audio_tokens']:,} (${total_costs['audio_cost']:.2f})")
    print(f"Text input: {totals['text_input_tokens']:,} (${total_costs['text_input_cost']:.2f})")
    print(f"Text output: {totals['text_output_tokens']:,} (${total_costs['text_output_cost']:.2f})")
    print(f"TOTAL: ${total_costs['total_cost']:.2f}")
    print(f"\nAudio: {totals['audio_tokens'] / AUDIO_TOKENS_PER_SECOND:.0f} sec = {totals['audio_tokens'] / AUDIO_TOKENS_PER_SECOND / 3600:.2f} hours")

if __name__ == "__main__":
    main()
