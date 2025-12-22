#!/usr/bin/env python3
"""
Script to estimate GPT-4o API costs for all Qwen experiments.
ENHANCED VERSION: Tracks total API calls and categorizes them into text-only vs multimodal.

Audio inference counts per experiment (based on experiment scripts):
- baseline: 2 (CoT generation + final answer) - MULTIMODAL
- adding_mistakes: per sentence: 1 text-only (mistake gen) + 2 audio (continue + final)
- paraphrasing: per sentence: 1 text-only (paraphrase) + 1 audio (trial)
- early_answering: 1 audio (answer extraction at each prefix point) - MULTIMODAL
- filler_text variants: 1 audio (final answer) - MULTIMODAL
- no_cot / no_reasoning: 1 audio (direct answer) - MULTIMODAL
- robustness_to_noise: 1 audio per record (each record = different SNR) - MULTIMODAL
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

# MULTIMODAL inferences per sample (how many times audio is sent to model per record in JSONL)
# These are API calls that include audio input
MULTIMODAL_MULTIPLIER = {
    "baseline": 2,  # CoT gen + final answer (both with audio)
    "adding_mistakes": 2,  # continue_reasoning + run_final_trial (both with audio, after text-only mistake gen)
    "paraphrasing": 1,  # run_paraphrasing_trial (with audio, after text-only paraphrase gen)
    "early_answering": 1,  # answer extraction at each prefix (with audio)
    "filler_text": 1,  # final answer (with audio)
    "flipped_partial_filler_text": 1,  # final answer (with audio)
    "random_partial_filler_text": 1,  # final answer (with audio)
    "partial_filler_text": 1,  # final answer (with audio)
    "no_cot": 1,  # direct answer (with audio)
    "no_reasoning": 1,  # direct answer (with audio)
    "robustness_to_noise": 1,  # answer with noisy audio
}

# Kept for backward compatibility with cost calculation
AUDIO_MULTIPLIER = MULTIMODAL_MULTIPLIER

# TEXT-ONLY inferences per sample
# These are API calls that only use text, no audio (for generating perturbations)
TEXT_ONLY_MULTIPLIER = {
    "baseline": 0,  # No text-only calls
    "adding_mistakes": 1,  # 1 text-only call for mistake generation
    "paraphrasing": 1,  # 1 text-only call for paraphrase generation
    "early_answering": 0,  # No text-only calls
    "filler_text": 0,  # No text-only calls
    "flipped_partial_filler_text": 0,  # No text-only calls
    "random_partial_filler_text": 0,  # No text-only calls
    "partial_filler_text": 0,  # No text-only calls
    "no_cot": 0,  # No text-only calls
    "no_reasoning": 0,  # No text-only calls
    "robustness_to_noise": 0,  # No text-only calls
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
    results = {
        "num_samples": 0,
        "audio_tokens": 0,
        "text_input_tokens": 0,
        "text_output_tokens": 0,
        "multimodal_api_calls": 0,  # NEW: API calls with audio
        "text_only_api_calls": 0,   # NEW: API calls without audio (text-only)
    }
    if not filepath.exists(): return results
    
    multimodal_mult = MULTIMODAL_MULTIPLIER.get(experiment, 1)
    text_only_mult = TEXT_ONLY_MULTIPLIER.get(experiment, 0)
    
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = json.loads(line)
            except:
                continue
            
            results["num_samples"] += 1
            
            # Count API calls
            results["multimodal_api_calls"] += multimodal_mult  # Each sample makes multimodal_mult multimodal calls
            results["text_only_api_calls"] += text_only_mult  # Each sample makes text_only_mult text calls
            
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
                results["audio_tokens"] += int(duration * AUDIO_TOKENS_PER_SECOND * multimodal_mult)
    
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
    print("=" * 80)
    print("GPT-4o Cost Estimation for Qwen Experiments (WITH API CALL TRACKING)")
    print("=" * 80)
    print(f"\nAudio: {AUDIO_TOKENS_PER_SECOND} tokens/sec")
    print(f"Pricing: Audio ${PRICE_AUDIO_INPUT_PER_M}/1M, Text in ${PRICE_TEXT_INPUT_PER_M}/1M, Text out ${PRICE_TEXT_OUTPUT_PER_M}/1M\n")
    
    files = find_experiment_files()
    print(f"Found {len(files)} experiment files.\n")
    
    totals = {
        "num_samples": 0,
        "audio_tokens": 0,
        "text_input_tokens": 0,
        "text_output_tokens": 0,
        "multimodal_api_calls": 0,
        "text_only_api_calls": 0,
    }
    by_experiment = defaultdict(lambda: {
        "num_samples": 0,
        "audio_tokens": 0,
        "text_input_tokens": 0,
        "text_output_tokens": 0,
        "multimodal_api_calls": 0,
        "text_only_api_calls": 0,
    })
    by_dataset = defaultdict(lambda: {
        "num_samples": 0,
        "audio_tokens": 0,
        "text_input_tokens": 0,
        "text_output_tokens": 0,
        "multimodal_api_calls": 0,
        "text_only_api_calls": 0,
    })
    
    for file_info in files:
        exp, dataset, filepath = file_info["experiment"], file_info["dataset"], file_info["filepath"]
        print(f"Processing: {exp}/{dataset}...")
        results = analyze_jsonl_file(filepath, exp)
        
        for key in totals.keys():
            totals[key] += results[key]
            by_experiment[exp][key] += results[key]
            by_dataset[dataset][key] += results[key]
    
    print("\n" + "=" * 80 + "\nRESULTS BY EXPERIMENT\n" + "=" * 80)
    for exp in sorted(by_experiment.keys()):
        data = by_experiment[exp]
        costs = calculate_cost(data["audio_tokens"], data["text_input_tokens"], data["text_output_tokens"])
        total_api_calls = data["multimodal_api_calls"] + data["text_only_api_calls"]
        print(f"\n{exp}:")
        print(f"  Samples: {data['num_samples']:,}")
        print(f"  API calls: {total_api_calls:,} (Multimodal: {data['multimodal_api_calls']:,}, Text-only: {data['text_only_api_calls']:,})")
        print(f"  Tokens: {data['audio_tokens']:,} audio, {data['text_input_tokens']:,} in, {data['text_output_tokens']:,} out")
        print(f"  Cost: ${costs['total_cost']:.2f}")
    
    print("\n" + "=" * 80 + "\nRESULTS BY DATASET\n" + "=" * 80)
    for dataset in sorted(by_dataset.keys()):
        data = by_dataset[dataset]
        costs = calculate_cost(data["audio_tokens"], data["text_input_tokens"], data["text_output_tokens"])
        total_api_calls = data["multimodal_api_calls"] + data["text_only_api_calls"]
        print(f"\n{dataset}:")
        print(f"  Samples: {data['num_samples']:,}")
        print(f"  API calls: {total_api_calls:,} (Multimodal: {data['multimodal_api_calls']:,}, Text-only: {data['text_only_api_calls']:,})")
        print(f"  Tokens: {data['audio_tokens']:,} audio, {data['text_input_tokens']:,} in, {data['text_output_tokens']:,} out")
        print(f"  Cost: ${costs['total_cost']:.2f}")
    
    print("\n" + "=" * 80 + "\nGRAND TOTALS\n" + "=" * 80)
    total_costs = calculate_cost(totals["audio_tokens"], totals["text_input_tokens"], totals["text_output_tokens"])
    total_api_calls = totals["multimodal_api_calls"] + totals["text_only_api_calls"]
    
    print(f"\nSamples: {totals['num_samples']:,}")
    print(f"\n=== API CALL BREAKDOWN ===")
    print(f"Total API calls: {total_api_calls:,}")
    print(f"  - Multimodal (with audio): {totals['multimodal_api_calls']:,} ({100 * totals['multimodal_api_calls'] / total_api_calls:.1f}%)")
    print(f"  - Text-only (no audio): {totals['text_only_api_calls']:,} ({100 * totals['text_only_api_calls'] / total_api_calls:.1f}%)")
    
    print(f"\n=== TOKEN BREAKDOWN ===")
    print(f"Audio tokens: {totals['audio_tokens']:,} (${total_costs['audio_cost']:.2f})")
    print(f"Text input: {totals['text_input_tokens']:,} (${total_costs['text_input_cost']:.2f})")
    print(f"Text output: {totals['text_output_tokens']:,} (${total_costs['text_output_cost']:.2f})")
    
    print(f"\n=== COST SUMMARY ===")
    print(f"TOTAL: ${total_costs['total_cost']:.2f}")
    print(f"\nAudio duration: {totals['audio_tokens'] / AUDIO_TOKENS_PER_SECOND:.0f} sec = {totals['audio_tokens'] / AUDIO_TOKENS_PER_SECOND / 3600:.2f} hours")

if __name__ == "__main__":
    main()
