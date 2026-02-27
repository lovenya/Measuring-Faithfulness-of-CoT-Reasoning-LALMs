#!/usr/bin/env python3
# experiments/testing/prompt_variance_test.py

"""
Prompt Variance Test Experiment

Investigates why the baseline model shows high variance across chains.
Tests configurable prompt strategies and temperatures side-by-side.

Strategies:
  two_turn_cot: Current baseline (Turn 1: CoT, Turn 2: answer). Controllable temperature.
  one_turn_cot: Single turn asking for step-by-step reasoning + answer on last line.
  no_cot:       Zero-shot, no reasoning. Just asks for the answer letter.

Output goes to: results/{model}/test/variance_{strategy}_t{temperature}_{model}_{dataset}.jsonl

Usage:
    python main.py --model qwen --experiment prompt_variance_test --dataset mmar \\
        --strategy two_turn_cot --temperature 0.1 --num-samples 10
    python main.py --model qwen --experiment prompt_variance_test --dataset mmar \\
        --strategy no_cot --num-samples 10
"""

import os
import json
import time
import logging
from pathlib import Path
from tqdm import tqdm

EXPERIMENT_TYPE = "foundational"

VALID_STRATEGIES = ['two_turn_cot', 'one_turn_cot', 'no_cot']


def run_strategy(model, processor, model_utils, audio_path: str,
                 question: str, choices: str, strategy: str,
                 temperature: float, top_p: float, top_k: int) -> dict:
    """Executes a single trial for the given strategy and returns outputs."""

    if strategy == 'two_turn_cot':
        # --- Same structure as baseline, but with controllable temperature ---
        cot_messages = [
            {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
            {"role": "assistant", "content": "Let's think step by step:"}
        ]
        generated_cot = model_utils.run_inference(
            model, processor, cot_messages, audio_path,
            max_new_tokens=768, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k
        )
        sanitized_cot = model_utils.sanitize_cot(generated_cot)

        final_messages = [
            {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
            {"role": "assistant", "content": sanitized_cot},
            {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else."}
        ]
        final_answer_text = model_utils.run_inference(
            model, processor, final_messages, audio_path,
            max_new_tokens=10, do_sample=False
        )
        return {
            "final_answer_raw": final_answer_text,
            "generated_cot": generated_cot,
            "sanitized_cot": sanitized_cot,
        }

    elif strategy == 'one_turn_cot':
        # --- Single turn: think step by step, answer on last line ---
        messages = [
            {"role": "user", "content": (
                f"audio\n\nQuestion: {question}\nChoices:\n{choices}\n"
                "Think step by step and then provide your final answer "
                "as a single letter in parentheses on the last line."
            )},
        ]
        response = model_utils.run_inference(
            model, processor, messages, audio_path,
            max_new_tokens=768, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k
        )
        # Extract the last non-empty line as the answer
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        final_answer_text = lines[-1] if lines else response

        return {
            "final_answer_raw": final_answer_text,
            "generated_cot": response,
            "sanitized_cot": response,
        }

    elif strategy == 'no_cot':
        # --- Zero-shot, no reasoning, deterministic ---
        messages = [
            {"role": "user", "content": (
                f"audio\n\nQuestion: {question}\nChoices:\n{choices}\n"
                "Provide your final answer as a single letter in parentheses, "
                "and nothing else."
            )},
        ]
        response = model_utils.run_inference(
            model, processor, messages, audio_path,
            max_new_tokens=10, do_sample=False
        )
        return {
            "final_answer_raw": response,
            "generated_cot": "",
            "sanitized_cot": "",
        }

    raise ValueError(f"Unknown strategy: {strategy}")


def run(model, processor, tokenizer, model_utils, data_samples, config):
    """Orchestrates the prompt variance test experiment."""

    dataset_name = config.DATASET_NAME
    model_alias = config.MODEL_ALIAS
    num_chains = config.NUM_CHAINS_PER_QUESTION

    # Read strategy, temperature, top_p, top_k from config (set by main.py argparser)
    strategy = getattr(config, 'STRATEGY', 'two_turn_cot')
    temperature = getattr(config, 'TEMPERATURE', None)
    top_p = getattr(config, 'TOP_P', None)
    top_k = getattr(config, 'TOP_K', None)

    # Set model-specific defaults if not provided by the user
    if model_alias == 'qwen':
        if temperature is None: temperature = 1.0
        if top_p is None: top_p = 0.01
        if top_k is None: top_k = 0
    elif 'salmonn' in model_alias:
        if temperature is None: temperature = 1.0
        if top_p is None: top_p = 0.9
        if top_k is None: top_k = 50
    elif model_alias in ('flamingo', 'flamingo_hf'):
        if temperature is None: temperature = 0.7
        if top_p is None: top_p = 0.8
        if top_k is None: top_k = 20
    else:
        # Fallback defaults
        if temperature is None: temperature = 1.0
        if top_p is None: top_p = 0.9
        if top_k is None: top_k = 50

    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Invalid strategy '{strategy}'. Must be one of: {VALID_STRATEGIES}")

    # --- Override output path to results/{model}/test/ ---
    test_dir = Path(config.RESULTS_DIR) / model_alias / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"variance_{strategy}_t{temperature}_p{top_p}_k{top_k}_{model_alias}_{dataset_name}.jsonl"
    output_path = test_dir / output_filename

    logging.info(f"--- Running Prompt Variance Test ---")
    logging.info(f"  Model:       {model_alias.upper()}")
    logging.info(f"  Dataset:     {dataset_name}")
    logging.info(f"  Strategy:    {strategy}")
    logging.info(f"  Temperature: {temperature}")
    logging.info(f"  Top-P:       {top_p}")
    logging.info(f"  Top-K:       {top_k}")
    logging.info(f"  Chains:      {num_chains}")
    logging.info(f"  Output:      {output_path}")

    # Apply num_samples limit
    samples = data_samples
    if config.NUM_SAMPLES_TO_RUN > 0:
        samples = samples[:config.NUM_SAMPLES_TO_RUN]

    total_samples = len(samples)
    logging.info(f"  Samples:     {total_samples}")

    # --- Restartability ---
    completed_trials = set()
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed_trials.add((data['id'], data['chain_id']))
                except (json.JSONDecodeError, KeyError):
                    pass
        logging.info(f"Found {len(completed_trials)} completed trials. They will be skipped.")

    new_count = 0
    skipped_count = 0
    error_count = 0
    start_time = time.time()

    # --- Main Loop ---
    with open(output_path, 'a') as f:
        for sample_idx, sample in enumerate(tqdm(samples, desc=f"Samples ({strategy})")):
            sample_id = sample['id']
            question = sample['question']
            choices_formatted = model_utils.format_choices_for_prompt(sample['choices'])
            audio_path = sample.get('audio_path', '')
            correct_choice = chr(ord('A') + sample['answer_key'])

            if not audio_path or not os.path.exists(audio_path):
                logging.warning(f"Audio not found for sample {sample_id}: {audio_path}")
                error_count += 1
                continue

            for chain_id in range(1, num_chains + 1):
                trial_key = (sample_id, chain_id)

                if trial_key in completed_trials:
                    skipped_count += 1
                    continue

                try:
                    outputs = run_strategy(
                        model, processor, model_utils,
                        audio_path, question, choices_formatted,
                        strategy, temperature, top_p, top_k
                    )

                    parsed_choice = model_utils.parse_answer(outputs["final_answer_raw"])

                    result = {
                        "id": sample_id,
                        "chain_id": chain_id,
                        "strategy": strategy,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "predicted_choice": parsed_choice,
                        "correct_choice": correct_choice,
                        "is_correct": parsed_choice == correct_choice,
                        "final_answer_raw": outputs["final_answer_raw"],
                        "generated_cot": outputs["generated_cot"],
                        "sanitized_cot": outputs["sanitized_cot"],
                        "audio_path": audio_path,
                        "question": question,
                        "choices": choices_formatted,
                    }

                    # Preserve metadata
                    if 'track' in sample:
                        result['track'] = sample['track']
                    if 'source' in sample:
                        result['source'] = sample['source']
                    if 'hop_type' in sample:
                        result['hop_type'] = sample['hop_type']

                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()
                    new_count += 1

                except Exception as e:
                    logging.error(f"Error on sample {sample_id}, chain {chain_id}: {e}")
                    error_count += 1

            # Progress logging
            if (sample_idx + 1) % 5 == 0 or sample_idx == total_samples - 1:
                elapsed = time.time() - start_time
                avg_per_sample = elapsed / (sample_idx + 1)
                remaining = avg_per_sample * (total_samples - sample_idx - 1)
                logging.info(
                    f"Progress: {sample_idx + 1}/{total_samples} | "
                    f"New: {new_count} | Skipped: {skipped_count} | Errors: {error_count} | "
                    f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m"
                )

    total_elapsed = time.time() - start_time
    logging.info(f"--- Prompt Variance Test Complete ---")
    logging.info(f"  Strategy:    {strategy}")
    logging.info(f"  Temperature: {temperature}")
    logging.info(f"  Top-P:       {top_p}")
    logging.info(f"  Top-K:       {top_k}")
    logging.info(f"  New trials:  {new_count}")
    logging.info(f"  Skipped:     {skipped_count}")
    logging.info(f"  Errors:      {error_count}")
    logging.info(f"  Total time:  {total_elapsed/60:.1f} minutes")
    logging.info(f"  Output:      {output_path}")
