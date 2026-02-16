# experiments/audio_interventions/jasco_masking.py

"""
JASCO Masking Experiment: Joint Audio-Speech Co-Reasoning

Tests whether models rely on audio sounds, speech, or both submodalities
by running inference on 3 audio variants:
  - full:        original audio (audio + speech)
  - audio_only:  speech segment removed, environmental audio only
  - speech_only: only the speech segment retained

Each variant is tested with 8 different prompts per audio clip.
Evaluation is done via keyword matching (separate script: analysis/evaluate_jasco.py).

This experiment is model-agnostic and uses the standardized JASCO JSONL.

Usage:
    python main.py --model qwen --experiment jasco_masking --dataset jasco --jasco-variant full
    python main.py --model qwen --experiment jasco_masking --dataset jasco --jasco-variant audio_only
    python main.py --model qwen --experiment jasco_masking --dataset jasco --jasco-variant speech_only
"""

import os
import json
import logging

# This is a 'foundational' experiment — it operates on its own data
# and does not require baseline results for consistency comparison.
EXPERIMENT_TYPE = "foundational"


def run_trial(model, processor, tokenizer, model_utils, prompt: str, audio_path: str) -> dict:
    """
    Runs inference on a single JASCO sample.
    
    Unlike MCQ experiments, JASCO is open-ended QA — the model generates
    a free-form response which is later evaluated via keyword matching.
    """
    messages = [
        {"role": "user", "content": f"audio\n\n{prompt}"},
    ]
    
    # Generate response (open-ended, no CoT needed)
    response = model_utils.run_inference(
        model, processor, messages, audio_path,
        max_new_tokens=256, do_sample=False, temperature=1.0, top_p=0.9
    )
    
    return {
        "prompt": prompt,
        "audio_path": audio_path,
        "model_output": response,
    }


def run(model, processor, tokenizer, model_utils, data_samples, config):
    """
    Orchestrates the JASCO Masking experiment.
    
    For each audio sample and each prompt variant, runs inference using
    the selected audio variant (full/audio_only/speech_only).
    
    Supports:
    - --jasco-variant to select audio variant
    - --num-samples to limit samples
    - --num-chains as prompt repetitions for consistency
    - Restartable design (skips completed trials)
    """
    output_path = config.OUTPUT_PATH
    variant = getattr(config, 'JASCO_VARIANT', 'full')
    num_chains = config.NUM_CHAINS_PER_QUESTION
    
    # Map variant name to the JSONL field
    variant_field_map = {
        'full': 'audio_path',
        'audio_only': 'audio_only_path',
        'speech_only': 'speech_only_path',
    }
    
    audio_field = variant_field_map.get(variant)
    if not audio_field:
        logging.error(f"Unknown JASCO variant: {variant}. Choose from: {list(variant_field_map.keys())}")
        return
    
    logging.info(f"--- Running JASCO Masking Experiment ---")
    logging.info(f"  Model:   {config.MODEL_ALIAS.upper()}")
    logging.info(f"  Variant: {variant}")
    logging.info(f"  Chains:  {num_chains}")
    logging.info(f"  Output:  {output_path}")
    
    # --- Restartability Logic ---
    completed_trials = set()
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking for completed work...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Key: (sample_id, prompt_index, chain_id)
                    completed_trials.add((
                        data['id'],
                        data['prompt_index'],
                        data['chain_id']
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue
        logging.info(f"Found {len(completed_trials)} completed trials. They will be skipped.")
    
    # Apply num_samples limit
    samples = data_samples
    if config.NUM_SAMPLES_TO_RUN > 0:
        samples = samples[:config.NUM_SAMPLES_TO_RUN]
    
    total_samples = len(samples)
    total_expected = 0
    skipped_count = 0
    new_count = 0
    
    logging.info(f"  Samples: {total_samples}")
    
    # --- Main Experiment Loop ---
    with open(output_path, 'a') as f:
        for sample_idx, sample in enumerate(samples):
            sample_id = sample['id']
            prompts = sample.get('prompts', [])
            audio_path = sample.get(audio_field)
            
            if not audio_path or not os.path.exists(audio_path):
                logging.warning(f"Audio file not found for sample {sample_id}: {audio_path}")
                continue
            
            for prompt_idx, prompt in enumerate(prompts):
                for chain_id in range(1, num_chains + 1):
                    total_expected += 1
                    trial_key = (sample_id, prompt_idx, chain_id)
                    
                    if trial_key in completed_trials:
                        skipped_count += 1
                        continue
                    
                    # Run inference
                    try:
                        trial_result = run_trial(
                            model, processor, tokenizer, model_utils,
                            prompt, audio_path
                        )
                    except Exception as e:
                        logging.error(f"Error on sample {sample_id}, prompt {prompt_idx}, chain {chain_id}: {e}")
                        continue
                    
                    # Build result entry
                    result = {
                        'id': sample_id,
                        'prompt_index': prompt_idx,
                        'chain_id': chain_id,
                        'variant': variant,
                        'audio_path': audio_path,
                        'prompt': prompt,
                        'model_output': trial_result['model_output'],
                        'target_keywords': sample.get('target_keywords', []),
                        'correct_answer': sample.get('correct_answer', ''),
                        'audio_only_answer': sample.get('audio_only_answer', ''),
                        'speech_only_answer': sample.get('speech_only_answer', ''),
                        'audio_sound': sample.get('audio_sound', ''),
                        'spoken_text': sample.get('spoken_text', ''),
                    }
                    
                    f.write(json.dumps(result) + '\n')
                    f.flush()
                    new_count += 1
            
            # Progress logging
            if (sample_idx + 1) % 10 == 0 or sample_idx == total_samples - 1:
                logging.info(
                    f"Progress: {sample_idx + 1}/{total_samples} samples | "
                    f"New: {new_count} | Skipped: {skipped_count}"
                )
    
    logging.info(f"--- JASCO Masking Experiment Complete ---")
    logging.info(f"  Total expected: {total_expected}")
    logging.info(f"  New trials:     {new_count}")
    logging.info(f"  Skipped:        {skipped_count}")
    logging.info(f"  Results:        {output_path}")
