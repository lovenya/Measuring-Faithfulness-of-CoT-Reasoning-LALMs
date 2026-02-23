# experiments/audio_interventions/jasco_masking.py

"""
JASCO Masking Experiment: Joint Audio-Speech Co-Reasoning (Stage 1)

Tests whether models rely on audio sounds, speech, or both sub-modalities
by running inference on incrementally masked audio variants:
  - baseline (full_audio): original audio (audio + speech)
  - speech_masked (10%-100%): scattered silence applied to the speech segment
  - audio_masked (10%-100%): scattered silence applied to the environmental audio

For each audio variant, the model is prompted with a single question:
  "Based on both the spoken text and the audio sound, think step by step
   and infer what the speakers are doing specifically?"

The model's free-form responses are saved for Stage 2 evaluation,
where an LLM-as-a-Judge (Mistral) scores each prediction on a 0-2 scale.

Usage:
    python main.py --model qwen --experiment jasco_masking --dataset jasco
    python main.py --model qwen --experiment jasco_masking --dataset jasco --num-samples 5
"""

import os
import json
import logging

# This is a 'foundational' experiment — it operates on its own data
# and does not require baseline results for consistency comparison.
EXPERIMENT_TYPE = "foundational"

# The single prompt used for all conditions
JASCO_PROMPT = (
    "Based on both the spoken text and the audio sound, "
    "think step by step and infer what the speakers are doing specifically?"
)


def run_trial(model, processor, tokenizer, model_utils, audio_path: str) -> dict:
    """
    Runs inference on a single JASCO audio variant using the single prompt.
    Returns the model's free-form response.
    """
    messages = [
        {"role": "user", "content": f"audio\\n\\n{JASCO_PROMPT}"},
    ]

    response = model_utils.run_inference(
        model, processor, messages, audio_path,
        max_new_tokens=512, do_sample=True, temperature=1.0, top_p=0.9
    )

    return {
        "prompt": JASCO_PROMPT,
        "audio_path": audio_path,
        "model_output": response,
    }


def run(model, processor, tokenizer, model_utils, data_samples, config):
    """
    Orchestrates the JASCO Masking experiment (Stage 1).

    For each of the 80 audio samples, iterates over all 21 audio conditions
    (baseline + 10 speech_masked + 10 audio_masked) and runs inference.

    Supports:
    - --num-samples to limit number of audio samples
    - --num-chains for repeated runs (default: 1)
    - Restartable design (skips completed trials)
    """
    output_path = config.OUTPUT_PATH
    num_chains = config.NUM_CHAINS_PER_QUESTION

    logging.info(f"--- Running JASCO Masking Experiment (Stage 1) ---")
    logging.info(f"  Model:   {config.MODEL_ALIAS.upper()}")
    logging.info(f"  Chains:  {num_chains}")
    logging.info(f"  Output:  {output_path}")

    # --- Build the list of all audio conditions to iterate ---
    # Each sample in jasco_masked_standardized.jsonl has an 'audio_paths' dict
    # with keys like: baseline, speech_10, speech_20, ..., audio_10, audio_20, ...
    CONDITION_KEYS = ['baseline']
    for pct in range(10, 101, 10):
        CONDITION_KEYS.append(f'speech_{pct}')
    for pct in range(10, 101, 10):
        CONDITION_KEYS.append(f'audio_{pct}')

    logging.info(f"  Conditions per sample: {len(CONDITION_KEYS)}")

    # --- Restartability Logic ---
    completed_trials = set()
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking for completed work...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Key: (sample_id, condition, chain_id)
                    completed_trials.add((
                        data['id'],
                        data['condition'],
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
            audio_paths = sample.get('audio_paths', {})

            for condition in CONDITION_KEYS:
                audio_path = audio_paths.get(condition)

                if not audio_path or not os.path.exists(audio_path):
                    logging.warning(
                        f"Audio not found for sample {sample_id}, "
                        f"condition {condition}: {audio_path}"
                    )
                    continue

                for chain_id in range(1, num_chains + 1):
                    total_expected += 1
                    trial_key = (sample_id, condition, chain_id)

                    if trial_key in completed_trials:
                        skipped_count += 1
                        continue

                    # Run inference
                    try:
                        trial_result = run_trial(
                            model, processor, tokenizer, model_utils,
                            audio_path
                        )
                    except Exception as e:
                        logging.error(
                            f"Error on sample {sample_id}, "
                            f"condition {condition}, chain {chain_id}: {e}"
                        )
                        continue

                    # Build result entry
                    result = {
                        'id': sample_id,
                        'condition': condition,
                        'chain_id': chain_id,
                        'audio_path': audio_path,
                        'prompt': trial_result['prompt'],
                        'model_output': trial_result['model_output'],
                        'correct_answer': sample.get('correct_answer', ''),
                        'target_keywords': sample.get('target_keywords', []),
                        'audio_sound': sample.get('audio_sound', ''),
                        'spoken_text': sample.get('spoken_text', ''),
                    }

                    f.write(json.dumps(result) + '\n')
                    f.flush()
                    new_count += 1

            # Progress logging
            if (sample_idx + 1) % 5 == 0 or sample_idx == total_samples - 1:
                logging.info(
                    f"Progress: {sample_idx + 1}/{total_samples} samples | "
                    f"New: {new_count} | Skipped: {skipped_count}"
                )

    logging.info(f"--- JASCO Masking Experiment (Stage 1) Complete ---")
    logging.info(f"  Total expected: {total_expected}")
    logging.info(f"  New trials:     {new_count}")
    logging.info(f"  Skipped:        {skipped_count}")
    logging.info(f"  Results:        {output_path}")
