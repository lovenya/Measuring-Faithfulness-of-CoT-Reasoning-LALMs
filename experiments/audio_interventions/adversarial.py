# experiments/audio_interventions/adversarial.py

"""
This script conducts the "Adversarial Audio" experiment, testing how robust 
audio language models are when adversarial speech content (TTS-synthesized 
answer choices) is injected into the original audio.

Methodology:
1. For each audio sample, we have two adversarial variants:
   - "correct": The correct answer choice is synthesized to speech and 
     concatenated/overlayed onto the original audio.
   - "wrong": A random incorrect answer choice is synthesized to speech 
     and concatenated/overlayed onto the original audio.

2. We run the exact same two-turn CoT prompting as the baseline experiment:
   - Turn 1: Generate a Chain-of-Thought reasoning
   - Turn 2: Elicit the final answer from the CoT

3. A robust model should focus on the original audio content and not be
   swayed by the adversarial speech. If accuracy/consistency drops, this
   indicates a bias toward semantic speech content in the audio.

This is a FOUNDATIONAL experiment - it generates fresh CoTs on the adversarial
audio. Baseline predicted choices are loaded for consistency comparison.
"""

import os
import json
import collections
import logging

# This is a 'foundational' experiment - fresh CoT generation on adversarial audio.
EXPERIMENT_TYPE = "foundational"


def load_baseline_lookup(baseline_path: str) -> dict:
    """
    Load baseline results and build a lookup by (id, chain_id) -> predicted_choice.
    This allows us to compare adversarial predictions against baseline predictions.
    """
    lookup = {}
    if not os.path.exists(baseline_path):
        logging.warning(f"Baseline results not found at '{baseline_path}'. "
                        "Consistency with baseline will not be computed.")
        return lookup
    
    with open(baseline_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                lookup[(data['id'], data['chain_id'])] = data['predicted_choice']
            except (json.JSONDecodeError, KeyError):
                continue
    
    logging.info(f"Loaded {len(lookup)} baseline predictions from '{baseline_path}'")
    return lookup


def run_adversarial_trial(model, processor, tokenizer, model_utils, question: str, choices: str, audio_path: str) -> dict:
    """
    Runs a full, two-turn adversarial trial for a single question.
    Uses the EXACT same prompting strategy as baseline.py:
    - Turn 1: CoT generation with sampling
    - Turn 2: Final answer with greedy decoding
    """
    # --- Turn 1: Generate the Chain-of-Thought ---
    cot_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": "Let's think step by step:"}
    ]
    
    generated_cot = model_utils.run_inference(
        model, processor, cot_prompt_messages, audio_path, 
        max_new_tokens=768, do_sample=True, temperature=1.0, top_p=0.9   
    )

    # --- Pre-computation Step: Sanitize the CoT ---
    sanitized_cot = model_utils.sanitize_cot(generated_cot)
        
    # --- Turn 2: Elicit the Final Answer ---
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": sanitized_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else."}
    ]
    
    final_answer_text = model_utils.run_inference(
        model, processor, final_answer_prompt_messages, audio_path, 
        max_new_tokens=50, do_sample=False, temperature=1.0, top_p=0.9
    )
    
    parsed_choice = model_utils.parse_answer(final_answer_text)

    return {
        "question": question,
        "choices": choices,
        "audio_path": audio_path,
        "generated_cot": generated_cot,
        "sanitized_cot": sanitized_cot,
        "final_answer_raw": final_answer_text,
        "predicted_choice": parsed_choice,
        "final_prompt_messages": final_answer_prompt_messages,
    }


def run(model, processor, tokenizer, model_utils, data_samples, config):
    """
    Orchestrates the adversarial experiment with a robust, restartable design.
    Mirrors the baseline experiment's structure exactly.
    """
    output_path = config.OUTPUT_PATH
    adversarial_aug = getattr(config, 'ADVERSARIAL_AUG', 'unknown')
    adversarial_variant = getattr(config, 'ADVERSARIAL_VARIANT', 'unknown')
    
    logging.info(f"--- Running Adversarial Experiment (Model: {config.MODEL_ALIAS.upper()}, "
                 f"Aug: {adversarial_aug}, Variant: {adversarial_variant}): "
                 f"Saving to {output_path} ---")

    # --- Load Baseline Predictions for Consistency ---
    # Construct baseline path: results/{model}/baseline/baseline_{model}_{dataset}.jsonl
    baseline_path = os.path.join(
        config.RESULTS_DIR, config.MODEL_ALIAS, 'baseline',
        f"baseline_{config.MODEL_ALIAS}_{config.DATASET_NAME}.jsonl"
    )
    baseline_lookup = load_baseline_lookup(baseline_path)

    # --- RESTARTABILITY LOGIC (identical to baseline.py) ---
    completed_chains = collections.defaultdict(int)
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking for completed work...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed_chains[data['id']] += 1
                except json.JSONDecodeError:
                    logging.warning(f"Found a corrupted line in {output_path}. It will be ignored.")
                    continue
    
    fully_completed_ids = {
        q_id for q_id, count in completed_chains.items() 
        if count >= config.NUM_CHAINS_PER_QUESTION
    }
    
    if fully_completed_ids:
        logging.info(f"Found {len(fully_completed_ids)} fully completed questions. They will be skipped.")

    skipped_samples_count = 0
    with open(output_path, 'a') as f:
        for i, sample in enumerate(data_samples):
            try:
                if sample['id'] in fully_completed_ids:
                    continue

                if config.VERBOSE:
                    logging.info(f"Processing sample {i+1}/{len(data_samples)}: {sample['id']}")
                
                choices_formatted = model_utils.format_choices_for_prompt(sample['choices'])
                
                chains_to_generate = config.NUM_CHAINS_PER_QUESTION - completed_chains[sample['id']]
                
                for j in range(chains_to_generate):
                    current_chain_num = completed_chains[sample['id']] + j + 1
                    chain_id = current_chain_num - 1
                    if config.VERBOSE:
                        logging.info(f"  - Generating chain {current_chain_num}/{config.NUM_CHAINS_PER_QUESTION}...")
                    
                    trial_result = run_adversarial_trial(
                        model, processor, tokenizer, model_utils,
                        sample['question'], 
                        choices_formatted,
                        sample['audio_path']
                    )
                    
                    # Add metadata
                    trial_result['id'] = sample['id']
                    trial_result['chain_id'] = chain_id
                    
                    correct_choice_letter = chr(ord('A') + sample['answer_key'])
                    trial_result['correct_choice'] = correct_choice_letter
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == correct_choice_letter)
                    
                    # Baseline consistency
                    baseline_pred = baseline_lookup.get((sample['id'], chain_id))
                    
                    # Explicitly order keys for readability
                    final_ordered_result = {
                        "id": trial_result['id'],
                        "chain_id": trial_result['chain_id'],
                        "adversarial_aug": adversarial_aug,
                        "adversarial_variant": adversarial_variant,
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": trial_result['correct_choice'],
                        "is_correct": trial_result['is_correct'],
                        "corresponding_baseline_predicted_choice": baseline_pred,
                        "is_consistent_with_baseline": (trial_result['predicted_choice'] == baseline_pred) if baseline_pred is not None else None,
                        "final_answer_raw": trial_result['final_answer_raw'],
                        "final_prompt_messages": trial_result['final_prompt_messages'],
                        "question": trial_result['question'],
                        "choices": trial_result['choices'],
                        "audio_path": trial_result['audio_path'],
                        "generated_cot": trial_result['generated_cot'],
                        "sanitized_cot": trial_result['sanitized_cot'],
                    }
                    
                    # Add optional metadata if available
                    if 'track' in sample: final_ordered_result['track'] = sample['track']
                    if 'source' in sample: final_ordered_result['source'] = sample['source']
                    if 'hop_type' in sample: final_ordered_result['hop_type'] = sample['hop_type']
                    
                    f.write(json.dumps(final_ordered_result, ensure_ascii=False) + "\n")
                    f.flush()                  
                    
            except Exception as e:
                skipped_samples_count += 1
                logging.exception(f"SKIPPING SAMPLE due to unhandled error. ID: {sample.get('id', 'N/A')}")
                continue

    # Final summary
    total_processed_in_this_run = len(data_samples) - len(fully_completed_ids)
    logging.info(f"--- Adversarial Experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    logging.info("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    logging.info(f"Total samples in dataset: {len(data_samples)}")
    logging.info(f"Adversarial augmentation: {adversarial_aug}")
    logging.info(f"Adversarial variant: {adversarial_variant}")
    logging.info(f"Baseline predictions loaded: {len(baseline_lookup)}")
    logging.info(f"Samples already complete: {len(fully_completed_ids)}")
    logging.info(f"Samples processed in this run: {total_processed_in_this_run - skipped_samples_count}")
    logging.info(f"Skipped samples due to errors in this run: {skipped_samples_count}")
    logging.info(f"Results saved to: {output_path}")
    logging.info("="*65)
