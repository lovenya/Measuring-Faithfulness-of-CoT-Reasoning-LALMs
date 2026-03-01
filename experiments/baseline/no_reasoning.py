# experiments/no_reasoning.py

"""
This script conducts the "No-Reasoning" experiment. It is a crucial foundational
baseline that measures the model's performance when it is given the exact same
prompt structure as the main baseline, but with a deliberately empty reasoning string.

This allows us to isolate and measure the model's "structural prior"—its ability
to answer a question based on the format of the prompt alone, without any
semantic reasoning content. The results from this experiment are used as an
optimization in several dependent experiments (like early_answering).
"""

import os
import json
import collections
import logging

# This is a 'foundational' experiment.
EXPERIMENT_TYPE = "foundational"

def run_no_reasoning_trial(model, processor, tokenizer, model_utils, question: str, choices: str, audio_path: str) -> dict:
    """
    Runs a single, deterministic trial with no reasoning prompt.

    Delegates to the centralized ``run_no_reasoning_trial`` from
    prompt_strategies, which selects the correct no-reasoning prompt
    based on the model backend.
    """
    from core.prompt_strategies import run_no_reasoning_trial as _run_no_reasoning

    result = _run_no_reasoning(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        model_utils=model_utils,
        question=question,
        choices=choices,
        audio_path=audio_path,
    )

    return {
        "question": question,
        "choices": choices,
        "audio_path": audio_path,
        "final_answer_raw": result.get("final_answer_raw", ""),
        "predicted_choice": result.get("predicted_choice"),
        "final_prompt_messages": result.get("final_prompt_messages", []),
    }

def run(model, processor, tokenizer, model_utils, data_samples, config):
    """
    Orchestrates the full "No-Reasoning" experiment with a robust, restartable design.
    """
    output_path = config.OUTPUT_PATH
    logging.info(f"--- Running 'No-Reasoning' Experiment (Model: {config.MODEL_ALIAS.upper()}): Saving to {output_path} ---")

    # --- RESTARTABILITY LOGIC (SIMPLIFIED FOR DETERMINISTIC EXPERIMENT) ---
    completed_ids = set()
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking for completed work...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    # For this deterministic experiment, we only need to know if an ID
                    # exists in the file. If it does, we consider it complete.
                    completed_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    continue # Ignore corrupted lines
    
    if completed_ids:
        logging.info(f"Found {len(completed_ids)} completed questions. They will be skipped.")
    # --- END OF RESTARTABILITY LOGIC ---
    
    skipped_samples_count = 0
    # Open the file in 'append' mode ('a') to preserve existing work.
    with open(output_path, 'a') as f:
        for i, sample in enumerate(data_samples):
            try:
                # If this question has already been processed, skip it instantly.
                if sample['id'] in completed_ids:
                    continue

                if config.VERBOSE:
                    logging.info(f"Processing sample {i+1}/{len(data_samples)}: {sample['id']}")
                
                choices_formatted = model_utils.format_choices_for_prompt(sample['choices'])
                
                # --- OPTIMIZATION: RUN INFERENCE ONLY ONCE ---
                # Since this is a deterministic experiment, the result will be the same
                # for all chains. We run the expensive inference call a single time...
                cached_result = run_no_reasoning_trial(
                    model, processor, tokenizer, model_utils,
                    sample['question'], 
                    choices_formatted, 
                    sample['audio_path']
                )
                # ...and then we reuse this result for each chain entry.
                
                for j in range(config.NUM_CHAINS_PER_QUESTION):
                    trial_result = cached_result.copy()

                    trial_result['id'] = sample['id']
                    trial_result['chain_id'] = j
                    
                    correct_choice_letter = chr(ord('A') + sample['answer_key'])
                    trial_result['correct_choice'] = correct_choice_letter
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == correct_choice_letter)
                    
                    if 'track' in sample: trial_result['track'] = sample['track']
                    if 'source' in sample: trial_result['source'] = sample['source']
                    if 'hop_type' in sample: trial_result['hop_type'] = sample['hop_type']

                    # We explicitly order the keys to make the output file easy to read.
                    final_ordered_result = {
                        "id": trial_result['id'],
                        "chain_id": trial_result['chain_id'],
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": trial_result['correct_choice'],
                        "is_correct": trial_result['is_correct'],
                        "final_answer_raw": trial_result['final_answer_raw'],
                        "final_prompt_messages": trial_result['final_prompt_messages'],
                        "question": trial_result['question'],
                        "choices": trial_result['choices'],
                        "audio_path": trial_result['audio_path']
                    }

                    f.write(json.dumps(final_ordered_result, ensure_ascii=False) + "\n")
                    f.flush()

            except Exception as e:
                skipped_samples_count += 1
                logging.exception(f"SKIPPING SAMPLE due to unhandled error. ID: {sample.get('id', 'N/A')}")
                continue

    # The final summary provides a clear report of what was accomplished in this specific run.
    total_processed_in_this_run = len(data_samples) - len(completed_ids)
    logging.info(f"--- 'No-Reasoning' experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    logging.info("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    logging.info(f"Total samples in dataset: {len(data_samples)}")
    logging.info(f"Samples already complete: {len(completed_ids)}")
    logging.info(f"Samples processed in this run: {total_processed_in_this_run - skipped_samples_count}")
    logging.info(f"Skipped samples due to errors in this run: {skipped_samples_count}")
    logging.info(f"Results saved to: {output_path}")
    logging.info("="*65)