# experiments/baseline.py

"""
This script conducts the "Baseline" experiment, which is the most fundamental
test in our framework. It serves two primary purposes:

1. To measure the model's peak performance when it is explicitly prompted to
   reason step-by-step before giving an answer.
2. To generate the high-quality reasoning chains (CoTs) that are the foundational
   data for almost all of our other, more complex 'dependent' experiments.
"""

import os
import json
import collections
import logging
from core.prompt_strategies import get_prompt_strategy, run_reasoning_trial

# This is a 'foundational' experiment.
EXPERIMENT_TYPE = "foundational"

def run_baseline_trial(
    model,
    processor,
    tokenizer,
    model_utils,
    question: str,
    choices: str,
    audio_path: str,
    prompt_strategy: str,
) -> dict:
    """
    Runs a full, two-turn baseline trial for a single question.
    This function first elicits a reasoning chain, then uses that chain to get a final answer.
    """
    prompt_outputs = run_reasoning_trial(
        model=model,
        processor=processor,
        model_utils=model_utils,
        question=question,
        choices=choices,
        audio_path=audio_path,
        strategy=prompt_strategy,
    )
    
    parsed_choice = model_utils.parse_answer(prompt_outputs["final_answer_raw"])

    # We return a comprehensive dictionary containing all artifacts of the trial.
    # Saving both the original and sanitized CoT is important for transparency.
    return {
        "question": question,
        "choices": choices,
        "audio_path": audio_path,
        "generated_cot": prompt_outputs["generated_cot"],
        "sanitized_cot": prompt_outputs["sanitized_cot"],
        "final_answer_raw": prompt_outputs["final_answer_raw"],
        "predicted_choice": parsed_choice,
        "final_prompt_messages": prompt_outputs["final_prompt_messages"],
    }


def run(model, processor, tokenizer, model_utils, data_samples, config):
    """
    Orchestrates the full baseline experiment with a robust, restartable design.
    """
    output_path = config.OUTPUT_PATH
    prompt_strategy = get_prompt_strategy(config)
    logging.info(f"--- Running Baseline Experiment (Model: {config.MODEL_ALIAS.upper()}): Saving to {output_path} ---")
    logging.info(f"Prompt strategy: {prompt_strategy}")

    # --- RESTARTABILITY LOGIC ---
    # This block makes our long-running experiments resilient to interruptions.
    # It checks for an existing results file and determines what work is already done.
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
    
    # A question is considered "fully complete" only if it has all of its required chains.
    fully_completed_ids = {
        q_id for q_id, count in completed_chains.items() 
        if count >= config.NUM_CHAINS_PER_QUESTION
    }
    
    if fully_completed_ids:
        logging.info(f"Found {len(fully_completed_ids)} fully completed questions. They will be skipped.")
    # --- END OF RESTARTABILITY LOGIC ---

    skipped_samples_count = 0
    # We open the file in 'append' mode ('a') to ensure we add to the existing file.
    with open(output_path, 'a') as f:
        for i, sample in enumerate(data_samples):
            try:
                # If this question is already done, we skip it instantly.
                if sample['id'] in fully_completed_ids:
                    continue

                if config.VERBOSE:
                    logging.info(f"Processing sample {i+1}/{len(data_samples)}: {sample['id']}")
                
                choices_formatted = model_utils.format_choices_for_prompt(sample['choices'])
                
                # We calculate exactly how many chains are still missing for this question.
                chains_to_generate = config.NUM_CHAINS_PER_QUESTION - completed_chains[sample['id']]
                
                for j in range(chains_to_generate):
                    current_chain_num = completed_chains[sample['id']] + j + 1
                    if config.VERBOSE:
                        logging.info(f"  - Generating chain {current_chain_num}/{config.NUM_CHAINS_PER_QUESTION}...")
                    
                    trial_result = run_baseline_trial(
                        model, processor, tokenizer, model_utils,
                        sample['question'], 
                        choices_formatted,
                        sample['audio_path'],
                        prompt_strategy,
                    )
                    
                    # Add all the necessary metadata for downstream analysis.
                    trial_result['id'] = sample['id']
                    trial_result['chain_id'] = current_chain_num - 1
                    
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
                        "audio_path": trial_result['audio_path'],
                        "generated_cot": trial_result['generated_cot'],
                        "sanitized_cot": trial_result['sanitized_cot']
                    }
                    
                    f.write(json.dumps(final_ordered_result, ensure_ascii=False) + "\n")
                    f.flush()                  
                    
            except Exception as e:
                skipped_samples_count += 1
                logging.exception(f"SKIPPING SAMPLE due to unhandled error. ID: {sample.get('id', 'N/A')}")
                continue

    # The final summary provides a clear report of what was accomplished in this specific run.
    total_processed_in_this_run = len(data_samples) - len(fully_completed_ids)
    logging.info(f"--- Baseline Experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    logging.info("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    logging.info(f"Total samples in dataset: {len(data_samples)}")
    logging.info(f"Samples already complete: {len(fully_completed_ids)}")
    logging.info(f"Samples processed in this run: {total_processed_in_this_run - skipped_samples_count}")
    logging.info(f"Skipped samples due to errors in this run: {skipped_samples_count}")
    logging.info(f"Results saved to: {output_path}")
    logging.info("="*65)
