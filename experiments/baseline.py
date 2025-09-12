# experiments/baseline.py

import os
import json
import collections

# This is a 'foundational' experiment. Its primary job is to generate the
# core reasoning chains (CoTs) that many of our other 'dependent' experiments
# will use as their starting point.
EXPERIMENT_TYPE = "foundational"

def run_baseline_trial(model, processor, model_utils, question: str, choices: str, audio_path: str) -> dict:
    """
    Runs a full, two-turn baseline trial for a single question.
    This function first elicits a reasoning chain, then uses that chain to get a final answer.
    """
    # --- Turn 1: Generate the Chain-of-Thought ---
    # We prompt the model with the core question and ask it to "think step by step."
    # This is designed to produce the model's natural, unstructured reasoning process.
    cot_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": "Let's think step by step:"}
    ]
    
    # We use sampling (do_sample=True) here because we want to see the diversity
    # in the model's reasoning. This allows it to generate different logical paths
    # across the multiple chains we run for each question.
    generated_cot = model_utils.run_inference(
        model, processor, cot_prompt_messages, audio_path, 
        max_new_tokens=768, do_sample=True, temperature=0.7, top_p=0.9   
    )

    # --- Pre-computation Step: Sanitize the CoT ---
    # We clean the generated reasoning by removing the final sentence. This is a crucial
    # step to prevent the model from "cheating" in Turn 2 by simply finding a
    # "spoiler" sentence like "Therefore, the answer is (C)."
    sanitized_cot = model_utils.sanitize_cot(generated_cot)
        
    # --- Turn 2: Elicit the Final Answer ---
    # We now present the model with its own cleaned-up reasoning and ask it to
    # make a final, definitive choice in a structured format.
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": sanitized_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
    ]
    
    # For this turn, we use deterministic generation (do_sample=False). We want to know
    # the single most likely answer given this specific reasoning chain.
    final_answer_text = model_utils.run_inference(
        model, processor, final_answer_prompt_messages, audio_path, 
        max_new_tokens=50, do_sample=False, temperature=0.7, top_p=0.9
    )
    
    parsed_choice = model_utils.parse_answer(final_answer_text)

    # We return a comprehensive dictionary containing all artifacts of the trial.
    # Saving both the original and sanitized CoT is important for transparency.
    return {
        "question": question,
        "choices": choices,
        "audio_path": audio_path,
        "generated_cot": generated_cot,
        "sanitized_cot": sanitized_cot,
        "final_answer_raw": final_answer_text,
        "predicted_choice": parsed_choice,
    }


def run(model, processor, model_utils, data_samples, config):
    """
    Orchestrates the full baseline experiment with a robust, restartable design.
    """
    output_path = config.OUTPUT_PATH
    print(f"\n--- Running Baseline Experiment (Model: {config.MODEL_ALIAS.upper()}): Saving to {output_path} ---")

    # --- RESTARTABILITY LOGIC ---
    # This block makes our long-running experiments resilient to interruptions.
    # It checks for an existing results file and determines what work is already done.
    completed_chains = collections.defaultdict(int)
    if os.path.exists(output_path):
        print("  - Found existing results file. Checking for completed work...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    # We count how many chains have been successfully generated for each question ID.
                    data = json.loads(line)
                    completed_chains[data['id']] += 1
                except json.JSONDecodeError:
                    # This handles cases where a job was killed mid-write, leaving a corrupted line.
                    print(f"  - WARNING: Found a corrupted line in {output_path}. It will be ignored.")
                    continue
    
    # A question is considered "fully complete" only if it has all of its required chains.
    fully_completed_ids = {
        q_id for q_id, count in completed_chains.items() 
        if count >= config.NUM_CHAINS_PER_QUESTION
    }
    
    if fully_completed_ids:
        print(f"  - Found {len(fully_completed_ids)} fully completed questions. They will be skipped.")
    # --- END OF RESTARTABILITY LOGIC ---

    skipped_samples_count = 0
    # We open the file in 'append' mode ('a'). This is key to the restartable design,
    # as it ensures we add to the existing file instead of overwriting it.
    with open(output_path, 'a') as f:
        for i, sample in enumerate(data_samples):
            try:
                # If this question is already done, we skip it instantly.
                if sample['id'] in fully_completed_ids:
                    continue

                if config.VERBOSE:
                    print(f"Processing sample {i+1}/{len(data_samples)}: {sample['id']}")
                
                choices_formatted = model_utils.format_choices_for_prompt(sample['choices'])
                
                # This is the core of the partial-completion logic. We calculate exactly
                # how many chains are still missing for this question.
                chains_to_generate = config.NUM_CHAINS_PER_QUESTION - completed_chains[sample['id']]
                
                for j in range(chains_to_generate):
                    # We calculate the true chain number for accurate logging.
                    current_chain_num = completed_chains[sample['id']] + j + 1
                    if config.VERBOSE:
                        print(f"  - Generating chain {current_chain_num}/{config.NUM_CHAINS_PER_QUESTION}...")
                    
                    trial_result = run_baseline_trial(
                        model, processor, model_utils,
                        sample['question'], 
                        choices_formatted,
                        sample['audio_path']
                    )
                    
                    # Add all the necessary metadata for downstream analysis.
                    trial_result['id'] = sample['id']
                    trial_result['chain_id'] = current_chain_num - 1 # Use 0-based indexing for the ID
                    
                    correct_choice_letter = chr(ord('A') + sample['answer_key'])
                    trial_result['correct_choice'] = correct_choice_letter
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == correct_choice_letter)
                    
                    if 'track' in sample: trial_result['track'] = sample['track']
                    if 'source' in sample: trial_result['source'] = sample['source']
                    if 'hop_type' in sample: trial_result['hop_type'] = sample['hop_type']
                    
                    # We ensure our output JSONL is human-readable, especially for non-English text.
                    f.write(json.dumps(trial_result, ensure_ascii=False) + "\n")
                    
                    f.flush()                  
                    
            except Exception as e:
                # Our standard error handling block to ensure one bad sample doesn't kill the whole job.
                skipped_samples_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING SAMPLE DUE TO ERROR.")
                print(f"  - Sample ID: {sample.get('id', 'Not Available')}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    # The final summary provides a clear report of what was accomplished in this specific run.
    total_processed_in_this_run = len(data_samples) - len(fully_completed_ids)
    print(f"\n--- Baseline Experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    print("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    print(f"Total samples in dataset: {len(data_samples)}")
    print(f"Samples already complete: {len(fully_completed_ids)}")
    print(f"Samples processed in this run: {total_processed_in_this_run - skipped_samples_count}")
    print(f"Skipped samples due to errors in this run: {skipped_samples_count}")
    print(f"Results saved to: {output_path}")
    print("="*65)