# experiments/no_reasoning.py

import os
import json

# This is a 'foundational' experiment. It provides a crucial baseline for
# the model's performance when given the prompt structure but no reasoning content.
EXPERIMENT_TYPE = "foundational"

def run_no_reasoning_trial(model, processor, model_utils, question: str, choices: str, audio_path: str) -> dict:
    """
    Runs a single, deterministic trial with an empty CoT.
    """
    # This prompt structure perfectly matches the final prompt of the baseline
    # experiment, but with an empty string for the assistant's CoT. This isolates
    # the effect of the prompt format itself.
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": ""}, # The reasoning is an empty string
        {"role": "user", "content": "What is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else."}
    ]

    # This is a deterministic call (do_sample=False) since there's no creative generation.
    final_answer_text = model_utils.run_inference(
        model, processor, final_answer_prompt_messages, audio_path, 
        max_new_tokens=10, do_sample=False, temperature=0.7, top_p=0.9
    )
    
    parsed_choice = model_utils.parse_answer(final_answer_text)

    # Return a self-documenting dictionary, including the prompt, per our SOP.
    return {
        "question": question,
        "choices": choices,
        "audio_path": audio_path,
        "final_answer_raw": final_answer_text,
        "predicted_choice": parsed_choice,
        "final_prompt_messages": final_answer_prompt_messages
    }

def run(model, processor, model_utils, data_samples, config):
    """
    Orchestrates the full "No-Reasoning" experiment with a robust, restartable design.
    """
    output_path = config.OUTPUT_PATH
    print(f"\n--- Running 'No-Reasoning' Experiment (Model: {config.MODEL_ALIAS.upper()}): Saving to {output_path} ---")

    # --- RESTARTABILITY LOGIC (SIMPLIFIED FOR DETERMINISTIC EXPERIMENT) ---
    completed_ids = set()
    if os.path.exists(output_path):
        print("  - Found existing results file. Checking for completed work...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    # For this deterministic experiment, we only need to know if an ID
                    # exists in the file. If it does, we consider it complete.
                    completed_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    continue # Ignore corrupted lines or lines missing an 'id'
    
    if completed_ids:
        print(f"  - Found {len(completed_ids)} completed questions. They will be skipped.")
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
                    print(f"Processing sample {i+1}/{len(data_samples)}: {sample['id']}")
                
                choices_formatted = model_utils.format_choices_for_prompt(sample['choices'])
                
                # --- OPTIMIZATION: RUN INFERENCE ONLY ONCE ---
                # Since this is a deterministic experiment, the result will be the same
                # for all chains. We run the expensive inference call a single time...
                cached_result = run_no_reasoning_trial(
                    model, processor, model_utils,
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

                    f.write(json.dumps(trial_result, ensure_ascii=False) + "\n")
                    
                    f.flush()

            except Exception as e:
                skipped_samples_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING SAMPLE DUE TO ERROR.")
                print(f"  - Sample ID: {sample.get('id', 'Not Available')}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    # The final summary provides a clear report of what was accomplished in this specific run.
    total_processed_in_this_run = len(data_samples) - len(completed_ids)
    print(f"\n--- 'No-Reasoning' experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    print("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    print(f"Total samples in dataset: {len(data_samples)}")
    print(f"Samples already complete: {len(completed_ids)}")
    print(f"Samples processed in this run: {total_processed_in_this_run - skipped_samples_count}")
    print(f"Skipped samples due to errors in this run: {skipped_samples_count}")
    print(f"Results saved to: {output_path}")
    print("="*65)