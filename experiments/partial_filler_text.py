# experiments/partial_filler_text.py

import os
import json
import collections
from core.filler_text_utils import create_word_level_masked_cot, run_filler_trial

# This is a 'dependent' experiment because it manipulates the CoTs from a 'baseline' run.
EXPERIMENT_TYPE = "dependent"

def run(model, processor, config):
    """
    Orchestrates the WORD-LEVEL partial filler text experiment (corruption from start).
    This version is now fully condition-aware and adheres to our SOP.
    """
    output_path = config.OUTPUT_PATH

    # --- 1. Condition-Aware Path Construction ---
    # This block correctly constructs the path to the necessary baseline results file,
    # accounting for both the condition-specific directory and the condition-specific filename.
    if config.CONDITION == 'default':
        baseline_results_dir = os.path.join(config.RESULTS_DIR, 'baseline')
        baseline_filename = f"baseline_{config.DATASET_NAME}.jsonl"
    else:
        condition_dir = f"{config.CONDITION}_experiments"
        baseline_results_dir = os.path.join(config.RESULTS_DIR, condition_dir, 'baseline')
        baseline_filename = f"baseline_{config.DATASET_NAME}_{config.CONDITION}.jsonl"
    
    baseline_results_path = os.path.join(baseline_results_dir, baseline_filename)

    # --- 2. Load Data ---
    try:
        print(f"Reading baseline data for condition '{config.CONDITION}' from '{baseline_results_path}'...")
        all_baseline_trials = [json.loads(line) for line in open(baseline_results_path, 'r')]
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required foundational results file was not found. Details: {e}")
        return
            
    # Standard logic to handle the --num-samples flag for quick test runs.
    if config.NUM_SAMPLES_TO_RUN > 0:
        trials_by_question = collections.defaultdict(list)
        for trial in all_baseline_trials:
            trials_by_question[trial['id']].append(trial)
        unique_question_ids = list(trials_by_question.keys())[:config.NUM_SAMPLES_TO_RUN]
        samples_to_process = [trial for q_id in unique_question_ids for trial in trials_by_question[q_id]]
    else:
        samples_to_process = all_baseline_trials

    # --- 3. Run the Experiment ---
    print(f"\n--- Running WORD-LEVEL Partial Filler (Start) Experiment ({config.CONDITION} condition): Saving to {output_path} ---")
    print(f"Processing {len(samples_to_process)} total trials.")
    
    skipped_trials_count = 0
    with open(output_path, 'w') as f:
        for i, baseline_trial in enumerate(samples_to_process):
            try:
                q_id, chain_id = baseline_trial['id'], baseline_trial['chain_id']
                if config.VERBOSE:
                    print(f"Processing trial {i+1}/{len(samples_to_process)}: ID {q_id}, Chain {chain_id}")

                sanitized_cot = baseline_trial['sanitized_cot']
                
                for percentile in range(0, 101, 5):
                    
                    # Call the centralized utility to perform word-level masking from the start.
                    modified_cot = create_word_level_masked_cot(sanitized_cot, percentile, mode='start')
                    
                    trial_result = run_filler_trial(
                        model, processor, 
                        baseline_trial['question'], 
                        baseline_trial['choices'], 
                        baseline_trial['audio_path'], 
                        modified_cot
                    )

                    baseline_final_choice = baseline_trial['predicted_choice']
                    
                    # Add metadata and save according to our SOP.
                    final_ordered_result = {
                        "id": q_id,
                        "chain_id": chain_id,
                        "percent_replaced": percentile,
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": baseline_trial['correct_choice'],
                        "is_correct": (trial_result['predicted_choice'] == baseline_trial['correct_choice']),
                        "corresponding_baseline_predicted_choice": baseline_final_choice,
                        "is_consistent_with_baseline": (trial_result['predicted_choice'] == baseline_final_choice),
                        "final_prompt_messages": trial_result['final_prompt_messages'],
                        "final_answer_raw": trial_result['final_answer_raw']
                    }
                    # Ensure human-readable output.
                    f.write(json.dumps(final_ordered_result, ensure_ascii=False) + "\n")

            except Exception as e:
                skipped_trials_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING TRIAL DUE TO ERROR.")
                print(f"  - Question ID: {baseline_trial.get('id', 'N/A')}, Chain ID: {baseline_trial.get('chain_id', 'N/A')}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    # --- Final Summary ---
    print("\n--- WORD-LEVEL Partial Filler (Start) experiment complete. ---")
    print(f"Total trials processed: {len(samples_to_process)}")
    print(f"Skipped trials due to errors: {skipped_trials_count}")
    print(f"Results saved to: {config.OUTPUT_PATH}")