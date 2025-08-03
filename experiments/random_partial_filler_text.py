# experiments/random_partial_filler_text.py

import os
import json
import collections
import nltk
import random
from core.filler_text_utils import create_filler_for_text, create_word_level_masked_cot, run_filler_trial

EXPERIMENT_TYPE = "dependent"

# --- HPC-Safe NLTK Initialization ---
local_nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.append(local_nltk_data_path)
else:
    print(f"FATAL: NLTK data directory not found at '{local_nltk_data_path}'.")
    exit(1)


def run(model, processor, config):
    """
    Orchestrates the WORD-LEVEL RANDOM partial filler text experiment.
    """
    # 1. Load dependent data
    baseline_results_path = os.path.join(config.RESULTS_DIR, "baseline", f"baseline_{config.DATASET_NAME}.jsonl")
    if not os.path.exists(baseline_results_path):
        print(f"FATAL ERROR: Baseline results file not found at '{baseline_results_path}'")
        return

    print(f"Reading and grouping baseline data from '{baseline_results_path}'...")
    all_baseline_trials = [json.loads(line) for line in open(baseline_results_path, 'r')]
            
    if config.NUM_SAMPLES_TO_RUN > 0:
        trials_by_question = collections.defaultdict(list)
        for trial in all_baseline_trials:
            trials_by_question[trial['id']].append(trial)
        unique_question_ids = list(trials_by_question.keys())[:config.NUM_SAMPLES_TO_RUN]
        samples_to_process = [trial for q_id in unique_question_ids for trial in trials_by_question[q_id]]
    else:
        samples_to_process = all_baseline_trials

    # 2. Run the experiment
    print(f"\n--- Running WORD-LEVEL Partial Filler (Random) Experiment: Saving to {config.OUTPUT_PATH} ---")
    print(f"Processing {len(samples_to_process)} total trials.")
    
    skipped_trials_count = 0
    with open(config.OUTPUT_PATH, 'w') as f:
        for i, baseline_trial in enumerate(samples_to_process):
            try:
                q_id, chain_id = baseline_trial['id'], baseline_trial['chain_id']
                if config.VERBOSE:
                    print(f"Processing trial {i+1}/{len(samples_to_process)}: ID {q_id}, Chain {chain_id}")

                sanitized_cot = baseline_trial['sanitized_cot']
                
                for percentile in range(0, 101, 5):
                    
                    # --- THE NEW, SIMPLIFIED LOGIC ---
                    # Call the centralized utility to perform random word-level masking.
                    modified_cot = create_word_level_masked_cot(sanitized_cot, percentile, mode='random')
                    
                    trial_result = run_filler_trial(
                        model, processor, 
                        baseline_trial['question'], 
                        baseline_trial['choices'], 
                        baseline_trial['audio_path'], 
                        modified_cot
                    )

                    baseline_final_choice = baseline_trial['predicted_choice']

                    # Add metadata and save according to our SOP
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
                    f.write(json.dumps(final_ordered_result) + "\n")

            except Exception as e:
                skipped_trials_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING TRIAL DUE TO ERROR.")
                print(f"  - Question ID: {baseline_trial.get('id', 'N/A')}, Chain ID: {baseline_trial.get('chain_id', 'N/A')}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    # Final summary
    print("\n--- WORD-LEVEL Partial Filler (Random) experiment complete. ---")
    print(f"Total trials processed: {len(samples_to_process)}")
    print(f"Skipped trials due to errors: {skipped_trials_count}")
    print(f"Results saved to: {config.OUTPUT_PATH}")