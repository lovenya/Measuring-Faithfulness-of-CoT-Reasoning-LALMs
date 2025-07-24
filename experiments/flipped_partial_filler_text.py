# experiments/flipped_partial_filler_text.py

import os
import json
import collections
import nltk
from core.filler_text_utils import create_filler_for_text, run_filler_trial

EXPERIMENT_TYPE = "dependent"

# --- HPC-Safe NLTK Initialization ---
local_nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.append(local_nltk_data_path)
else:
    print(f"FATAL: NLTK data directory not found at '{local_nltk_data_path}'.")
    exit(1)


def run(model, processor, config):
    """Orchestrates the FLIPPED partial filler text experiment, adhering to the SOP."""
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
    print(f"\n--- Running FLIPPED Partial Filler Text Experiment: Saving to {config.OUTPUT_PATH} ---")
    print(f"Processing {len(samples_to_process)} total trials.")
    
    skipped_trials_count = 0
    with open(config.OUTPUT_PATH, 'w') as f:
        for i, baseline_trial in enumerate(samples_to_process):
            try:
                q_id, chain_id = baseline_trial['id'], baseline_trial['chain_id']
                if config.VERBOSE:
                    print(f"Processing trial {i+1}/{len(samples_to_process)}: ID {q_id}, Chain {chain_id}")

                original_cot = baseline_trial['generated_cot']
                sentences = nltk.sent_tokenize(original_cot)
                total_sentences = len(sentences)
                if total_sentences == 0: continue

                for percentile in range(0, 101, 10):
                    num_to_replace = int((percentile / 100) * total_sentences)
                    
                    # --- FLIPPED LOGIC ---
                    if num_to_replace == 0:
                        part_to_replace = ""
                        remaining_part = original_cot
                    else:
                        # Replace the LAST num_to_replace sentences
                        part_to_replace = " ".join(sentences[-num_to_replace:])
                        remaining_part = " ".join(sentences[:-num_to_replace])
                    # --- END OF FLIPPED LOGIC ---
                    
                    filler_text = create_filler_for_text(processor, part_to_replace)
                    modified_cot = (remaining_part + " " + filler_text).strip()
                    
                    trial_result = run_filler_trial(
                        model, processor, baseline_trial['question'], baseline_trial['choices'], baseline_trial['audio_path'], modified_cot
                    )

                    trial_result['id'] = q_id
                    trial_result['chain_id'] = chain_id
                    trial_result['percent_replaced'] = percentile
                    trial_result['correct_choice'] = baseline_trial['correct_choice']
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == trial_result['correct_choice'])
                    
                    final_ordered_result = {
                        "id": trial_result['id'], "chain_id": trial_result['chain_id'],
                        "percent_replaced": trial_result['percent_replaced'],
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": trial_result['correct_choice'],
                        "is_correct": trial_result['is_correct'],
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
    print("\n--- FLIPPED Partial Filler Text experiment complete. ---")
    print(f"Total trials processed: {len(samples_to_process)}")
    print(f"Skipped trials due to errors: {skipped_trials_count}")
    print(f"Results saved to: {config.OUTPUT_PATH}")