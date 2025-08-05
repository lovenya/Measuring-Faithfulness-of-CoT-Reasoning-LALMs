# experiments/robustness_to_noise.py

import os
import json
import collections
from core.lalm_utils import run_inference, parse_answer
from data_loader.data_loader import load_dataset

# This is a 'dependent' experiment. Its core purpose is to test the reasoning chains
# we generated in the 'baseline' experiment, making it dependent on that output.
EXPERIMENT_TYPE = "dependent"

def run_noise_trial(model, processor, question: str, choices: str, audio_path: str, sanitized_cot: str) -> dict:
    """
    This is the core workhorse function for a single inference.
    It takes a specific noisy audio file and a pre-generated CoT and gets the model's answer.
    """
    # The prompt is structured to be identical to the second turn of our baseline experiment.
    # This is a critical control. It ensures that the *only* variables being changed
    # are the audio quality and the SNR level, allowing for a fair, scientific comparison.
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": sanitized_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
    ]

    # We use deterministic inference (do_sample=False) because we want the single most
    # likely answer for this specific combination of CoT and noisy audio.
    final_answer_text = run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=10, do_sample=False
    )
    
    parsed_choice = parse_answer(final_answer_text)

    # The function returns a self-documenting dictionary, adhering to our SOP.
    return {
        "predicted_choice": parsed_choice,
        "final_answer_raw": final_answer_text,
        "final_prompt_messages": final_answer_prompt_messages
    }

def run(model, processor, config):
    """
    This is the main orchestrator for the experiment. It re-evaluates each
    baseline CoT against multiple levels of audio noise to measure the impact on
    both accuracy and consistency.
    """
    # --- 1. Load All Necessary Data ---
    # We need two sources of data: the baseline results (for the CoTs) and the
    # noisy dataset (for the paths to the corrupted audio files).

    baseline_results_path = os.path.join(config.RESULTS_DIR, "baseline", f"baseline_{config.DATASET_NAME}.jsonl")
    if not os.path.exists(baseline_results_path):
        print(f"FATAL ERROR: Baseline results file not found at '{baseline_results_path}'")
        return

    noisy_dataset_name = f"{config.DATASET_NAME}-noisy"
    noisy_jsonl_path = config.DATASET_MAPPING.get(noisy_dataset_name)
    if not noisy_jsonl_path or not os.path.exists(noisy_jsonl_path):
        print(f"FATAL ERROR: Noisy dataset file not found for '{config.DATASET_NAME}'. Expected at '{noisy_jsonl_path}'")
        return

    print(f"Reading baseline CoTs from: {baseline_results_path}")
    all_baseline_trials = [json.loads(line) for line in open(baseline_results_path, 'r')]
    
    print(f"Reading noisy audio data from: {noisy_jsonl_path}")
    noisy_data = [json.loads(line) for line in open(noisy_jsonl_path, 'r')]

    # --- 2. Prepare Data for Efficient Processing ---
    # To avoid searching through the noisy data file repeatedly, we create a lookup dictionary.
    # This lets us instantly find the path for a specific audio file at a specific SNR level.
    noisy_audio_lookup = {
        (d['original_audio_path'].split('/')[-1].split('.')[0], d['snr_db']): d['audio_path']
        for d in noisy_data
    }

    # This block handles the --num-samples flag, allowing for quick test runs.
    # It correctly filters to the first N *unique questions* from the baseline data.
    if config.NUM_SAMPLES_TO_RUN > 0:
        trials_by_question = collections.defaultdict(list)
        for trial in all_baseline_trials:
            trials_by_question[trial['id']].append(trial)
        
        unique_question_ids = list(trials_by_question.keys())[:config.NUM_SAMPLES_TO_RUN]
        samples_to_process = [trial for q_id in unique_question_ids for trial in trials_by_question[q_id]]
    else:
        samples_to_process = all_baseline_trials

    # --- 3. Run the Main Experiment Loop ---
    output_path = config.OUTPUT_PATH
    print(f"\n--- Running 'Robustness to Noise' Experiment: Saving to {output_path} ---")
    print(f"Processing {len(samples_to_process)} baseline chains against {len(config.SNR_LEVELS_TO_TEST)} noise levels each.")
    
    skipped_trials_count = 0
    with open(output_path, 'w') as f:
        # The main loop iterates through each unique reasoning chain from our baseline results.
        for i, baseline_trial in enumerate(samples_to_process):
            try:
                if config.VERBOSE:
                    print(f"Processing baseline trial {i+1}/{len(samples_to_process)}: ID {baseline_trial['id']}, Chain {baseline_trial['chain_id']}")

                original_audio_key = baseline_trial['audio_path'].split('/')[-1].split('.')[0]

                # The inner loop tests the current reasoning chain against each specified SNR level.
                for snr_level in config.SNR_LEVELS_TO_TEST:
                    lookup_key = (original_audio_key, snr_level)
                    if lookup_key not in noisy_audio_lookup:
                        if config.VERBOSE:
                            print(f"  - WARNING: Noisy audio not found for key {lookup_key}. Skipping.")
                        continue

                    noisy_audio_path = noisy_audio_lookup[lookup_key]
                    
                    # Run the core inference logic for this specific CoT + noisy audio pair.
                    trial_result = run_noise_trial(
                        model, processor,
                        baseline_trial['question'],
                        baseline_trial['choices'],
                        noisy_audio_path,
                        baseline_trial['sanitized_cot']
                    )


                    # We retrieve the original answer from the baseline trial. This is our reference point.
                    baseline_final_choice = baseline_trial['predicted_choice']
                    
                    # Consistency check included
                    final_ordered_result = {
                        "id": baseline_trial['id'],
                        "chain_id": baseline_trial['chain_id'],
                        "snr_db": snr_level,
                        "noisy_audio_path_used": noisy_audio_path,
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": baseline_trial['correct_choice'],
                        "is_correct": (trial_result['predicted_choice'] == baseline_trial['correct_choice']),
                        "corresponding_baseline_predicted_choice": baseline_final_choice,
                        "is_consistent_with_baseline": (trial_result['predicted_choice'] == baseline_final_choice),
                        "final_prompt_messages": trial_result['final_prompt_messages'],
                        "final_answer_raw": trial_result['final_answer_raw'],
                    }

                    f.write(json.dumps(final_ordered_result) + "\n")

            except Exception as e:
                # Our standard, robust error handling block to prevent a single bad
                # data point from crashing a long-running job.
                skipped_trials_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING TRIAL DUE TO ERROR.")
                print(f"  - Question ID: {baseline_trial.get('id', 'N/A')}, Chain ID: {baseline_trial.get('chain_id', 'N/A')}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    # --- 4. Final Summary Report ---
    print("\n--- 'Robustness to Noise' experiment complete. ---")
    print("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    print(f"Total baseline chains processed: {len(samples_to_process)}")
    print(f"Skipped baseline chains due to errors: {skipped_trials_count}")
    print(f"Results saved to: {output_path}")
    print("="*65)