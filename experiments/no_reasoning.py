# experiments/no_reasoning.py

import os
import json
from core.lalm_utils import run_inference, parse_answer
from data_loader.data_loader import format_choices_for_prompt

# This is a 'foundational' experiment. It provides a crucial baseline for
# the model's performance when given the prompt structure but no reasoning content.
EXPERIMENT_TYPE = "foundational"

def run_no_reasoning_trial(model, processor, question: str, choices: str, audio_path: str) -> dict:
    """
    Runs a single trial with an empty CoT. This is our most constrained baseline.
    """
    # This prompt structure perfectly matches the final prompt of the baseline
    # experiment, but with an empty string for the assistant's CoT. This isolates
    # the effect of the prompt format itself.
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": ""}, # The reasoning is an empty string
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
    ]

    # This is a deterministic call (do_sample=False) since there's no creative generation.
    final_answer_text = run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=10, do_sample=False
    )
    
    parsed_choice = parse_answer(final_answer_text)

    # Return a self-documenting dictionary, including the prompt, per our SOP.
    return {
        "question": question,
        "choices": choices,
        "audio_path": audio_path,
        "final_answer_raw": final_answer_text,
        "predicted_choice": parsed_choice,
        "final_prompt_messages": final_answer_prompt_messages
    }

def run(model, processor, data_samples, config):
    """
    Orchestrates the full "No-Reasoning" experiment, adhering to our SOP.
    """
    output_path = config.OUTPUT_PATH

    print(f"\n--- Running 'No-Reasoning' Experiment ({config.CONDITION} condition): Saving to {output_path} ---")
    
    skipped_samples_count = 0
    with open(output_path, 'w') as f:
        for i, sample in enumerate(data_samples):
            try:
                if config.VERBOSE:
                    print(f"Processing sample {i+1}/{len(data_samples)}: {sample['id']}")
                
                choices_formatted = format_choices_for_prompt(sample['choices'])
                
                # This is a deterministic experiment, so we only need to run it once per question.
                # We iterate through the number of chains to create a result for each one,
                # ensuring our data structure is consistent with the baseline results.
                for j in range(config.NUM_CHAINS_PER_QUESTION):
                    if j == 0:
                        # We only need to run the inference once.
                        trial_result = run_no_reasoning_trial(
                            model, processor, 
                            sample['question'], 
                            choices_formatted, 
                            sample['audio_path']
                        )
                        # Store the result to reuse for other chains.
                        cached_result = trial_result.copy()
                    else:
                        # For subsequent chains, reuse the cached result.
                        trial_result = cached_result.copy()

                    trial_result['id'] = sample['id']
                    # Add chain_id for consistency with other result files.
                    trial_result['chain_id'] = j
                    
                    correct_choice_letter = chr(ord('A') + sample['answer_key'])
                    trial_result['correct_choice'] = correct_choice_letter
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == correct_choice_letter)
                    
                    if 'track' in sample: trial_result['track'] = sample['track']
                    if 'source' in sample: trial_result['source'] = sample['source']
                    if 'hop_type' in sample: trial_result['hop_type'] = sample['hop_type']

                    # Ensure human-readable output.
                    f.write(json.dumps(trial_result, ensure_ascii=False) + "\n")

            except Exception as e:
                skipped_samples_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING SAMPLE DUE TO ERROR.")
                print(f"  - Sample ID: {sample.get('id', 'Not Available')}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    # Standard final summary report.
    print("\n--- 'No-Reasoning' experiment complete. ---")
    print("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    print(f"Total samples in dataset: {len(data_samples)}")
    print(f"Successfully processed samples: {len(data_samples) - skipped_samples_count}")
    print(f"Skipped samples due to errors: {skipped_samples_count}")
    print(f"Results saved to: {output_path}")
    print("="*65)