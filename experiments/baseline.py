# experiments/baseline.py

import os
import json
from core.lalm_utils import run_inference, parse_answer, sanitize_cot
from data_loader.data_loader import format_choices_for_prompt


EXPERIMENT_TYPE = "foundational"


def run_baseline_trial(model, processor, question: str, choices: str, audio_path: str) -> dict:
    """
    Runs a full baseline trial for a single question with audio input.
    """
    # Turn 1: Generate CoT
    cot_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": "Let's think step by step:"}
    ]
    generated_cot = run_inference(
        model, processor, cot_prompt_messages, audio_path, max_new_tokens=768, do_sample=True
    )

    sanitized_cot = sanitize_cot(generated_cot)
        
    # Turn 2: Elicit Final Answer
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": sanitized_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
    ]
    final_answer_text = run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=50, do_sample=False
    )
    
    parsed_choice = parse_answer(final_answer_text)

    return {
        "question": question,
        "choices": choices,
        "audio_path": audio_path,
        "generated_cot": generated_cot,
        "sanitized_cot": sanitized_cot,
        "final_answer_raw": final_answer_text,
        "predicted_choice": parsed_choice,
    }


def run(model, processor, data_samples, config):
    """
    Runs the full baseline experiment on a list of data samples, now with robust error handling.
    """
    output_path = config.OUTPUT_PATH

    print(f"\n--- Running Baseline LALM Experiment: Saving to {output_path} ---")
    
    # --- ROBUSTNESS ENHANCEMENT ---
    # Counter for skipped samples
    skipped_samples_count = 0

    with open(output_path, 'w') as f:
        for i, sample in enumerate(data_samples):
            try:
                # The entire processing for one sample is wrapped in a try-except block.
                print(f"Processing sample {i+1}/{len(data_samples)}: {sample['id']}")
                
                choices_formatted = format_choices_for_prompt(sample['choices'])
                
                for j in range(config.NUM_CHAINS_PER_QUESTION):
                    print(f"  - Generating chain {j+1}/{config.NUM_CHAINS_PER_QUESTION}...")
                    trial_result = run_baseline_trial(
                        model, processor, 
                        sample['question'], 
                        choices_formatted,
                        sample['audio_path']
                    )
                    
                    trial_result['id'] = sample['id']
                    trial_result['chain_id'] = j
                    
                    # This is the line that caused the original error.
                    correct_choice_letter = chr(ord('A') + sample['answer_key'])
                    trial_result['correct_choice'] = correct_choice_letter
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == correct_choice_letter)
                    
                    if 'track' in sample:
                        trial_result['track'] = sample['track']
                    if 'source' in sample:
                        trial_result['source'] = sample['source']
                    if 'hop_type' in sample:
                        trial_result['hop_type'] = sample['hop_type']
                    
                    f.write(json.dumps(trial_result) + "\n")

            except Exception as e:
                # If any error occurs for a sample, log it and continue.
                skipped_samples_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING SAMPLE DUE TO ERROR.")
                # Use .get() for safety in case the 'id' field itself is missing.
                print(f"  - Sample ID: {sample.get('id', 'Not Available')}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue # Move to the next sample in the dataset

    # --- END OF ROBUSTNESS ENHANCEMENT ---

    print("\n--- Baseline LALM experiment complete. ---")
    
    # Final summary report
    print("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    print(f"Total samples in dataset: {len(data_samples)}")
    print(f"Successfully processed samples: {len(data_samples) - skipped_samples_count}")
    print(f"Skipped samples due to errors: {skipped_samples_count}")
    print(f"Results saved to: {output_path}")
    print("="*65)