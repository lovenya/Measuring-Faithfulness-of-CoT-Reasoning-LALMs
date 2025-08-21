# experiments/baseline.py

import os
import json
import nltk
from core.lalm_utils import run_inference, parse_answer, sanitize_cot
from data_loader.data_loader import format_choices_for_prompt

# This is a 'foundational' experiment because it generates the primary data
# (the reasoning chains) that many other 'dependent' experiments rely on.
EXPERIMENT_TYPE = "foundational"

def run_baseline_trial(model, processor, question: str, choices: str, audio_path: str) -> dict:
    """
    Runs a full baseline trial for a single question.
    This involves two steps: generating a CoT, and then using that CoT to get a final answer.
    """
    # --- Turn 1: Generate the Chain-of-Thought ---
    # We prompt the model to think step-by-step to elicit its reasoning process.
    cot_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": "Let's think step by step:"}
    ]
    
    # We use sampling (do_sample=True) to allow the model to generate diverse reasoning chains.
    generated_cot = run_inference(
        model, processor, cot_prompt_messages, audio_path, max_new_tokens=768, do_sample=True
    )

    # --- Pre-computation Step: Sanitize the CoT ---
    # We remove the final "spoiler" sentence from the CoT. This sanitized version
    # is what we'll use in Turn 2 and what all dependent experiments will use.
    sanitized_cot = sanitize_cot(generated_cot)
        
    # --- Turn 2: Elicit the Final Answer ---
    # We present the model with the original context plus its own sanitized reasoning,
    # and then ask for a final, clean answer.
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": sanitized_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
    ]
    
    # We use deterministic inference (do_sample=False) here to get the single most likely answer for the given CoT.
    final_answer_text = run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=50, do_sample=False
    )
    
    parsed_choice = parse_answer(final_answer_text)

    # We return a comprehensive dictionary containing all artifacts of the trial.
    # Crucially, we save BOTH the original and the sanitized CoT for full transparency.
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
    Orchestrates the full baseline experiment, iterating through all samples and chains.
    """
    output_path = config.OUTPUT_PATH

    print(f"\n--- Running Baseline LALM Experiment ({config.CONDITION} condition): Saving to {output_path} ---")
    
    skipped_samples_count = 0
    with open(output_path, 'w') as f:
        for i, sample in enumerate(data_samples):
            try:
                # The main progress indicator for the run.
                if config.VERBOSE:
                    print(f"Processing sample {i+1}/{len(data_samples)}: {sample['id']}")
                
                choices_formatted = format_choices_for_prompt(sample['choices'])
                
                # We generate multiple reasoning chains for each question to measure stability.
                for j in range(config.NUM_CHAINS_PER_QUESTION):
                    if config.VERBOSE:
                        print(f"  - Generating chain {j+1}/{config.NUM_CHAINS_PER_QUESTION}...")
                    
                    trial_result = run_baseline_trial(
                        model, processor, 
                        sample['question'], 
                        choices_formatted,
                        sample['audio_path']
                    )
                    
                    # Add all the necessary metadata for downstream analysis.
                    trial_result['id'] = sample['id']
                    trial_result['chain_id'] = j
                    
                    correct_choice_letter = chr(ord('A') + sample['answer_key'])
                    trial_result['correct_choice'] = correct_choice_letter
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == correct_choice_letter)
                    
                    # Add optional metadata from the dataset if it exists.
                    if 'track' in sample: trial_result['track'] = sample['track']
                    if 'source' in sample: trial_result['source'] = sample['source']
                    if 'hop_type' in sample: trial_result['hop_type'] = sample['hop_type']
                    
                    # This ensures our output JSONL is human-readable, especially for non-English text.
                    f.write(json.dumps(trial_result, ensure_ascii=False) + "\n")

            except Exception as e:
                # Standard robust error handling.
                skipped_samples_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING SAMPLE DUE TO ERROR.")
                print(f"  - Sample ID: {sample.get('id', 'Not Available')}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    # Standard final summary report.
    print("\n--- Baseline LALM experiment complete. ---")
    print("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    print(f"Total samples in dataset: {len(data_samples)}")
    print(f"Successfully processed samples: {len(data_samples) - skipped_samples_count}")
    print(f"Skipped samples due to errors: {skipped_samples_count}")
    print(f"Results saved to: {output_path}")
    print("="*65)