import os
import json
import argparse
import re
from pathlib import Path

def sanitize_cot(reasoning: str) -> str:
    # Remove XML tags like <Reasoning> and </Reasoning>
    reasoning = re.sub(r'<Reasoning>', '', reasoning, flags=re.IGNORECASE)
    reasoning = re.sub(r'</Reasoning>', '', reasoning, flags=re.IGNORECASE)
    return reasoning.strip()

def process_baselines():
    base_dir = Path("/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs")
    
    models_to_map = {
        'af3': 'flamingo_hf',
        'qwen2.5': 'qwen_omni'
    }
    tracks = ['animal', 'emotion', 'gender', 'language']
    
    for source_model, target_model in models_to_map.items():
        for track in tracks:
            if source_model == 'af3':
                pooneh_file = base_dir / f"pooneh_version/result/baseline/{source_model}/{track}/baseline_{track}_REAS_post_process.jsonl"
            else:
                pooneh_file = base_dir / f"pooneh_version/result/baseline/{source_model}/{track}/baseline_{track}_REAS.jsonl"
                
            dataset_file = base_dir / f"data/sakura/{track}/sakura_{track}_test_standardized.jsonl"
            
            out_dir = base_dir / f"results/{target_model}/baseline"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"baseline_{target_model}_sakura-{track}.jsonl"

            if not pooneh_file.exists():
                print(f"Skipping {pooneh_file} - not found.")
                continue
            
            if not dataset_file.exists():
                print(f"Skipping mapping for {track} - dataset file {dataset_file} not found.")
                continue

            # Load dataset mapping
            dataset_mapping = {}
            with open(dataset_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        dataset_mapping[data['id']] = data
                    except json.JSONDecodeError:
                        pass

            print(f"Processing {track} for {source_model} -> {target_model}...")
            processed_count = 0
            missing_count = 0
            
            with open(pooneh_file, 'r') as f_in, open(out_file, 'w') as f_out:
                for line in f_in:
                    try:
                        poo = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                        
                    q_id = poo['id']
                    if q_id not in dataset_mapping:
                        print(f"  Warning: {q_id} not found in dataset mapping!")
                        missing_count += 1
                        continue
                        
                    ds = dataset_mapping[q_id]

                    formatted_choices = []
                    for idx, c in enumerate(ds['choices']):
                        letter = chr(ord('A') + idx)
                        formatted_choices.append(f"({letter}) {c}")
                    choices_formatted_str = "\n".join(formatted_choices)

                    # Extract the reasoning. Note Pooneh files might have list or string.
                    raw_reasoning_list = poo.get('reasoning', [])
                    reasoning = raw_reasoning_list[0] if raw_reasoning_list else ""
                    
                    sanitized = sanitize_cot(reasoning)

                    raw_outputs_list = poo.get('raw_model_outputs', [])
                    final_answer_raw = raw_outputs_list[0] if raw_outputs_list else ""

                    pred_letters = poo.get('predicted_letters', [])
                    predicted_choice = pred_letters[0] if pred_letters else ""
                    
                    correct_choice = poo.get('true_letter', "")
                    is_correct = bool(poo.get('accuracy', 0.0) == 1.0)

                    rec = {
                        "id": q_id,
                        "chain_id": 0,
                        "predicted_choice": predicted_choice,
                        "correct_choice": correct_choice,
                        "is_correct": is_correct,
                        "final_answer_raw": final_answer_raw,
                        "final_prompt_messages": [],
                        "question": ds['question'],
                        "choices": choices_formatted_str,
                        "audio_path": ds['audio_path'],
                        "generated_cot": reasoning,
                        "sanitized_cot": sanitized
                    }

                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    processed_count += 1
            
            print(f"  Successfully processed {processed_count} records. Missed {missing_count} due to missing dataset info.")

if __name__ == "__main__":
    process_baselines()
