import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import argparse
# ==========================================
# 1. LOAD MODEL GLOBALLY
# ==========================================
model_id = "nvidia/audio-flamingo-3-hf"
local_id = snapshot_download(model_id)

processor = AutoProcessor.from_pretrained(local_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(local_id, device_map="auto", 
                                                            #    attn_implementation="flash_attention_2" 
                                                               )

non_lora_path = os.path.join(local_id, "think", "non_lora_trainables.bin")
non_lora_trainables = torch.load(non_lora_path)
model.load_state_dict(non_lora_trainables, strict=False)

model = PeftModel.from_pretrained(model, local_id, subfolder="think")
model.eval()

# ==========================================
# 2. PARSING FUNCTIONS
# ==========================================
# ==========================================
# 2. PARSING FUNCTIONS
# ==========================================
def parse_model_output(raw_text: str) -> dict:
    cleaned_text = raw_text.strip()
    
    # Find all instances of A, B, C, or D that are standalone (handles (A), A., or " A ")
    matches = list(re.finditer(r'(?:^|[^a-zA-Z])([A-D])(?:[^a-zA-Z]|$)', cleaned_text, re.IGNORECASE))
    
    if matches:
        # Grab the very last match found in the text
        last_match = matches[-1]
        predicted_choice = last_match.group(1).upper()
        
        # Reasoning is everything before this final letter
        reasoning = cleaned_text[:last_match.start()].strip()
        
        # Clean up any trailing filler words the model might have used right before the letter
        reasoning = re.sub(r'(?i)(therefore|so|thus)?\s*,?\s*the (final )?(most likely )?(answer|prediction|match|choice) is\s*:?\s*$', '', reasoning).strip()
    else:
        predicted_choice = None
        reasoning = cleaned_text

    return {
        "raw_output": raw_text,
        "reasoning": reasoning,
        "predicted_choice": predicted_choice
    }

def extract_true_letter(answer_str: str) -> str:
    match = re.search(r'\(([a-zA-Z])\)', answer_str)
    if match:
        return match.group(1).upper()
    return None

# ==========================================
# 3. SINGLE-ITEM INFERENCE FUNCTION
# ==========================================
def run_audio_inference(instruction: str, raw_audio_path: str) -> dict:
    abs_audio_path = os.path.abspath(raw_audio_path)
            
    if not os.path.exists(abs_audio_path):
        raise FileNotFoundError(f"Missing file: {abs_audio_path}")

    # Using your preferred, natural prompt!
    prompt_text = (
        f"{instruction}\n\n"
        "Please think step-by-step about the audio and the choices provided. "
        "At the very end of your response, explicitly state your final prediction "
        "using only the single letter of the correct choice (e.g., A, B, C, or D)."
    )
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio", "path": abs_audio_path},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,
        )

    decoded_output = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )[0] 
    
    return parse_model_output(decoded_output)
# ==========================================
# 4. EXECUTION LOOP (SEQUENTIAL)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input .jsonl manifest")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl results")
    parser.add_argument("--data_root", type=str, default="", help="Root dir for audio files")
    parser.add_argument("--num_runs", type=int, default=3)
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    
    NUM_RUNS = 1    
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]
        
    print(f"Loaded {len(dataset)} items. Starting sequential inference ({NUM_RUNS} runs per item)...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Loop through one item at a time
        for item in tqdm(dataset, desc="Processing Items"):
            instruction = item['instruction']
            audio_path = item['audio_path']
            audio_path = audio_path.replace("{DATA_ROOT}", args.data_root) if "{DATA_ROOT}" in audio_path else os.path.join(args.data_root, audio_path)

            true_raw = item['answer']
            true_letter = extract_true_letter(true_raw)
            
            item_runs_data = []
            
            # Execute multiple runs for the single item
            for run_idx in range(NUM_RUNS):
                try:
                    result = run_audio_inference(instruction, audio_path)
                    item_runs_data.append(result)
                except Exception as e:
                    print(f"\n[ERROR] ID: {item['id']}, Run {run_idx+1}: {e}")
                    item_runs_data.append({"predicted_choice": None, "reasoning": f"CRASH: {e}", "raw_output": ""})
            
            # Aggregate stats for this item
            predictions_for_item = []
            scores_for_item = []
            reasonings_for_item = []
            
            for result in item_runs_data:
                pred_letter = result['predicted_choice']
                
                is_correct = 1 if (pred_letter and true_letter and pred_letter == true_letter) else 0
                
                predictions_for_item.append(pred_letter)
                scores_for_item.append(is_correct)
                reasonings_for_item.append(result['reasoning'])
            
            accuracy_std = float(np.std(scores_for_item))
            mean_accuracy = float(np.mean(scores_for_item))
            
            final_record = {
                "id": item["id"],
                "audio_path": audio_path,
                "true_answer_letter": true_letter,
                "runs_predictions": predictions_for_item,
                "runs_scores": scores_for_item,
                "mean_accuracy": mean_accuracy,
                "accuracy_std": accuracy_std,
                "runs_reasonings": reasonings_for_item
            }
            
            f.write(json.dumps(final_record, ensure_ascii=False) + '\n')
            f.flush()
                
    print(f"\nInference complete! Variance test results saved to {output_file}")