import argparse
import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ==========================================
# 1. LOAD MODEL GLOBALLY
# ==========================================
# Enable flash_attention_2 if your A100 environment supports it for memory/speed gains
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", 
    torch_dtype="auto", 
    device_map="auto",
    # attn_implementation="flash_attention_2", 
)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model.eval()

# ==========================================
# 2. PARSING FUNCTIONS
# ==========================================
def parse_model_output(raw_text: str) -> dict:
    reasoning = ""
    predicted_choice = None

    # 1. Try forgiving template parsing first (catches <Reasoning>, <Reasonreason>, etc.)
    reason_match = re.search(r'<Reason.*?>(.*?)</Reason.*?>', raw_text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip()
        
    conclusion_match = re.search(r'<Conclu.*?>\s*([A-Za-z])\s*</Conclu.*?>', raw_text, re.IGNORECASE)
    if conclusion_match:
        predicted_choice = conclusion_match.group(1).upper()
        
    # 2. Fallback if the model completely ignored the tags
    if not predicted_choice:
        cleaned_text = raw_text.strip()
        fallback_match = re.search(r'(?:[^a-zA-Z]|^)([A-D])(?:[^a-zA-Z]*)$', cleaned_text, re.IGNORECASE)
        if fallback_match:
            predicted_choice = fallback_match.group(1).upper()
            if not reasoning:
                reasoning = cleaned_text[:fallback_match.start()].strip()

    return {
        "raw_output": raw_text,
        "reasoning": reasoning if reasoning else raw_text,
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

    # Enforce the strict XML template in the prompt
    prompt_text = (
        f"{instruction}\n\n"
        "You must analyze the audio and provide your answer strictly following the template below. "
        "Do not include any other text outside of these tags.\n\n"
        "Template:\n"
        "<Reasoning>\n"
        "[Your step-by-step thinking here]\n"
        "</Reasoning>\n"
        "<Conclusion>\n"
        "[Single letter choice here, e.g., A]\n"
        "</Conclusion>"
    )
    
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio", "audio": abs_audio_path},
            ],
        }
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=False
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        text_ids, generated_audio = model.generate(
            **inputs, 
            max_new_tokens=1024,
            use_audio_in_video=False
        )

    # Slice the text_ids to remove the prompt tokens before decoding
    generated_ids = text_ids[:, inputs.input_ids.shape[1]:]

    decoded_output = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
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
        for item in tqdm(dataset, desc="Processing Items"):
            instruction = item['instruction']
            audio_path = item['audio_path']
            audio_path = audio_path.replace("{DATA_ROOT}", args.data_root) if "{DATA_ROOT}" in audio_path else os.path.join(args.data_root, audio_path)

            true_raw = item['answer']
            true_letter = extract_true_letter(true_raw)
            
            item_runs_data = []
            
            for run_idx in range(NUM_RUNS):
                try:
                    result = run_audio_inference(instruction, audio_path)
                    item_runs_data.append(result)
                except Exception as e:
                    print(f"\n[ERROR] ID: {item['id']}, Run {run_idx+1}: {e}")
                    item_runs_data.append({"predicted_choice": None, "reasoning": f"CRASH: {e}", "raw_output": ""})
            
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
                
    print(f"\nInference complete! Results saved to {output_file}")