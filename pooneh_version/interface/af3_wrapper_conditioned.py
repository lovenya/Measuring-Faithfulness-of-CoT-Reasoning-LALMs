import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from peft import PeftModel
from huggingface_hub import snapshot_download
import argparse

# ==========================================
# 1. LOAD MODEL
# ==========================================
print("Loading AF3 Model...")
model_id = "nvidia/audio-flamingo-3-hf"
local_id = snapshot_download(model_id)
processor = AutoProcessor.from_pretrained(local_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(local_id, device_map="auto")

non_lora_path = os.path.join(local_id, "think", "non_lora_trainables.bin")
if os.path.exists(non_lora_path):
    model.load_state_dict(torch.load(non_lora_path), strict=False)
model = PeftModel.from_pretrained(model, local_id, subfolder="think")
model.eval()

# ==========================================
# 2. ROBUST SEMANTIC PARSING
# ==========================================
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", 
             "or", "in", "on", "it", "this", "that", "for", "with", "as", 
             "by", "at", "but", "not", "be", "about", "which", "they", "i"}

def get_meaningful_words(text):
    if not text: return set()
    clean = re.sub(r'[^\w\s]', '', str(text)).strip().lower()
    return set(clean.split()) - STOPWORDS

def parse_conditioned_output(raw_text: str, choices_list: list = None) -> str:
    if not raw_text: return None
    cleaned = raw_text.strip()
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    # Priority 1: Strict Formatting Match in the last 100 chars
    end_chunk = cleaned[-100:]
    paren_matches = list(re.finditer(r'\(([A-J])\)', end_chunk, re.IGNORECASE))
    if paren_matches:
        return paren_matches[-1].group(1).upper()
        
    prefix_match = list(re.finditer(r'(?:option|choice|answer|answer\s*is|is)\s*[:*]*\s*([A-J])\b', end_chunk, re.IGNORECASE))
    if prefix_match:
        return prefix_match[-1].group(1).upper()

    # Priority 2: Semantic Word Overlap
    if choices_list:
        target_words = get_meaningful_words(cleaned)
        best_letter = None
        max_score = 0.0
        
        for i, opt in enumerate(choices_list):
            opt_words = get_meaningful_words(opt)
            if not opt_words: continue
            
            overlap = opt_words.intersection(target_words)
            score = len(overlap) + (len(overlap) / len(opt_words))
            
            if score > max_score and len(overlap) > 0:
                max_score = score
                best_letter = letters[i]
        
        if best_letter:
            return best_letter

    # Priority 3: Standalone Letter
    standalone = re.search(r'\b([A-J])\b[^\w]*$', cleaned[-30:], re.IGNORECASE)
    if standalone and standalone.group(1).upper() not in ['A', 'I']:
        return standalone.group(1).upper()

    return None

# ==========================================
# 3. CONDITIONED INFERENCE
# ==========================================
def run_conditioned_inference(item: dict, provided_reasoning: str, data_root: str) -> dict:
    raw_path = item['audio_path'].replace("{DATA_ROOT}", data_root)
    abs_audio_path = os.path.abspath(raw_path)

    # Append reasoning to the prompt and restrict to a short, direct answer
    prompt_text = (
            f"{item['question']} Select one option from the provided choices.\n{item['choices']}. "
            "Please think and reason about the input audio before you respond.\n\n"
            f"{provided_reasoning}\n\n"
            "Therefore, the answer is:"
        )

    conversation = [{"role": "user", "content": [
        {"type": "text", "text": prompt_text},
        {"type": "audio", "path": abs_audio_path},
    ]}]

    inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_dict=True).to(model.device)

    with torch.no_grad():
        # Short token limit because the model only needs to output the final conclusion
        outputs = model.generate(**inputs, max_new_tokens=20)

    raw_output = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    return {
        "raw_output": raw_output,
        "predicted_choice": parse_conditioned_output(raw_output, item.get('choices_list', []))
    }

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Original manifest file with questions and choices")
    parser.add_argument("--results_in", type=str, required=True, help="JSONL from the previous reasoning run")
    parser.add_argument("--output", type=str, required=True, help="Path to save the new conditioned results")
    parser.add_argument("--data_root", type=str, default="")
    args = parser.parse_args()

    # 1. Load Original Manifest for Question/Choices
    manifest_map = {}
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                manifest_map[d['id']] = d

    # 2. Load Reasoning Results
    with open(args.results_in, 'r', encoding='utf-8') as f:
        prev_results = [json.loads(line) for line in f if line.strip()]

    print(f"\nLoaded {len(prev_results)} items. Starting Conditioned Inference...")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
 # 1. LOAD COMPLETED IDs FROM PREVIOUS RUN
    processed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f_check:
            for line in f_check:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['id'])
                except json.JSONDecodeError:
                    continue # Skip partially written lines
        print(f"Resuming: Found {len(processed_ids)} already processed items. Skipping...")

    # 2. FILTER TO-PROCESS LIST
    # Only process items that are NOT in processed_ids
    to_process = [res for res in prev_results if res['id'] not in processed_ids]

    # 3. OPEN IN APPEND MODE ('a')
    with open(args.output, 'a', encoding='utf-8') as f_out:
        for res in tqdm(to_process, desc="Conditioned Inference"):
            item_id = res['id']
            item_info = manifest_map.get(item_id)
            
            if not item_info:
                print(f"[Warning] ID {item_id} not found in manifest. Skipping.")
                continue

            # Extract reasoning from the first run of previous results
            provided_reasoning = res.get('reasoning', [""])[0]
            original_pred = res.get('predicted_letters', [None])[0]
            
            # Skip if previous step failed
            if not provided_reasoning or "CRASH" in provided_reasoning:
                continue

            try:
                inference_res = run_conditioned_inference(item_info, provided_reasoning, args.data_root)
                new_pred = inference_res["predicted_choice"]
                true_letter = item_info.get("true_letter")
                
                final_record = {
                    "id": item_id,
                    "true_answer": item_info.get("answer"),
                    "true_letter": true_letter,
                    "original_prediction": original_pred,
                    "conditioned_prediction": new_pred,
                    "raw_conditioned_output": inference_res["raw_output"],
                    "provided_reasoning": provided_reasoning,
                    "is_correct": 1.0 if (new_pred and true_letter and new_pred == true_letter) else 0.0,
                    "prediction_changed": (new_pred != original_pred)
                }
                
                # Write and flush immediately
                f_out.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                f_out.flush()
                
            except Exception as e:
                print(f"\n[ERROR] ID: {item_id}: {e}")

    print(f"\n✅ Inference complete! Results saved to {args.output}")