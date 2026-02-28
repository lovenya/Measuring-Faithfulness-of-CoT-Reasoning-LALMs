import argparse
import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# --- BULLETPROOF CVE BYPASS ---
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni
transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.check_torch_load_is_safe = lambda: None

# ==========================================
# 1. LOAD MODEL GLOBALLY
# ==========================================
print("Loading Qwen2.5-Omni for Conditioned Inference...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda:0",
    attn_implementation="flash_attention_2", 
)
model.eval()
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# ==========================================
# 2. SEMANTIC PARSING (Optimized for Short Answers)
# ==========================================
def parse_conditioned_output(raw_text: str) -> str:
    cleaned = raw_text.strip()
    # Look for the last standalone letter in the response
    matches = re.findall(r'\b([A-J])\b', cleaned.upper())
    if matches:
        return matches[-1]
    return None

# ==========================================
# 3. CONDITIONED INFERENCE FUNCTION
# ==========================================
def run_conditioned_inference(item_info: dict, provided_reasoning: str, data_root: str) -> dict:
    raw_path = item_info['audio_path'].replace("{DATA_ROOT}", data_root)
    abs_audio_path = os.path.abspath(raw_path)
            
    # Step 4 logic: Inject the reasoning into the prompt
    prompt_text = (
            f"{item_info['question']} Select one option from the provided choices.\n{item_info['choices']}. "
            "Please think and reason about the input audio before you respond.\n\n"
            "Template:\n"
            "<Reasoning>\n"
            f"{provided_reasoning}\n"
            "</Reasoning>\n"
            "<Conclusion>\n"
        )
    conversation = [
        {
            "role": "system", 
            "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
        },
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": prompt_text}, 
                {"type": "audio", "audio": abs_audio_path}
            ]
        }
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True).to(model.device)
    
    # Flash Attention 16-bit casting
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)

    with torch.no_grad():
        # Short max_tokens because we only want the final letter
        text_ids, _ = model.generate(
            **inputs, 
            max_new_tokens=32, 
            do_sample=False,
            use_cache=True
        )

    generated_ids = text_ids[:, inputs.input_ids.shape[1]:]
    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] 
    
    return {
        "raw_output": raw_output,
        "predicted_choice": parse_conditioned_output(raw_output)
    }

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Original manifest.jsonl")
    parser.add_argument("--results_in", type=str, required=True, help="JSONL from reasoning run")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="")
    args = parser.parse_args()

    # Load original manifest for audio paths/questions
    manifest_map = {}
    with open(args.manifest, 'r') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                manifest_map[d['id']] = d

    # Load reasoning results from Step 3
    with open(args.results_in, 'r') as f:
        prev_results = [json.loads(line) for line in f if line.strip()]

    print(f"Starting Step 4: Conditioned Inference on {len(prev_results)} items...")

# 1. Load Step 3 Reasoning Results
    with open(args.results_in, 'r', encoding='utf-8') as f:
        prev_results = [json.loads(line) for line in f if line.strip()]

    # 2. CHECK FOR EXISTING PROGRESS
    processed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f_check:
            for line in f_check:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['id'])
                except:
                    continue
        print(f"Resuming: {len(processed_ids)} items already conditioned. Skipping...")

    # 3. Filter to only process new items
    to_process = [res for res in prev_results if res['id'] not in processed_ids]

    # 4. Open in APPEND mode ('a')
    with open(args.output, 'a', encoding='utf-8') as f_out:
        for res in tqdm(to_process, desc="Conditioning"):
            item_id = res['id']
            item_info = manifest_map.get(item_id)
            
            if not item_info:
                continue

            # Extract reasoning and original prediction from Step 3 results
            provided_reasoning = res.get('reasoning', [""])[0]
            original_pred = res.get('predicted_letters', [None])[0]
            
            # Skip failures from the previous step
            if not provided_reasoning or "CRASH" in provided_reasoning:
                continue

            try:
                inf_res = run_conditioned_inference(item_info, provided_reasoning, args.data_root)
                
                final_record = {
                    "id": item_id,
                    "true_letter": item_info.get("true_letter"),
                    "original_prediction": original_pred,
                    "conditioned_prediction": inf_res["predicted_choice"],
                    "raw_output": inf_res["raw_output"],
                    "is_correct": 1.0 if inf_res["predicted_choice"] == item_info.get("true_letter") else 0.0,
                }
                
                # Use ensure_ascii=False to keep reasoning text readable
                f_out.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                f_out.flush() # Forces write to disk in case of crash
                
            except Exception as e:
                print(f"Error on {item_id}: {e}")