#!/usr/bin/env python3
import json
import re
from pathlib import Path

def parse_timestamps_from_pooneh_path(path: str):
    """
    path is e.g. {DATA_ROOT}/audio/BV1j1ABeqEyp_00-06-02_00-06-16.wav
    Returns (video_id, start_ts, end_ts) e.g. ("BV1j1ABeqEyp", "00:06:02", "00:06:16")
    """
    filename = path.split('/')[-1].replace('.wav', '')
    parts = filename.rsplit('_', 2)
    if len(parts) == 3:
        vid = parts[0]
        start = parts[1].replace('-', ':')
        end = parts[2].replace('-', ':')
        return vid, start, end
    return filename, "", ""

def extract_video_id_from_url(url: str, source: str) -> str:
    if "bilibili" in source.lower():
        m = re.search(r'/video/(BV[a-zA-Z0-9]+)', url)
        if m: return m.group(1)
        m = re.search(r'bvid=(BV[a-zA-Z0-9]+)', url)
        if m: return m.group(1)
    elif "youtube" in source.lower():
        m = re.search(r'v=([a-zA-Z0-9_-]+)', url)
        if m: return m.group(1)
        m = re.search(r'youtu\.be/([a-zA-Z0-9_-]+)', url)
        if m: return m.group(1)
    return "unknown"

def sanitize_cot(reasoning: str) -> str:
    reasoning = re.sub(r'<Reasoning>', '', reasoning, flags=re.IGNORECASE)
    reasoning = re.sub(r'</Reasoning>', '', reasoning, flags=re.IGNORECASE)
    return reasoning.strip()

def main():
    base_dir = Path("/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs")
    
    our_data_file = base_dir / "data/mmar/mmar_test_standardized.jsonl"
    pooneh_manifest_file = base_dir / "pooneh_version/data/mmar/mmar_manifest.jsonl"
    
    # 1. Load Our Standardized Data
    print("Loading Standardized MMAR Dataset...")
    our_data = {}  # our_id -> entry
    our_key_map = {} # (video_id, start_ts, end_ts) -> our_id
    our_question_map = {} # question.lower() -> our_id
    
    with open(our_data_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            our_id = d['id']
            our_data[our_id] = d
            
            vid = extract_video_id_from_url(d['url'], d['source'])
            
            # d['timestamp'] e.g. "00:06:02,00:06:16"
            ts_parts = d['timestamp'].split(',')
            if len(ts_parts) >= 2:
                start_ts, end_ts = ts_parts[0].strip(), ts_parts[1].strip()
            else:
                start_ts, end_ts = d['timestamp'], d['timestamp']
            
            our_key_map[(vid, start_ts, end_ts)] = our_id
            
            # exact question fallback
            q_norm = d['question'].strip().lower()
            our_question_map[q_norm] = our_id
            
    print(f"  Loaded {len(our_data)} standardized entries")
    
    # 2. Map Pooneh ID -> Our ID
    print("Mapping Pooneh MMAR IDs to Standardized IDs...")
    pooneh_to_our_id = {}
    with open(pooneh_manifest_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            poo_id = d['id']
            
            vid, start_ts, end_ts = parse_timestamps_from_pooneh_path(d['audio_path'])
            
            # Primary Map
            our_id = our_key_map.get((vid, start_ts, end_ts))
            
            # Fallback map
            if not our_id:
                q_norm = d['question'].strip().lower()
                our_id = our_question_map.get(q_norm)
                
            if our_id:
                pooneh_to_our_id[poo_id] = our_id
                
    print(f"  Mapped {len(pooneh_to_our_id)} / 997 entries")
    
    # 3. Process the Baselines
    models = {
        "af3": {
            "target": "flamingo_hf",
            "result_file": "pooneh_version/result/baseline/af3/mmar/baseline_mmar_REAS_post_process.jsonl",
            "has_reasoning": False
        },
        "qwen2.5": {
            "target": "qwen_omni",
            "result_file": "pooneh_version/result/baseline/qwen2.5/mmar/baseline_mmar_REAS.jsonl",
            "has_reasoning": True
        }
    }
    
    for source_model, cfg in models.items():
        pooneh_result = base_dir / cfg['result_file']
        target_model = cfg['target']
        
        out_dir = base_dir / f"results/{target_model}/baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"baseline_{target_model}_mmar.jsonl"
        
        if not pooneh_result.exists():
            print(f"Skipping {source_model} -> {target_model}: file not found.")
            continue
            
        print(f"\nProcessing MMAR: {source_model} -> {target_model}...")
        processed = 0
        unmatched = 0
        
        with open(pooneh_result, 'r') as f_in, open(out_file, 'w') as f_out:
            for line in f_in:
                try:
                    poo = json.loads(line)
                except json.JSONDecodeError:
                    continue
                    
                poo_id = poo['id']
                our_id = pooneh_to_our_id.get(poo_id)
                
                if not our_id:
                    unmatched += 1
                    continue
                    
                ds = our_data[our_id]
                
                # Format choices
                choices_list = ds.get('choices', [])
                if isinstance(choices_list, list):
                    formatted = []
                    for idx, c in enumerate(choices_list):
                        letter = chr(ord('A') + idx)
                        formatted.append(f"({letter}) {c}")
                    choices_str = "\n".join(formatted)
                else:
                    choices_str = str(choices_list)
                    
                # Extract reasoning
                if cfg['has_reasoning']:
                    reasoning_list = poo.get('reasoning', [])
                    reasoning = reasoning_list[0] if reasoning_list else ""
                else:
                    raw_outputs = poo.get('raw_model_outputs', [])
                    reasoning = raw_outputs[0] if raw_outputs else ""
                    
                sanitized = sanitize_cot(reasoning)
                
                raw_outputs = poo.get('raw_model_outputs', [])
                final_answer_raw = raw_outputs[0] if raw_outputs else ""
                
                pred_letters = poo.get('predicted_letters', [])
                predicted_choice = pred_letters[0] if pred_letters else ""
                
                correct_choice = poo.get('true_letter', "")
                is_correct = bool(poo.get('accuracy', 0.0) == 1.0)
                
                rec = {
                    "id": our_id,
                    "source_id": poo_id,
                    "chain_id": 0,
                    "predicted_choice": predicted_choice,
                    "correct_choice": correct_choice,
                    "is_correct": is_correct,
                    "final_answer_raw": final_answer_raw,
                    "final_prompt_messages": [],
                    "question": ds['question'],
                    "choices": choices_str,
                    "audio_path": ds['audio_path'],
                    "generated_cot": reasoning,
                    "sanitized_cot": sanitized
                }
                
                f_out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                processed += 1
                
        print(f"  Processed: {processed}  |  Unmatched: {unmatched}")

if __name__ == "__main__":
    main()
