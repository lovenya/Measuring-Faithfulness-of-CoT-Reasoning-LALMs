import pandas as pd
import json

import pandas as pd
import json

def convert_jsonl_to_jasco_csv(jsonl_path, metadata_csv_path, output_csv_path):
    # Load your specific metadata CSV
    meta_df = pd.read_csv(metadata_csv_path)
    
    # Force the 'audio_id' column to be a string so it perfectly matches the JSONL "id": "1"
    meta_df['audio_id'] = meta_df['audio_id'].astype(str)
    
    # Create a fast lookup dictionary using 'audio_id' as the key
    meta_dict = meta_df.set_index('audio_id').to_dict('index')

    jasco_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # Get the ID as a string to match the dictionary
            item_id_str = str(item.get("id", ""))
            
            # Extract the model's prediction safely
            raw_outs = item.get("raw_model_outputs", [""])
            prediction = raw_outs[0] if raw_outs else ""
            
            # Look up the corresponding row in your CSV
            row_meta = meta_dict.get(item_id_str, {})
            
            # Build the exact dictionary JASCO expects by mapping YOUR column names
            jasco_row = {
                "id": item_id_str,
                "tgt_text": item.get("true_answer", row_meta.get("Correct Answer", "")),
                "target_keywords": item.get("keyword", row_meta.get("Correct Answer Keyword", "")),
                "allm_output": prediction,
                
                # We add a default prompt here since it wasn't in your CSV snippet
                "prompt": "What are the speakers likely doing?", 
                
                # MAPPING YOUR CSV COLUMNS
                "audio_sound": row_meta.get("Audio", ""),
                "spoken_text": row_meta.get("Speech", ""),
                "audio_only_target": row_meta.get("Audio-Only Answer", ""),
                "speech_only_target": row_meta.get("Speech-Only Answer", "")
            }
            jasco_data.append(jasco_row)

    # Save to CSV for the vLLM judge
    pd.DataFrame(jasco_data).to_csv(output_csv_path, index=False)
    print(f"✅ Saved JASCO-formatted CSV to {output_csv_path}")
# Run this for all 3 conditions: Original, Audio Mask, Speech Mask
convert_jsonl_to_jasco_csv(
    jsonl_path="pooneh_version/result/baseline/qwen2.5/jasco/audio_mask/baseline_audio_mask_REAS.jsonl",
    metadata_csv_path="data/jasco/dataset/v0/v0.csv",
    output_csv_path="pooneh_version/result/baseline/qwen2.5/jasco/audio_mask/baseline_audio_mask_REAS.csv"
)