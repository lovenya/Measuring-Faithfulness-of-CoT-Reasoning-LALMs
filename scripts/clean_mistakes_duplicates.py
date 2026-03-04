import json
import os
from collections import defaultdict

def clean_file(filepath):
    if not os.path.exists(filepath):
        return

    print(f"Cleaning {filepath}...")
    
    # Read all lines
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
                
    # Group by ID
    by_id = defaultdict(list)
    for entry in lines:
        by_id[entry['id']].append(entry)
        
    # For each ID, keep only the most recent complete set
    cleaned_lines = []
    for q_id, entries in by_id.items():
        # Usually each ID has exactly 1 entry in paraphrasing/mistakes (unless it's an empty placeholder, which is also 1 entry)
        # But wait! The actual generation outputs multiple sentences per ID!
        # Let's group by (id, chain_id, sentence_idx) and keep the last one.
        
        # ACTUALLY, Mistral generations just append all sentences for an ID in a block. 
        # So we should just take the LAST N entries for this ID where N is the number of sentences in that ID's most recent run.
        # An easier way: group by (id, chain_id, sentence_idx) and keep the LAST occurrence of each.
        
        # Let's do that!
        pass

    # Actually, the safest way is to keep the last occurrence of each (id, chain_id, sentence_idx)
    unique_entries = {}
    for entry in lines:
        key = (entry['id'], entry.get('chain_id', 0), entry.get('sentence_idx', 0))
        unique_entries[key] = entry
        
    # Sort them back into original ID order or just write them out
    # Writing in the order they were last seen is fine.
    
    # Wait, if an earlier run had 3 sentences, and a later run had 1 placeholder, 
    # the keys would be (id, 0, 0), (id, 0, 1), (id, 0, 2) from earlier run,
    # and (id, 0, -1) from the later run.
    # We shouldn't mix them! We should only keep the LAST continuous block for each ID.
    
    # Better logic: traverse backwards. Once we see an ID, we collect all its entries until we see a different ID.
    # That block is the last run for that ID. We ignore any earlier blocks for that ID.
    
    final_blocks = {}
    current_id = None
    current_block = []
    
    for entry in reversed(lines):
        q_id = entry['id']
        
        if q_id != current_id:
            # We switched to a new ID (or first ID)
            if current_id is not None and current_id not in final_blocks:
                # Save the completed block (reverse it back to normal chronological order)
                final_blocks[current_id] = list(reversed(current_block))
            current_id = q_id
            current_block = [entry]
        else:
            # Same ID, keep accumulating the block
            current_block.append(entry)
            
    # Don't forget the last block
    if current_id is not None and current_id not in final_blocks:
        final_blocks[current_id] = list(reversed(current_block))
        
    # Reconstruct the file in the original ID order (based on their first appearance or just sort by ID)
    # Let's preserve original relative order of IDs by tracking when we first saw them
    seen_ids = []
    for entry in lines:
        if entry['id'] not in seen_ids:
            seen_ids.append(entry['id'])
            
    # Write updated lines
    with open(filepath, 'w') as f:
        for q_id in seen_ids:
            for entry in final_blocks[q_id]:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
    print(f"Done. Original lines: {len(lines)}, New lines: {sum(len(b) for b in final_blocks.values())}")

# Clean all mistral raw files
base_dir = "results/external_llm_perturbations/mistral"
for model in ["flamingo_hf", "qwen_omni"]:
    for ds in ["mmar", "mmau", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]:
        for exp in ["mistakes.jsonl", "paraphrased.jsonl"]:
            path = os.path.join(base_dir, model, ds, "raw", exp)
            clean_file(path)

