import json
import os
import nltk
from collections import defaultdict

def reconstruct_paraphrases(filepath):
    if not os.path.exists(filepath):
        return

    print(f"Checking {filepath} for missing progressives...")

    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))

    # Group by ID
    by_id = defaultdict(list)
    for entry in lines:
        by_id[entry['id']].append(entry)

    new_entries = []
    reconstructed_count = 0

    for q_id, entries in by_id.items():
        # Get existing n values
        existing_n = set(e['num_sentences_paraphrased'] for e in entries)
        
        # Get max n entry
        max_entry = max(entries, key=lambda e: e['num_sentences_paraphrased'])
        max_n = max_entry['num_sentences_paraphrased']
        
        # We need 1 to max_n
        expected_n = set(range(1, max_n + 1))
        missing_n = expected_n - existing_n
        
        if not missing_n:
            continue
            
        # We need to construct them from max_entry
        orig_sents = nltk.sent_tokenize(max_entry['original_text'])
        para_sents = nltk.sent_tokenize(max_entry['paraphrased_text'])
        
        # Sometimes Mistral changes the number of sentences. 
        # If so, we just proportionally slice para_sents based on n / max_n, or just take the first n sentences of para_sents if it's close.
        # But safest is to map n original sentences to approximately n * (len(para_sents) / len(orig_sents)) paraphrased sentences.
        para_ratio = len(para_sents) / max(1, len(orig_sents))

        for n in missing_n:
            # We are paraphrasing the first n original sentences.
            curr_orig_text = " ".join(orig_sents[:n])
            
            # Take roughly the same proportion of paraphrased sentences
            num_para_to_take = max(1, round(n * para_ratio))
            # Don't take more than available
            num_para_to_take = min(num_para_to_take, len(para_sents))
            
            curr_para_text = " ".join(para_sents[:num_para_to_take])
            
            new_entry = {
                "id": q_id,
                "chain_id": max_entry['chain_id'],
                "num_sentences_paraphrased": n,
                "original_text": curr_orig_text,
                "paraphrased_text": curr_para_text
            }
            new_entries.append(new_entry)
            reconstructed_count += 1
            
    if reconstructed_count == 0:
        print("  - All good, nothing missing.")
        return

    print(f"  - Reconstructed {reconstructed_count} missing entries. Appending to file...")
    with open(filepath, 'a') as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("  - Done.")


base_dir = "results/external_llm_perturbations/mistral"
for model in ["flamingo_hf", "qwen_omni"]:
    for ds in ["mmar", "mmau", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]:
        path = os.path.join(base_dir, model, ds, "raw", "paraphrased.jsonl")
        reconstruct_paraphrases(path)

print("All missing progressive paraphrases reconstructed!")
