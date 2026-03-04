import json
import os
from collections import defaultdict

def check_missing_paraphrased(filepath):
    if not os.path.exists(filepath):
        return

    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))

    by_id = defaultdict(list)
    for entry in lines:
        by_id[entry['id']].append(entry['num_sentences_paraphrased'])

    missing_count = 0
    missing_ids = []
    for q_id, seqs in by_id.items():
        max_seq = max(seqs)
        # Sequence should be 0, 1, 2... max_seq  OR  1, 2... max_seq
        # Usually it starts from 1, though empty is 0
        expected = set(range(1, max_seq + 1))
        actual = set(seqs)
        
        missing = expected - actual
        if missing:
            missing_count += len(missing)
            missing_ids.append(q_id)

    if missing_count > 0:
        print(f"{os.path.basename(os.path.dirname(os.path.dirname(filepath)))}: Missing {missing_count} sentences across {len(missing_ids)} IDs.")

base_dir = "results/external_llm_perturbations/mistral"
for model in ["flamingo_hf", "qwen_omni"]:
    for ds in ["mmar", "mmau", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]:
        path = os.path.join(base_dir, model, ds, "raw", "paraphrased.jsonl")
        check_missing_paraphrased(path)
