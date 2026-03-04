import json
import sys
from collections import defaultdict

def check_file(filepath):
    print(f"\nChecking {filepath}...")
    counts = defaultdict(int)
    total_lines = 0
    with open(filepath, 'r') as f:
        for line in f:
            total_lines += 1
            data = json.loads(line)
            sample_id = data.get('id')
            chain_id = data.get('chain_id')
            if sample_id is not None and chain_id is not None:
                counts[(sample_id, chain_id)] += 1
    
    extra_counts = {k: v for k, v in counts.items() if v > 21}
    print(f"Total lines: {total_lines}")
    print(f"Unique (id, chain_id) pairs: {len(counts)}")
    print(f"Pairs with > 21 lines: {len(extra_counts)}")
    
    if extra_counts:
        print("Examples of pairs with extra lines (showing up to 5):")
        for i, (k, v) in enumerate(extra_counts.items()):
            if i < 5:
                print(f"  ID: {k[0]}, Chain: {k[1]}, Count: {v}")
            else:
                break

check_file('results/flamingo_hf/random_partial_filler_text/random_partial_filler_text_flamingo_hf_sakura-animal-lorem.part_2.jsonl')
check_file('results/flamingo_hf/random_partial_filler_text/random_partial_filler_text_flamingo_hf_sakura-animal-lorem.part_3.jsonl')
check_file('results/flamingo_hf/random_partial_filler_text/random_partial_filler_text_flamingo_hf_sakura-animal-lorem.part_4.jsonl')
