# diagnose_missing_ids.py
import json

def get_unique_ids(filepath):
    """Reads a JSONL file and returns a set of unique 'id' values."""
    ids = set()
    with open(filepath, 'r') as f:
        for line in f:
            ids.add(json.loads(line)['id'])
    return ids

baseline_path = 'results/baseline/baseline_mmar.jsonl'
no_reasoning_path = 'results/no_reasoning/no_reasoning_mmar.jsonl'

baseline_ids = get_unique_ids(baseline_path)
no_reasoning_ids = get_unique_ids(no_reasoning_path)

print(f"Unique question IDs in '{baseline_path}': {len(baseline_ids)}")
print(f"Unique question IDs in '{no_reasoning_path}': {len(no_reasoning_ids)}")

missing_ids = baseline_ids - no_reasoning_ids

if missing_ids:
    print(f"\nFound {len(missing_ids)} question IDs that are in baseline but MISSING from no_reasoning:")
    # print(sorted(list(missing_ids))) # Uncomment to see the full list
else:
    print("\nAll question IDs from baseline are present in no_reasoning.")