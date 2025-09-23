import json
import re

def process_line(line):
    data = json.loads(line)
    # Extract number from audio path using regex
    audio_num = re.search(r'mmar_audio_(\d+)\.wav', data['audio_path'])
    if audio_num:
        # Create new ID in the format mmar_X
        data['id'] = f'mmar_{audio_num.group(1)}'
    return json.dumps(data)

# Read and process the file
input_file = 'data/mmar/mmar_test_standardized.jsonl'
output_lines = []

with open(input_file, 'r') as f:
    for line in f:
        processed_line = process_line(line.strip())
        output_lines.append(processed_line)

# Write back to the file
with open(input_file, 'w') as f:
    for line in output_lines:
        f.write(line + '\n')

print("IDs have been updated successfully!")
