import json
import re
import numpy as np
from tqdm import tqdm

def simplify(text):
    """Removes non-alphanumeric chars and makes lowercase for safe matching."""
    if not text: return ""
    return re.sub(r'[^\w\s]', '', str(text)).strip().lower()

def build_flexible_pattern(choice_text):
    """
    Strips suffixes to create a highly flexible regex pattern.
    - 'sadness' -> 'sad[a-z]*' (matches 'sad', 'sadly', 'sadness')
    - 'viviparous' -> 'vivipar[a-z]*' (matches 'viviparous', 'viviparity')
    - 'carnivore' -> 'carniv[a-z]*' (matches 'carnivore', 'carnivorous')
    """
    words = choice_text.split()
    pattern_words = []
    
    # List of common suffixes ordered from longest to shortest
    suffixes = r'(ness|ment|tion|sion|able|ible|ance|ence|ous|ity|ing|ive|ful|less|est|ed|ly|er|es|s|y)$'
    
    for w in words:
        if len(w) > 3:
            # Strip grammatical suffixes
            core = re.sub(suffixes, '', w)
            
            # If the root word is STILL very long, truncate it slightly to handle 
            # severe vowel changes (e.g., 'insectivore' -> 'insectiv')
            if len(core) > 7:
                core = core[:6]
                
            # If stripping destroyed the word (made it 1-2 letters), revert it
            if len(core) < 3:
                core = w
        else:
            # Keep short words intact (e.g., 'cat', 'dog', 'sad')
            core = w
            
        # Append [a-z]* so the core matches any ending the model decided to use
        pattern_words.append(re.escape(core) + r'[a-z]*')
        
    # Join words with flexible spacing
    return r'\b' + r'\s*'.join(pattern_words) + r'\b'

def final_sentence_parser(raw_text, choices_list):
    if not raw_text: return None
    
    cleaned = raw_text.strip()
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    # 1. STRICT PARENTHESES CHECK (Priority 1)
    paren_matches = list(re.finditer(r'\(([A-J])\)', cleaned, re.IGNORECASE))
    if paren_matches:
        return paren_matches[-1].group(1).upper()
        
    # 2. FLEXIBLE MORPHOLOGICAL KEYWORD MATCH
    if choices_list:
        # Isolate the final sentence(s) to avoid matching distractors early in the CoT
        sentences = [s.strip() for s in re.split(r'[.!?\n]', cleaned) if s.strip()]
        if sentences:
            conclusion = sentences[-1].lower()
            if len(conclusion.split()) < 4 and len(sentences) > 1:
                # Grab the previous sentence too if the last one is a fragment
                conclusion = sentences[-2].lower() + " " + sentences[-1].lower()
                
            # Sort choices by length descending so longer phrases match first
            sorted_choices = sorted(enumerate(choices_list), key=lambda x: len(str(x[1])), reverse=True)
            
            for i, opt in sorted_choices:
                clean_opt = simplify(opt)
                if not clean_opt: continue
                
                # Apply the smart suffix-stripping pattern
                flexible_pattern = build_flexible_pattern(clean_opt)
                if re.search(flexible_pattern, conclusion):
                    return letters[i]
                    
    # 3. EXPLICIT "ANSWER IS X" FALLBACK
    explicit_match = list(re.finditer(r'(?:answer|choice|option|is)\s*([A-J])\b', cleaned[-50:], re.IGNORECASE))
    if explicit_match:
        return explicit_match[-1].group(1).upper()

    return None

def update_with_smart_stemming(manifest_path, results_path, output_path):
    # Load manifest mappings
    manifest_dict = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            manifest_dict[item['id']] = item.get('choices_list', [])

    scores = []
    total, nulls, recovered_nulls = 0, 0, 0

    print(f"Reparsing results from: {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Applying Smart Stemming"):
            if not line.strip(): continue
            data = json.loads(line)
            total += 1
            
            item_id = data['id']
            raw_output = data['raw_model_outputs'][0]
            true_letter = data['true_letter'].upper()
            old_pred = data['predicted_letters'][0]
            
            choices = manifest_dict.get(item_id, [])
            new_pred = final_sentence_parser(raw_output, choices)
            
            if old_pred is None and new_pred is not None:
                recovered_nulls += 1
            if new_pred is None:
                nulls += 1
            
            data['predicted_letters'] = [new_pred]
            data['accuracy'] = 1.0 if new_pred == true_letter else 0.0
            scores.append(data['accuracy'])
            
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"\n" + "="*40)
    print(f"Total Items         : {total}")
    print(f"Remaining Nulls     : {nulls}")
    print(f"New Mean Accuracy   : {np.mean(scores)*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # Update these paths to point to your files
    # names =["animal","language","emotion","gender"]
    names =["mmar","mmau"]
    for name in names:
        print(f"\n🔧 Processing dataset: {name.upper()}")
        # MANIFEST_FILE = f"pooneh_version/data/sakura/{name}/{name}_manifest_new.jsonl" 
        # RESULTS_FILE = f"pooneh_version/output/af3/{name}/baseline_{name}_REAS.jsonl"
        # OUTPUT_FILE = f"pooneh_version/output/af3/{name}/baseline_{name}_REAS_fixed.jsonl"
        MANIFEST_FILE = f"pooneh_version/data/{name}/{name}_manifest_json_new.jsonl" 
        RESULTS_FILE = f"pooneh_version/output/af3/{name}/baseline_{name}_REAS.jsonl"
        OUTPUT_FILE = f"pooneh_version/output/af3/{name}/baseline_{name}_REAS_fixed.jsonl"
        OUTPUT_FILE = f"pooneh_version/output/af3/{name}/baseline_{name}_REAS_fixed.jsonl"
        
        update_with_smart_stemming(MANIFEST_FILE, RESULTS_FILE, OUTPUT_FILE)


