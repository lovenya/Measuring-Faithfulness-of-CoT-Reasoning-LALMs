import json
import re
import numpy as np
from tqdm import tqdm

# Words to ignore when matching the model's text against the choices
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", 
             "or", "in", "on", "it", "this", "that", "for", "with", "as", 
             "by", "at", "but", "not", "be", "about", "which", "they", "i", 
             "you", "he", "she", "we"}

def get_meaningful_words(text):
    """Strips punctuation and returns a set of lowercase keywords."""
    if not text: return set()
    clean = re.sub(r'[^\w\s]', '', str(text)).strip().lower()
    return set(clean.split()) - STOPWORDS

def extract_semantic_choice(raw_text, choices_list):
    if not raw_text: return None
    cleaned = raw_text.strip()
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    # ---------------------------------------------------------
    # 1. STRICT FORMATTING MATCH (Parentheses & Prefixes)
    # ---------------------------------------------------------
    end_chunk = cleaned[-300:] # Scan the last 300 chars
    paren_matches = list(re.finditer(r'\(([A-J])\)', end_chunk, re.IGNORECASE))
    if paren_matches:
        return paren_matches[-1].group(1).upper()
        
    prefix_match = list(re.finditer(r'(?:option|choice|answer|answer\s*is|is)\s*[:*]*\s*([A-J])\b', end_chunk, re.IGNORECASE))
    if prefix_match:
        return prefix_match[-1].group(1).upper()

    # ---------------------------------------------------------
    # 2. SEMANTIC WORD OVERLAP (Fixes MMAR paraphrasing)
    # ---------------------------------------------------------
    if choices_list:
        # Isolate the last few sentences (the conclusion)
        sentences = [s.strip() for s in re.split(r'[.!?\n]', cleaned) if s.strip()]
        conclusion_text = " ".join(sentences[-3:]) if len(sentences) >= 3 else cleaned
        
        target_words = get_meaningful_words(conclusion_text)
        
        best_letter = None
        max_score = 0.0
        
        for i, opt in enumerate(choices_list):
            opt_words = get_meaningful_words(opt)
            if not opt_words: continue
            
            # Count how many meaningful words from the option appear in the conclusion
            overlap = opt_words.intersection(target_words)
            
            # Scoring: Absolute overlap + the percentage of the option matched
            score = len(overlap) + (len(overlap) / len(opt_words))
            
            # Require at least 1 meaningful word to overlap
            if score > max_score and len(overlap) > 0:
                max_score = score
                best_letter = letters[i]
        
        if best_letter:
            return best_letter

    # ---------------------------------------------------------
    # 3. STANDALONE LETTER FALLBACK
    # ---------------------------------------------------------
    standalone = re.search(r'\b([A-J])\b[^\w]*$', cleaned[-30:], re.IGNORECASE)
    if standalone and standalone.group(1).upper() not in ['A', 'I']:
        return standalone.group(1).upper()

    return None

def process_semantic_file(manifest_path, results_path, output_path):
    manifest_dict = {}
    print(f"Loading manifest from: {manifest_path}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            manifest_dict[item['id']] = item.get('choices_list', [])

    scores = []
    total = 0
    nulls = 0
    recovered_nulls = 0

    print(f"Reparsing results from: {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Applying Semantic Parser"):
            if not line.strip(): continue
            data = json.loads(line)
            total += 1
            
            item_id = data['id']
            raw_output = data['raw_model_outputs'][0]
            true_letter = data['true_letter'].upper()
            
            # Safely get old prediction
            predicted_list = data.get('predicted_letters', [None])
            old_pred = predicted_list[0] if len(predicted_list) > 0 else None
            
            choices = manifest_dict.get(item_id, [])
            new_pred = extract_semantic_choice(raw_output, choices)
            
            if old_pred is None and new_pred is not None:
                recovered_nulls += 1
            if new_pred is None:
                nulls += 1
            
            data['predicted_letters'] = [new_pred]
            data['accuracy'] = 1.0 if new_pred == true_letter else 0.0
            scores.append(data['accuracy'])
            
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"\n" + "="*40)
    print(f"🧠 SEMANTIC OVERLAP REPAIR RESULTS")
    print(f"Total Items         : {total}")
    print(f"Nulls Recovered     : {recovered_nulls}")
    print(f"Remaining Nulls     : {nulls}")
    print(f"New Mean Accuracy   : {np.mean(scores)*100:.2f}%")
    print(f"Saved Fixed File to : {output_path}")
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
        
        process_semantic_file(MANIFEST_FILE, RESULTS_FILE, OUTPUT_FILE)


