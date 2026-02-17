#!/usr/bin/env python3
"""
Evaluate JASCO Masking Experiment Results using Mistral as LLM Judge.

Adapted from the original JASCO eval.py (which uses LLaMA 3.1 70B).
We use our local Mistral Small 3 instead, via the existing mistral_utils module.

The LLM judge classifies each model response into:
- Audio-Oriented (A): prediction based primarily on audio sounds
- Speech-Oriented (S): prediction based primarily on spoken text
- Good (G): prediction correctly uses BOTH audio and speech
- Neither (N): prediction doesn't address speaker actions

And gives a rating score (0-2) for answer quality.

Reports:
- Best-mean score per sample
- Orientation percentages (Audio% / Speech% / Both%)

Usage (must use mistral_env):
    source mistral_env/bin/activate
    python analysis/evaluate_jasco.py --results-dir results/qwen/jasco_masking/

    # Or a single file:
    python analysis/evaluate_jasco.py --results-file results/qwen/jasco_masking/full/jasco_masking_qwen_jasco_full.jsonl
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mistral_utils import load_mistral_model, run_mistral_inference
import config

# --- Evaluation Prompts (from original JASCO eval.py) ---

GENERAL_PROMPT = """[Audio Sound]
{audio_sound}

[Spoken Text]
{spoken_text}

[Question]
{question}

[Audio-Oriented Prediction]
{audio_only_target}

[Speech-Oriented Prediction]
{speech_only_target}

[Reference Answer]
{reference}

[Reference Answer Key Words]
{keywords}

[Model Prediction]
{prediction}

[Task1]
I am using a model to predict what the speakers are possibly doing based on both the audio sound and the spoken text. I want you to help me rate a score for the model's prediction on the speaker's action given a list of information [Audio Sound, Spoken Text, Question, Reference Answer, Reference Answer Key Words, Model Prediction]
Criteria: Assess if the model's prediction of the speaker's action mirrors the speaker's action in the reference answer in terms of content, logic, and relevance. Also assess if the model's prediction contains the Reference Answer Key Words or similar meanings. Do not care about the verb tense or other useless details in the model's response, focus only on the parts that speaker's actions are mentionned and the keywords. Very important: if the response mentions only the audio sound and the spoken text but not create a prediction of the speaker's specific action, rate it direcly 0, an exemple prediction like this can be 'The audio clip contains the sound of [some sounds]. The speaker says [some spoken texts]''.
Score0: The speaker's action predicted is completely misaligned, providing incorrect or irrelevant information compared to the speaker's action in the reference or the inference from audio sound and spoken text is not logical or based on only one modality (audio or speech), or the reponse is too general such as 'talking to someone' or 'having conversation'
Score1: The speaker's action predicted aligns with the speaker's action in the reference generally but lacks detailed keywords, the predicted action is based on both audio sound and spoken text and is logical enough but not the most possible.
Score2: The speaker's action predicted is highly accurate, and matches the speaker's action in the reference perfectly, capturing its essence and detailed keywords. The prediction is derived from both audio sound and spoken text and is very logical and the most probable.

[Task2]
Evaluate if the model's prediction of the speaker's action is inferred from audio sound or from spoken text or from both. You need to follow the below steps:
1. The model's response may contain multiple information, an example is 'The audio clip contains the sound of [detected audio sound] the speaker says [transcribed spoken text], this suggest that they are [predicted speaker's action]'. You need to first extract different components from the model's response: Part1-audio sound detected(may not exist), Part2-spoken text transcribed (may not exist), and Part3-speaker's action predicted(may not exist). If predicted speaker's action does not exist, the result is directly 'Neither'.
2. If Part3 exists, align it with Part1 and Part2. Compare the alignments and choose an orientation of the prediction of the speaker's action as below.
Audio-Oriented: The predicted speaker's action is explicitly and strongly related to the audio sound.
Speech-Oriented: The predicted speaker's action is explicitly and strongly related to the spoken text or they have a significant overlap. 
Good: The predicted speaker's action is explicitly and strongly related to both the audio sound and the spoken text. Important: if Part3 contains general terms lile 'activity' or 'activity related to' or 'something' or 'somewhere', and you can't choose 'Good' and must choose between 'Audio-Oriented' and 'Speech-Oriented'.
Remember only to use the extracted predicted speaker's action for assessment make sure you see a STRONG correlation when you make decisions.

Your response should be formatted as follows:
Explanation1: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. 'The provided audio sound is [BBB], the provided spoken text is [CCC], the reference answer is [XXX], the reference keywords are [KKK], while the model's answer is [YYY]. I think ...')
Rating: (int)
Explanation2: (Provide a concise explanation of your choice among Audio-Oriented/Speech-Oriented/Good/Neither, remember to focus on the texts you see and don't imagine too much. 'The provided audio sound is [BBB] and the provided spoken text is [CCC]. The detected audio sound in the model's reponse is [PPP]. The transcribed spoken text in the model's reponse is [QQQ]. The predicted speaker's action in the model's reponse is [YYY], I think ...')
Orientation: Audio-Oriented/Speech-Oriented/Good/Neither
"""

POST_PROCESS_PROMPT = """[Model Explanation]
{explanation}

The input model's explanation explicates how it makes the decision among Audio-Oriented/Speech-Oriented/Good/Neither. Based on the explanation, guess what final choice the model makes. The output format should be 'Audio-Oriented/Speech-Oriented/Good/Neither.'"""


def extract_orientation(response: str, model=None) -> str:
    """Extract orientation from the LLM judge response."""
    if "Orientation: Neither" in response or "Orientation: \nNeither" in response:
        return "N"
    elif "Orientation: Audio-Oriented" in response or "Orientation: \nAudio-Oriented" in response:
        return "A"
    elif "Orientation: Speech-Oriented" in response or "Orientation: \nSpeech-Oriented" in response:
        return "S"
    elif "Orientation: Good" in response or "Orientation: \nGood" in response or "Orientation: Both" in response:
        return "G"
    else:
        # If not in desired format, try post-processing
        if model is not None:
            explanation_2 = response.split("Explanation2: ")[-1]
            messages = [
                {"role": "system", "content": "You are an NLP assistant"},
                {"role": "user", "content": POST_PROCESS_PROMPT.format(explanation=explanation_2)},
            ]
            post_output = run_mistral_inference(
                model, messages, max_new_tokens=100, do_sample=False
            )
            if "Neither" in post_output:
                return "N"
            elif "Audio-Oriented" in post_output:
                return "A"
            elif "Speech-Oriented" in post_output:
                return "S"
            elif "Good" in post_output:
                return "G"
        return "?"


def extract_rating(response: str) -> int:
    """Extract rating score (0-2) from the LLM judge response."""
    for line in response.split("\n"):
        if "Rating: " in line:
            if "2" in line:
                return 2
            elif "1" in line:
                return 1
            else:
                return 0
    return 0


def load_results(filepath: str) -> list:
    """Load results from a JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def evaluate_with_judge(results: list, model) -> list:
    """
    Run LLM judge evaluation on all results.
    Returns list of results with added judge_output, orientation, and rating fields.
    """
    evaluated = []
    total = len(results)
    
    for i, entry in enumerate(results):
        # Build the judge prompt
        keywords = entry.get('target_keywords', [])
        keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        
        prompt_filled = GENERAL_PROMPT.format(
            audio_sound=entry.get('audio_sound', ''),
            spoken_text=entry.get('spoken_text', ''),
            question=entry.get('prompt', ''),
            reference=entry.get('correct_answer', ''),
            audio_only_target=entry.get('audio_only_answer', ''),
            speech_only_target=entry.get('speech_only_answer', ''),
            prediction=entry.get('model_output', ''),
            keywords=keywords_str,
        )
        
        messages = [
            {"role": "system", "content": "You are an NLP assistant"},
            {"role": "user", "content": prompt_filled},
        ]
        
        # Run judge inference
        judge_output = run_mistral_inference(
            model, messages, max_new_tokens=1000, do_sample=False
        )
        
        orientation = extract_orientation(judge_output, model)
        rating = extract_rating(judge_output)
        
        # If orientation is Good, keep the rating; otherwise force 0
        effective_score = rating if orientation == "G" else 0
        
        entry_evaluated = {
            **entry,
            'judge_output': judge_output,
            'orientation': orientation,
            'rating': rating,
            'effective_score': effective_score,
        }
        evaluated.append(entry_evaluated)
        
        if (i + 1) % 50 == 0 or i == total - 1:
            logging.info(f"  Evaluated {i + 1}/{total} samples")
    
    return evaluated


def compute_stats(evaluated: list, variant_label: str = ""):
    """Compute and print JASCO evaluation statistics."""
    # Group by sample ID
    by_id = defaultdict(list)
    for entry in evaluated:
        by_id[entry['id']].append(entry)
    
    # Best-mean score: for each sample, take the best score across prompts
    best_scores = []
    for sample_id, entries in by_id.items():
        best_score = max(e['effective_score'] for e in entries)
        best_scores.append(best_score)
    
    best_mean = sum(best_scores) / len(best_scores) if best_scores else 0.0
    
    # Orientation percentages (excluding Neither and ?)
    orientations = [e['orientation'] for e in evaluated]
    a_count = orientations.count("A")
    s_count = orientations.count("S")
    g_count = orientations.count("G")
    n_count = orientations.count("N")
    q_count = orientations.count("?")
    
    valid_total = a_count + s_count + g_count
    
    label = f" [{variant_label}]" if variant_label else ""
    print(f"\n{'='*60}")
    print(f"JASCO Evaluation Results{label}")
    print(f"{'='*60}")
    print(f"  Total responses:     {len(evaluated)}")
    print(f"  Unique samples:      {len(by_id)}")
    print(f"  Best-mean score:     {best_mean:.3f}")
    print(f"")
    print(f"  Orientation breakdown:")
    print(f"    Audio-Oriented:    {a_count} ({a_count/len(orientations):.1%})")
    print(f"    Speech-Oriented:   {s_count} ({s_count/len(orientations):.1%})")
    print(f"    Good (Both):       {g_count} ({g_count/len(orientations):.1%})")
    print(f"    Neither:           {n_count} ({n_count/len(orientations):.1%})")
    if q_count:
        print(f"    Unknown:           {q_count} ({q_count/len(orientations):.1%})")
    
    if valid_total > 0:
        print(f"")
        print(f"  Normalized orientation (excluding Neither):")
        print(f"    Audio-Oriented %:  {a_count/valid_total:.1%}")
        print(f"    Speech-Oriented %: {s_count/valid_total:.1%}")
        print(f"    Both-Oriented %:   {g_count/valid_total:.1%}")
    print(f"{'='*60}")
    
    return {
        'best_mean': best_mean,
        'audio_pct': a_count / valid_total if valid_total else 0,
        'speech_pct': s_count / valid_total if valid_total else 0,
        'both_pct': g_count / valid_total if valid_total else 0,
        'neither_count': n_count,
        'total': len(evaluated),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate JASCO results with Mistral LLM judge')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--results-dir', type=str,
                       help='Directory containing JASCO result JSONL files (evaluates all)')
    group.add_argument('--results-file', type=str,
                       help='Single JSONL result file to evaluate')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save evaluated JSONL files (default: same as input)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to Mistral model weights (default: from config.py)')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Load Mistral judge model
    model_path = args.model_path or config.MODEL_PATHS.get('mistral_small_3')
    logging.info(f"Loading Mistral judge model from: {model_path}")
    model = load_mistral_model(model_path)
    
    # Collect result files
    if args.results_file:
        files = [Path(args.results_file)]
    else:
        result_dir = Path(args.results_dir)
        # Search recursively for JSONL files in subdirs (full/, audio_only/, speech_only/)
        files = sorted(result_dir.rglob('*.jsonl'))
    
    if not files:
        print("No result files found.")
        return
    
    all_stats = {}
    
    for filepath in files:
        logging.info(f"\n--- Processing: {filepath.name} ---")
        results = load_results(str(filepath))
        if not results:
            logging.info(f"  Skipping empty file: {filepath}")
            continue
        
        variant = results[0].get('variant', 'unknown')
        
        # Run LLM judge
        evaluated = evaluate_with_judge(results, model)
        
        # Save evaluated results
        out_dir = args.output_dir or str(filepath.parent)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filepath.stem + '_evaluated.jsonl')
        with open(out_path, 'w') as f:
            for entry in evaluated:
                f.write(json.dumps(entry) + '\n')
        logging.info(f"  Saved evaluated results to: {out_path}")
        
        # Compute and print stats
        stats = compute_stats(evaluated, variant_label=variant)
        all_stats[variant] = stats
    
    # Summary across variants
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print(f"COMPARISON ACROSS VARIANTS")
        print(f"{'='*60}")
        print(f"{'Variant':<15} {'Best-Mean':>10} {'Audio%':>8} {'Speech%':>9} {'Both%':>8}")
        print(f"{'-'*52}")
        for variant in ['full', 'audio_only', 'speech_only']:
            if variant in all_stats:
                s = all_stats[variant]
                print(f"{variant:<15} {s['best_mean']:>10.3f} {s['audio_pct']:>7.1%} {s['speech_pct']:>8.1%} {s['both_pct']:>7.1%}")
        print(f"{'-'*52}")


if __name__ == '__main__':
    main()
