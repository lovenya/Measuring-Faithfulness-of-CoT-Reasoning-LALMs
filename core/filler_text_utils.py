# core/filler_text_utils.py

"""
This is a shared utility module for all 'filler text' type experiments.
It provides centralized, model-agnostic functions for creating and running
trials with modified reasoning chains.
"""

import random
import nltk

def create_word_level_masked_cot(cot_text: str, percentile: int, mode: str) -> str:
    """
    Creates a modified CoT by replacing a percentage of WORDS with filler text.

    This function uses nltk.word_tokenize to ensure that we are operating on
    the word level, which is the established methodology for this experiment
    and is crucial for maintaining consistency with previously generated results.

    Args:
        cot_text (str): The sanitized Chain-of-Thought.
        percentile (int): The percentage of words to replace (0-100).
        mode (str): The method of replacement ('start', 'end', or 'random').

    Returns:
        str: The CoT with words replaced by '...'.
    """
    
    # We use nltk.word_tokenize for a robust, word-level split.
    words = nltk.word_tokenize(cot_text)
    
    total_words = len(words)
    if total_words == 0:
        return ""

    num_to_replace = int((percentile / 100) * total_words)

    if mode == 'start':
        if num_to_replace > 0:
            new_words = ["..."] + words[num_to_replace:]
        else:
            new_words = words
    elif mode == 'end':
        if num_to_replace > 0:
            new_words = words[:-num_to_replace] + ["..."]
        else:
            new_words = words
    elif mode == 'random':
        indices_to_replace = set(random.sample(range(total_words), num_to_replace))
        new_words = [word if i not in indices_to_replace else "..." for i, word in enumerate(words)]
    else:
        return cot_text

    # We simply join the list of words back into a string.
    return " ".join(new_words)


def run_filler_trial(model, processor, tokenizer, model_utils, question: str, choices_formatted: str, audio_path: str, modified_cot: str) -> dict:
    """
    Runs a single trial with a modified CoT.
    The function signature accepts the tokenizer for architectural consistency,
    even though it is not used in this specific helper.
    """
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices_formatted}"},
        {"role": "assistant", "content": modified_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else."}
    ]
    
    final_answer_text = model_utils.run_inference(
        model, processor, final_answer_prompt_messages, audio_path, 
        max_new_tokens=10, do_sample=False, temperature=0.7, top_p=0.9
    )
    
    return {
        "question": question,
        "choices": choices_formatted,
        "audio_path": audio_path,
        "predicted_choice": model_utils.parse_answer(final_answer_text),
        "final_answer_raw": final_answer_text,
        "final_prompt_messages": final_answer_prompt_messages
    }