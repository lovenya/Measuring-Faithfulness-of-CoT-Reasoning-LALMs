# core/filler_text_utils.py

"""
This is a shared utility module for all 'filler text' type experiments.

Its purpose is to provide centralized, model-agnostic functions for the core
tasks of these experiments: creating a modified reasoning chain and running a
trial with it. By centralizing this logic, we ensure that all three partial
filler experiments (start, end, and random) are perfectly consistent in their
methodology and can be easily updated in one place.
"""

import random
import nltk

# NOTE: We no longer import 'run_inference' or 'parse_answer' from a specific
# utils file. Instead, the 'model_utils' object will be passed in, making
# these functions fully model-agnostic.

def create_word_level_masked_cot(model_utils, cot_text: str, percentile: int, mode: str) -> str:
    """
    Creates a modified CoT by replacing a percentage of WORDS with filler text.
    This version is now model-agnostic.

    Args:
        model_utils: The dynamically loaded utility module for the current model.
        cot_text (str): The sanitized Chain-of-Thought.
        percentile (int): The percentage of words to replace (0-100).
        mode (str): The method of replacement ('start', 'end', or 'random').

    Returns:
        str: The CoT with words replaced by '...'.
    """
    # We use nltk.word_tokenize for a more robust split than a simple .split().
    # This is now accessed through the model_utils object to ensure consistency.
    words = nltk.word_tokenize(cot_text)
    total_words = len(words)
    if total_words == 0:
        return ""

    num_to_replace = int((percentile / 100) * total_words)

    if mode == 'start':
        # Replace the first N words with a single '...' placeholder.
        # This is a cleaner visual representation than N separate placeholders.
        if num_to_replace > 0:
            new_words = ["..."] + words[num_to_replace:]
        else:
            new_words = words
    elif mode == 'end':
        # Replace the last N words.
        if num_to_replace > 0:
            new_words = words[:-num_to_replace] + ["..."]
        else:
            new_words = words
    elif mode == 'random':
        # Replace a random sample of N words.
        indices_to_replace = set(random.sample(range(total_words), num_to_replace))
        new_words = [word if i not in indices_to_replace else "..." for i, word in enumerate(words)]
    else:
        # A safety fallback.
        return cot_text

    return " ".join(new_words)


def run_filler_trial(model, processor, model_utils, question: str, choices_formatted: str, audio_path: str, modified_cot: str) -> dict:
    """
    Runs a single trial with a modified CoT.
    This version is now fully model-agnostic.
    """
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices_formatted}"},
        {"role": "assistant", "content": modified_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else."}
    ]
    
    # We now correctly use the 'model_utils' object for all model interactions.
    # This is the key to making our experiments model-agnostic.
    final_answer_text = model_utils.run_inference(
        model, processor, final_answer_prompt_messages, audio_path, 
        max_new_tokens=10, do_sample=False, temperature=0.7, top_p=0.9
    )
    
    # We return a complete dictionary to ensure a robust data flow.
    return {
        "question": question,
        "choices": choices_formatted,
        "audio_path": audio_path,
        "predicted_choice": model_utils.parse_answer(final_answer_text),
        "final_answer_raw": final_answer_text,
        "final_prompt_messages": final_answer_prompt_messages
    }