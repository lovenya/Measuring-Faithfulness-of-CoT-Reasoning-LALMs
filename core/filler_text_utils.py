# core/filler_text_utils.py

import random
from core.lalm_utils import run_inference, parse_answer

def create_word_level_masked_cot(cot_text: str, percentile: int, mode: str) -> str:
    """
    Creates a modified CoT by replacing a percentage of WORDS with filler text.

    Args:
        cot_text (str): The sanitized Chain-of-Thought.
        percentile (int): The percentage of words to replace (0-100).
        mode (str): The method of replacement ('start', 'end', or 'random').

    Returns:
        str: The CoT with words replaced by '...'.
    """
    words = cot_text.split()
    total_words = len(words)
    if total_words == 0:
        return ""

    num_to_replace = int((percentile / 100) * total_words)

    if mode == 'start':
        # Replace the first N words
        new_words = ["..."] * num_to_replace + words[num_to_replace:]
    elif mode == 'end':
        # Replace the last N words
        new_words = words[:-num_to_replace] + ["..."] * num_to_replace if num_to_replace > 0 else words
    elif mode == 'random':
        # Replace a random sample of N words
        indices_to_replace = set(random.sample(range(total_words), num_to_replace))
        new_words = [word if i not in indices_to_replace else "..." for i, word in enumerate(words)]
    else:
        # Should not happen, but good practice to handle
        return cot_text

    return " ".join(new_words)

def create_filler_for_text(processor, text_to_replace: str) -> str:
    """Helper function to create a filler string of the same token length as the input text."""
    target_token_length = len(processor.tokenizer.encode(text_to_replace, add_special_tokens=False))
    if target_token_length == 0:
        return ""
    
    filler_unit = "... "
    filler_text = filler_unit * int(target_token_length / 1.5)
    while len(processor.tokenizer.encode(filler_text, add_special_tokens=False)) < target_token_length:
        filler_text += filler_unit
    filler_text_tokens = processor.tokenizer.encode(filler_text, add_special_tokens=False)[:target_token_length]
    return processor.tokenizer.decode(filler_text_tokens, skip_special_tokens=True)

def run_filler_trial(model, processor, question: str, choices: str, audio_path: str, modified_cot: str) -> dict:
    """Runs a single trial with a modified CoT and returns a self-documenting result."""
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": modified_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
    ]
    final_answer_text = run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=10, do_sample=False
    )
    
    return {
        "predicted_choice": parse_answer(final_answer_text),
        "final_answer_raw": final_answer_text,
        "final_prompt_messages": final_answer_prompt_messages
    }