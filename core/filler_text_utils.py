# core/filler_text_utils.py

from core.lalm_utils import run_inference, parse_answer

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