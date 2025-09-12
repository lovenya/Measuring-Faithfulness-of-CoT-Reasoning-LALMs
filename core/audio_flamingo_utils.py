# core/audio_flamingo_utils.py

"""
This file is the specific 'driver' or 'utility module' for the Audio Flamingo model.

Its purpose is to act as an adapter. It takes the standard, model-agnostic commands
from our experiment scripts and translates them into the unique, specific API calls
that the Audio Flamingo model requires. This design is what allows our main
experiment code (like baseline.py) to remain clean and unaware of the specific
model it's working with.
"""

import sys
import os
import torch
import nltk
import re
from typing import Tuple, List, Dict

# --- Environment Setup for Custom Flamingo Code ---
# Audio Flamingo uses custom source code that isn't installed via pip. To make it
# work, we need to tell Python where to find it. This block handles that setup.

# 1. We define the absolute path to the 'audio-flamingo-code' directory.
#    This makes our script robust, no matter where it's run from.
_AUDIO_FLAMINGO_CODE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio-flamingo-code'))

# 2. We check if the directory actually exists to provide a clear error if it's missing.
if not os.path.exists(_AUDIO_FLAMINGO_CODE_PATH):
    raise FileNotFoundError(
        f"The directory 'audio-flamingo-code' was not found at the expected path: {_AUDIO_FLAMINGO_CODE_PATH}"
    )

# 3. We add this path to the list of places Python looks for libraries.
#    This is what makes 'import llava' possible.
if _AUDIO_FLAMINGO_CODE_PATH not in sys.path:
    sys.path.append(_AUDIO_FLAMINGO_CODE_PATH)
    print(f"INFO: Temporarily added '{_AUDIO_FLAMINGO_CODE_PATH}' to Python path for llava import.")

# 4. Now, with the path set up, we can safely import the necessary libraries.
try:
    import llava
    from llava import conversation as clib
    from llava.media import Sound
    from peft import PeftModel
    from transformers import GenerationConfig
except ImportError as e:
    print(f"FATAL: Failed to import 'llava' library. Check the 'audio-flamingo-code' directory. Error: {e}")
    sys.exit(1)


def load_model_and_tokenizer(model_path: str) -> Tuple[object, object]:
    """
    Loads the Audio Flamingo model and prepares it for our reasoning experiments.
    """
    # This utility from the Flamingo library automatically sets the correct
    # chat template based on the underlying language model (e.g., Llama-3).
    clib.auto_set_conversation_mode(model_path)

    print("Loading base Audio Flamingo model...")
    model = llava.load(model_path, torch_dtype=torch.float16, device_map="auto")

    # For our faithfulness research, we need the model's best reasoning capabilities.
    # The 'stage35' PEFT adapter is what enables Flamingo's "thinking mode".
    # We load this adapter on top of the base model every time.
    print("Loading 'thinking mode' PEFT adapter from 'stage35'...")
    think_model_path = os.path.join(model_path, 'stage35')
    if not os.path.exists(think_model_path):
        raise FileNotFoundError(f"The 'thinking mode' PEFT adapter was not found at: {think_model_path}")
    model = PeftModel.from_pretrained(model, think_model_path, device_map="auto", torch_dtype=torch.float16)
    
    # The Flamingo 'llava' object is a unified class that handles both model inference
    # and data processing. To fit our framework's (model, processor) API, we simply
    # return the same object twice.
    print("Audio Flamingo model loaded successfully.")
    return model, model


def _convert_messages_to_flamingo_prompt(messages: List[Dict[str, str]]) -> str:
    """
    A helper function that acts as a "translator." It converts our framework's
    standard, structured 'messages' list into the single, flat string that
    Audio Flamingo's API expects.
    """
    full_prompt_text = ""
    for msg in messages:
        # We find the user message that contains our 'audio' placeholder...
        if "audio\n\n" in msg.get("content", ""):
            # ...and we remove the placeholder, leaving just the question and choices.
            content = msg["content"].replace("audio\n\n", "").strip()
            full_prompt_text += content + "\n"
        else:
            # For all other messages (like the assistant's "Let's think..."),
            # we just append their content directly.
            full_prompt_text += msg.get("content", "").strip() + "\n"
    
    return full_prompt_text.strip()


def run_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    audio_path: str, max_new_tokens: int, do_sample: bool,
    temperature: float, top_p: float
) -> str:
    """
    This is the gatekeeper for all multi-modal interactions with the Flamingo model.
    It handles the model-specific logic for running an audio-plus-text inference.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # First, we translate our standard message format into Flamingo's format.
    text_prompt = _convert_messages_to_flamingo_prompt(messages)

    # --- Flamingo-Specific Prompt Injection ---
    # This is where we handle Flamingo's unique requirement for eliciting a CoT.
    # If the prompt is a request for a reasoning chain, we append the special
    # suffix. This keeps our main experiment scripts (like baseline.py) completely
    # model-agnostic and clean.
    if "Let's think step by step:" in text_prompt:
        user_question_part = text_prompt.split("\nLet's think step by step:")[0]
        user_question_part += "\nPlease think and reason about the input audio before you respond."
        text_prompt = user_question_part + "\nLet's think step by step:"

    # We construct the multi-modal prompt list that Flamingo's API requires.
    prompt_list = [Sound(audio_path), text_prompt]

    # All generation parameters must be bundled into this GenerationConfig object.
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature, top_p=top_p,
    )

    response = model.generate_content(prompt_list, generation_config=generation_config)
    return response


def run_text_only_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    max_new_tokens: int, do_sample: bool, temperature: float, top_p: float
) -> str:
    """
    Handles text-only tasks like paraphrasing or creating mistakes.
    """
    text_prompt = _convert_messages_to_flamingo_prompt(messages)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature, top_p=top_p,
    )
    # For text-only tasks, the prompt is just the string itself.
    response = model.generate_content(text_prompt, generation_config=generation_config)
    return response


def sanitize_cot(cot_text: str) -> str:
    """
    A model-agnostic utility to remove the final sentence from a CoT.
    This is a crucial step to prevent the model from "cheating" by reading a
    "spoiler" sentence like "Therefore, the answer is (A)."
    """
    if not cot_text: return ""
    sentences = nltk.sent_tokenize(cot_text)
    # If there's more than one sentence, we return all but the last one.
    # If there's only one, we return an empty string, as it might be the spoiler.
    if len(sentences) > 1:
        return " ".join(sentences[:-1])
    else:
        return ""


def parse_answer(text: str) -> str | None:
    """
    A model-agnostic utility to find the final letter choice in the model's output.
    This version includes all the robust patterns we developed previously.
    """
    if not text: return None
    cleaned_text = text.strip().strip('.').strip()

    # Pattern 1: (A) - It looks for '(X)' anywhere in the string.
    # This is much more robust to the model's conversational tendencies.
    match = re.search(r'\(([A-Z])\)', cleaned_text)
    if match: return match.group(1)

    # Pattern 2: A) - Handles cases where the opening parenthesis is missing.
    match = re.search(r'^([A-Z])\)$', cleaned_text)
    if match: return match.group(1)

    # Pattern 3: (A - Handles cases where the closing parenthesis is missing.
    match = re.search(r'^\(([A-Z])$', cleaned_text)
    if match: return match.group(1)

    # Pattern 4: "answer is A" - Verbose but clear.
    match = re.search(r'answer is\s+([A-Z])', cleaned_text, re.IGNORECASE)
    if match: return match.group(1)

    # Pattern 5: A - The most minimal case, if the entire output is just one letter.
    if len(cleaned_text) == 1 and 'A' <= cleaned_text <= 'Z':
        return cleaned_text
        
    # Flamingo-specific failure mode check.
    refusal_keywords = ["cannot be determined", "none of the choices",
                        "ambiguous", "not enough information", "no definitive answer"]
    if any(keyword in cleaned_text.lower() for keyword in refusal_keywords):
        return "REFUSAL"
        
    return None


def format_choices_for_prompt(choices: List[str]) -> str:
    """
    A model-agnostic utility to format a list of choices into a clean,
    lettered string for the prompt. e.g., ["cat", "dog"] -> "(A) cat\n(B) dog"
    """
    if not choices: return ""
    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        formatted_choices.append(f"({letter}) {choice}")
    return "\n".join(formatted_choices)