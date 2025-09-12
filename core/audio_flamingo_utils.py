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

# We import our config module to get access to global paths, like the silent audio file.
import config

# --- Environment Setup for Custom Flamingo Code ---
# This block handles the setup required to import Flamingo's custom source code.
_AUDIO_FLAMINGO_CODE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio-flamingo-code'))
if not os.path.exists(_AUDIO_FLAMINGO_CODE_PATH):
    raise FileNotFoundError(f"The directory 'audio-flamingo-code' was not found at: {_AUDIO_FLAMINGO_CODE_PATH}")
if _AUDIO_FLAMINGO_CODE_PATH not in sys.path:
    sys.path.append(_AUDIO_FLAMINGO_CODE_PATH)
    print(f"INFO: Temporarily added '{_AUDIO_FLAMINGO_CODE_PATH}' to Python path for llava import.")

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
    Loads the Audio Flamingo model and prepares it for our reasoning experiments by
    applying the 'thinking mode' PEFT adapter.
    """
    clib.auto_set_conversation_mode(model_path)
    print("Loading base Audio Flamingo model...")
    model = llava.load(model_path, torch_dtype=torch.float16, device_map="auto")
    print("Loading 'thinking mode' PEFT adapter from 'stage35'...")
    think_model_path = os.path.join(model_path, 'stage35')
    if not os.path.exists(think_model_path):
        raise FileNotFoundError(f"The 'thinking mode' PEFT adapter was not found at: {think_model_path}")
    model = PeftModel.from_pretrained(model, think_model_path, device_map="auto", torch_dtype=torch.float16)
    print("Audio Flamingo model loaded successfully.")
    return model, model


def _convert_messages_to_flamingo_prompt(messages: List[Dict[str, str]]) -> str:
    """
    A helper "translator" that converts our framework's standard 'messages' list
    into the single, flat string that Flamingo's API expects.
    """
    full_prompt_text = ""
    for msg in messages:
        if "audio\n\n" in msg.get("content", ""):
            content = msg["content"].replace("audio\n\n", "").strip()
            full_prompt_text += content + "\n"
        else:
            full_prompt_text += msg.get("content", "").strip() + "\n"
    return full_prompt_text.strip()


def run_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    audio_path: str, max_new_tokens: int, do_sample: bool,
    temperature: float, top_p: float
) -> str:
    """
    This is the gatekeeper for all multi-modal interactions with the Flamingo model.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
    text_prompt = _convert_messages_to_flamingo_prompt(messages)
    if "Let's think step by step:" in text_prompt:
        user_question_part = text_prompt.split("\nLet's think step by step:")[0]
        user_question_part += "\nPlease think and reason about the input audio before you respond."
        text_prompt = user_question_part + "\nLet's think step by step:"
    prompt_list = [Sound(audio_path), text_prompt]
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
    
    This function is a crucial part of our methodology. It calls the main
    inference function but provides a path to a silent audio file. This forces
    the model to perform the task using only the text prompt, ensuring a clean,
    scientifically pure test of its language capabilities.
    """
    # We fetch the path to our silent audio asset from the global config.
    silent_audio_path = config.SILENT_AUDIO_PATH
    if not os.path.exists(silent_audio_path):
        raise FileNotFoundError(f"Silent audio file not found! Expected at: {silent_audio_path}")

    # We call the standard run_inference function, but with the silent audio path.
    return run_inference(
        model, processor, messages,
        audio_path=silent_audio_path,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p
    )


def sanitize_cot(cot_text: str) -> str:
    """
    A model-agnostic utility to remove the final "spoiler" sentence from a CoT.
    """
    if not cot_text: return ""
    sentences = nltk.sent_tokenize(cot_text)
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

    # Pattern 1: (A) - Most specific, often at the end of a line.
    match = re.search(r'\(([A-Z])\)$', cleaned_text)
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
    return "\n".join(formatted_choices)```