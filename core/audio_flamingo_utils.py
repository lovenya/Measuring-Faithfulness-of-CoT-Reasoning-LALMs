# core/audio_flamingo_utils.py

"""
This file is the specific 'driver' or 'utility module' for the Audio Flamingo model.

Its purpose is to act as an adapter. It takes the standard, model-agnostic commands
from our experiment scripts and translates them into the unique, specific API calls
that the Audio Flamingo model requires. This design is what allows our main
experiment code to remain clean and unaware of the specific model it's working with.
"""

import sys
import os
import torch
import nltk
import re
from typing import Tuple, List, Dict

# We import our main config with an alias to avoid any confusion.
import config as framework_config

# --- Environment Setup for Custom Flamingo Code ---
# This block handles the setup required to import Flamingo's custom source code.
_AUDIO_FLAMINGO_CODE_PATH = framework_config.MODEL_PATHS['flamingo_code']
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


def load_model_and_tokenizer(model_path: str) -> Tuple[object, object, object]:
    """
    Loads the Audio Flamingo model and returns it in the (model, processor, tokenizer)
    format required by our framework's "contract".
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
    
    # --- THE CRITICAL FIX ---
    # The Flamingo 'llava' object is a unified class that handles all three roles.
    # To fulfill our framework's contract, we return the same object three times.
    # This makes the model compatible with our architecture without any downstream changes.
    processor = model
    tokenizer = model
    return model, processor, tokenizer
    # --- END OF FIX ---


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
    
    # This is the Flamingo-specific logic to encourage step-by-step reasoning.
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
    Handles text-only tasks by providing a dummy silent audio input.
    """
    silent_audio_path = framework_config.SILENT_AUDIO_PATH
    if not os.path.exists(silent_audio_path):
        raise FileNotFoundError(f"Silent audio file not found! Expected at: {silent_audio_path}")

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
    """ A universal, robust, and case-insensitive parser for all experiments. """
    if not text: return None
    cleaned_text = text.strip().strip('.').strip()

    # We check for patterns in a specific order, from most to least specific.
    # The character class [a-zA-Z] makes the regex case-insensitive.
    
    # Priority 1: The most robust pattern, `(X)`, searched anywhere.
    # This handles "The answer is (A)." etc.
    match = re.search(r'\(([a-zA-Z])\)', cleaned_text)
    if match:
        return match.group(1)

    # Priority 2: Strict check for the entire string `X)` having anywhere.
    match = re.search(r'([a-zA-Z])\)', cleaned_text)
    if match:
        return match.group(1)

    # Priority 3: Strict check for the string having `(X` anywhere.
    match = re.search(r'\(([a-zA-Z])', cleaned_text)
    if match:
        return match.group(1)

    # Priority 4: The most minimal case, a single letter.
    # This must check the length to avoid matching 'A' in "A good answer...".
    if len(cleaned_text) == 1 and cleaned_text.isalpha():
        return cleaned_text.upper()
        
    # If no choice is found, we check for a refusal to answer.
    refusal_keywords = ["cannot be determined", "none of the choices", "ambiguous", "not enough information", "no definitive answer"]
    if any(keyword in cleaned_text.lower() for keyword in refusal_keywords):
        return "REFUSAL"
        
    return None


def format_choices_for_prompt(choices: List[str]) -> str:
    """
    Model-agnostic function to format a list of choices into a string.
    """
    if not choices: return ""
    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        formatted_choices.append(f"({letter}) {choice}")
    return "\n".join(formatted_choices)