# core/audio_flamingo_utils.py

import sys
import os
import torch
import nltk
import re
from typing import Tuple, List, Dict

# --- NEW, ROBUST PATH SETUP (Your Solution) ---
# As per our pragmatic approach, we will not use 'pip install -e'.
# Instead, we directly and explicitly add the path to the model's custom
# source code to Python's system path. This makes the 'llava' library
# importable for the duration of our script's execution.

# 1. Define the absolute path to the cloned code repository.
_AUDIO_FLAMINGO_CODE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio-flamingo-code'))

# 2. Check if the path exists to provide a clear error message.
if not os.path.exists(_AUDIO_FLAMINGO_CODE_PATH):
    raise FileNotFoundError(
        f"The directory 'audio-flamingo-code' was not found at the expected path: {_AUDIO_FLAMINGO_CODE_PATH}"
    )

# 3. Add this path to the list of places Python looks for libraries.
if _AUDIO_FLAMINGO_CODE_PATH not in sys.path:
    sys.path.append(_AUDIO_FLAMINGO_CODE_PATH)
    print(f"INFO: Temporarily added '{_AUDIO_FLAMINGO_CODE_PATH}' to Python path for llava import.")

# 4. NOW, we can import 'llava' directly and with confidence.
try:
    import llava
    from llava import conversation as clib
    from llava.media import Sound
    from peft import PeftModel
    from transformers import GenerationConfig
except ImportError as e:
    print(f"FATAL: Failed to import 'llava' library. Check the 'audio-flamingo-code' directory. Error: {e}")
    sys.exit(1)
# --- END OF NEW SETUP ---


def load_model_and_tokenizer(model_path: str) -> Tuple[object, object]:
    """
    Loads the Audio Flamingo model and its necessary components.
    """
    # This step reads the model's config and sets up the correct
    # chat template (e.g., for Llama-3, Qwen, etc.) automatically.
    clib.auto_set_conversation_mode(model_path)

    print("Loading base Audio Flamingo model...")
    model = llava.load(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # For our reasoning-focused research, we always want to use the best-performing
    # version of the model. The 'stage35' PEFT adapter is the "thinking mode".
    print("Loading 'thinking mode' PEFT adapter from 'stage35'...")
    think_model_path = os.path.join(model_path, 'stage35')
    if not os.path.exists(think_model_path):
        raise FileNotFoundError(
            f"The 'thinking mode' PEFT adapter was not found at the expected path: {think_model_path}"
        )
    model = PeftModel.from_pretrained(
        model,
        think_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # The llava model object is a unified class that handles all processing,
    # so we return it twice to match our framework's (model, processor) signature.
    print("Audio Flamingo model loaded successfully.")
    return model, model


def _convert_messages_to_flamingo_prompt(messages: List[Dict[str, str]]) -> str:
    """
    A helper function to convert our framework's standard 'messages' format
    into the single, flat string that Audio Flamingo expects.
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
    Runs multi-modal (audio + text) inference using the Audio Flamingo model.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    text_prompt = _convert_messages_to_flamingo_prompt(messages)
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
    Runs text-only inference using the Audio Flamingo model.
    """
    text_prompt = _convert_messages_to_flamingo_prompt(messages)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature, top_p=top_p,
    )
    response = model.generate_content(text_prompt, generation_config=generation_config)
    return response


def sanitize_cot(cot_text: str) -> str:
    """ Model-agnostic function to remove the final sentence from a string. """
    if not cot_text: return ""
    sentences = nltk.sent_tokenize(cot_text)
    if len(sentences) > 1:
        return " ".join(sentences[:-1])
    else:
        return ""


def parse_answer(text: str) -> str | None:
    """ Model-agnostic function to parse text to find a multiple-choice answer. """
    if not text: return None
    cleaned_text = text.strip()
    match = re.search(r'\(([A-Z])\)', cleaned_text)
    if match: return match.group(1)
    match = re.search(r'^([A-Z])\)$', cleaned_text)
    if match: return match.group(1)
    match = re.search(r'^\(([A-Z])$', cleaned_text)
    if match: return match.group(1)
    if len(cleaned_text) == 1 and 'A' <= cleaned_text <= 'Z':
        return cleaned_text
    refusal_keywords = ["cannot be determined", "none of the choices", "ambiguous", "not enough information", "no definitive answer"]
    if any(keyword in cleaned_text.lower() for keyword in refusal_keywords):
        return "REFUSAL"
    return None


def format_choices_for_prompt(choices: List[str]) -> str:
    """ Model-agnostic function to format a list of choices into a string. """
    if not choices: return ""
    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        formatted_choices.append(f"({letter}) {choice}")
    return "\n".join(formatted_choices)