# core/salmonn_utils.py

import sys
import os
import torch
import nltk
import re
import librosa
import yaml
from typing import Tuple, List, Dict

# We import our project's config to get the master paths.
import config



## --- THE FINAL, CORRECT ENVIRONMENT SETUP ---
# This block ensures that Python can find the custom SALMONN source code.
_SALMONN_CODE_PATH = os.path.abspath(config.MODEL_PATHS['salmonn_code'])
if not os.path.exists(_SALMONN_CODE_PATH):
    raise FileNotFoundError(f"SALMONN source code not found at: {_SALMONN_CODE_PATH}")

# We use sys.path.insert(0, ...) to add the SALMONN source directory to the
# VERY BEGINNING of Python's search path. This is the most robust method.
# It guarantees that when the script looks for 'models' or 'utils', it finds
# the ones inside the SALMONN code first, preventing any conflicts.
if _SALMONN_CODE_PATH not in sys.path:
    sys.path.insert(0, _SALMONN_CODE_PATH)
    print(f"INFO: Temporarily added '{_SALMONN_CODE_PATH}' to the start of Python's import path.")

try:
    # Now that the path is correctly set, these imports will succeed.
    from models.salmonn import SALMONN
    from utils.conversation import Conversation, SeparatorStyle
    from transformers import WhisperProcessor
except ImportError as e:
    print(f"FATAL: Failed to import from SALMONN source code. Check the directory and dependencies. Error: {e}")
    sys.exit(1)
# --- END OF THE FIX ---
    



def load_model_and_tokenizer(model_path: str) -> Tuple[object, object]:
    """
    Assembles the SALMONN model from its multiple components and loads the processor.
    """
    print("Assembling SALMONN model from components...")
    
    # 1. Load the default YAML config from the SALMONN source code.
    # This is more robust than creating our own config from scratch.
    default_config_path = os.path.join(_SALMONN_CODE_PATH, 'configs/config.yaml')
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"SALMONN default config not found at: {default_config_path}")
    
    with open(default_config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. Programmatically OVERWRITE the paths in the config with our own.
    # This is the core of the integration. We use our config.py as the single source of truth.
    cfg['model']['llama_path'] = os.path.abspath(config.MODEL_PATHS['llama_path'])
    cfg['model']['whisper_path'] = os.path.abspath(config.MODEL_PATHS['whisper_path'])
    cfg['model']['beats_path'] = os.path.abspath(config.MODEL_PATHS['beats_path'])
    cfg['model']['ckpt'] = os.path.abspath(config.MODEL_PATHS['ckpt'])
    
    # 3. Instantiate the main SALMONN model with our modified config.
    model = SALMONN(cfg=cfg['model'])
    model.eval()
    
    # 4. Load the WhisperProcessor, which is a separate but essential component.
    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(config.MODEL_PATHS['whisper_path'])

    # 5. CRITICAL: Make the processor object conform to our "contract".
    # Our experiment scripts expect to find the text tokenizer at 'processor.tokenizer'.
    # For SALMONN, the text tokenizer is part of the main model object. We dynamically
    # attach it to the processor here, so our experiment scripts don't need to change.
    processor.tokenizer = model.llama_tokenizer

    print("SALMONN model and processor loaded successfully.")
    return model, processor


def _convert_messages_to_salmonn_prompt(messages: List[Dict[str, str]]) -> str:
    """
    A helper "translator" that correctly converts our standard 'messages' list
    into the structured prompt string that SALMONN's Conversation utility expects.
    """
    # We use the 'vicuna_v1' conversation template, as it's the base LLM.
    conv = Conversation(
        system="",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )
    
    for i, msg in enumerate(messages):
        role = "USER" if msg["role"] == "user" else "ASSISTANT"
        content = msg["content"]
        
        # The special <Audio> token must be placed in the first user turn.
        if i == 0 and role == "USER":
            # We remove our 'audio\n\n' placeholder and insert SALMONN's placeholder.
            content = content.replace("audio\n\n", "").strip()
            conv.append_message(role, f"<Audio>\n{content}")
        else:
            conv.append_message(role, content)
            
    return conv.get_prompt()


def run_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    audio_path: str, max_new_tokens: int, do_sample: bool,
    temperature: float, top_p: float
) -> str:
    """ Gatekeeper for all multi-modal interactions with the SALMONN model. """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # 1. Pre-process the audio using the WhisperProcessor.
    raw_audio, sr = librosa.load(audio_path, sr=16000)
    audio_input = processor(raw_audio, sampling_rate=sr, return_tensors="pt").input_features

    # 2. Format the text prompt using our robust conversion helper.
    text_input = _convert_messages_to_salmonn_prompt(messages)

    # 3. Generate the response.
    device = next(model.parameters()).device
    audio_input = audio_input.to(device)

    # The model's custom generate method uses 'max_length', not 'max_new_tokens'.
    response = model.generate(
        audio_input=audio_input,
        text_input=[text_input], # The API expects a list of strings
        max_length=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    return response[0] # The API returns a list containing a single response string.


def run_text_only_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    max_new_tokens: int, do_sample: bool, temperature: float, top_p: float
) -> str:
    """ Handles text-only tasks by using our robust "silent audio" methodology. """
    silent_audio_path = config.SILENT_AUDIO_PATH
    if not os.path.exists(silent_audio_path):
        raise FileNotFoundError(f"Silent audio file not found! Expected at: {silent_audio_path}")

    # We simply call the main inference function, but with the silent audio path.
    return run_inference(
        model, processor, messages,
        audio_path=silent_audio_path,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p
    )


def sanitize_cot(cot_text: str) -> str:
    """ A model-agnostic utility to remove the final 'spoiler' sentence from a CoT. """
    if not cot_text: return ""
    sentences = nltk.sent_tokenize(cot_text)
    if len(sentences) > 1:
        return " ".join(sentences[:-1])
    else:
        return ""


def parse_answer(text: str) -> str | None:
    """ A model-agnostic utility to find the final letter choice in the model's output. """
    # ... (This function is our standard, robust parser and remains unchanged) ...
    if not text: return None
    cleaned_text = text.strip().strip('.').strip()
    match = re.search(r'\(([A-Z])\)$', cleaned_text)
    if match: return match.group(1)
    match = re.search(r'^([A-Z])\)$', cleaned_text)
    if match: return match.group(1)
    match = re.search(r'^\(([A-Z])$', cleaned_text)
    if match: return match.group(1)
    match = re.search(r'answer is\s+([A-Z])', cleaned_text, re.IGNORECASE)
    if match: return match.group(1)
    if len(cleaned_text) == 1 and 'A' <= cleaned_text <= 'Z':
        return cleaned_text
    refusal_keywords = ["cannot be determined", "none of the choices", "ambiguous", "not enough information", "no definitive answer"]
    if any(keyword in cleaned_text.lower() for keyword in refusal_keywords):
        return "REFUSAL"
    return None


def format_choices_for_prompt(choices: List[str]) -> str:
    """ A model-agnostic utility to format choices into a lettered string. """
    # ... (This function is our standard one and remains unchanged) ...
    if not choices: return ""
    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        formatted_choices.append(f"({letter}) {choice}")
    return "\n".join(formatted_choices)