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

# --- Environment Setup for Custom SALMONN Code ---
# This block adds the SALMONN source code to Python's path, making it importable.
_SALMONN_CODE_PATH = os.path.abspath(config.MODEL_PATHS['salmonn_code'])
if not os.path.exists(_SALMONN_CODE_PATH):
    raise FileNotFoundError(f"SALMONN source code not found at: {_SALMONN_CODE_PATH}")
if _SALMONN_CODE_PATH not in sys.path:
    sys.path.insert(0, _SALMONN_CODE_PATH)
    print(f"INFO: Temporarily added '{_SALMONN_CODE_PATH}' to the start of Python's import path.")

try:
    from models.salmonn import SALMONN
    from transformers import WhisperFeatureExtractor
    from utils import prepare_one_sample
except ImportError as e:
    print(f"FATAL: Failed to import from SALMONN source code. Error: {e}")
    sys.exit(1)


def load_model_and_tokenizer(model_path: str) -> Tuple[object, object]:
    """
    Assembles the SALMONN model from its components and loads the feature extractor.
    """
    print("Assembling SALMONN model from components...")
    
    # 1. Load the default YAML config from the SALMONN source code.
    default_config_path = os.path.join(_SALMONN_CODE_PATH, 'configs/config.yaml')
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"SALMONN default config not found at: {default_config_path}")
    
    with open(default_config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. Programmatically OVERWRITE the paths in the config with our own from config.py.
    cfg['model']['llama_path'] = os.path.abspath(config.MODEL_PATHS['llama_path'])
    cfg['model']['whisper_path'] = os.path.abspath(config.MODEL_PATHS['whisper_path'])
    cfg['model']['beats_path'] = os.path.abspath(config.MODEL_PATHS['beats_path'])
    cfg['model']['ckpt'] = os.path.abspath(config.MODEL_PATHS['ckpt'])
    
    # 3. Instantiate the main SALMONN model with our modified config.
    model = SALMONN.from_config(cfg=cfg['model'])
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 4. Load the WhisperFeatureExtractor, which is the correct processor for this model.
    print("Loading Whisper feature extractor...")
    processor = WhisperFeatureExtractor.from_pretrained(config.MODEL_PATHS['whisper_path'])
    
    # 5. CRITICAL: Make the processor object conform to our "contract".
    # We dynamically attach the text tokenizer to the processor object so that our
    # experiment scripts can find it at 'processor.tokenizer' without needing to change.
    processor.tokenizer = model.llama_tokenizer

    print("SALMONN model and processor loaded successfully.")
    return model, processor


def _convert_messages_to_salmonn_prompt(messages: List[Dict[str, str]], prompt_template: str, is_audio_present: bool) -> List[str]:
    """
    A helper "translator" that correctly formats our 'messages' list into the
    precise prompt format that SALMONN expects.
    """
    full_text_prompt = ""
    # We flatten our messages list into a single string, preserving the turn structure.
    for msg in messages:
        content = msg["content"].replace("audio\n\n", "").strip()
        full_text_prompt += f"{content}\n"
    full_text_prompt = full_text_prompt.strip()

    audio_placeholder = "<Speech><SpeechHere></Speech> "
    
    if is_audio_present:
        final_prompt_content = audio_placeholder + full_text_prompt
    else:
        final_prompt_content = full_text_prompt

    # We use the template from the model's own config file.
    return [prompt_template.format(final_prompt_content)]


def run_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    audio_path: str, max_new_tokens: int, do_sample: bool,
    temperature: float, top_p: float
) -> str:
    """ Gatekeeper for all multi-modal interactions with the SALMONN model. """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # 1. Prepare the audio using the official 'prepare_one_sample' utility.
    samples = prepare_one_sample(audio_path, processor, cuda_enabled=torch.cuda.is_available())

    # 2. Prepare the text prompt.
    prompt_template = model.cfg.prompt_template
    prompts = _convert_messages_to_salmonn_prompt(messages, prompt_template, is_audio_present=True)

    # 3. Prepare generation config.
    generate_config = {
        "max_length": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
    }

    # 4. Call generate with the correct keyword arguments.
    with torch.cuda.amp.autocast(dtype=torch.float16):
        response = model.generate(samples, generate_config, prompts=prompts)
    return response[0]


def run_text_only_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    max_new_tokens: int, do_sample: bool, temperature: float, top_p: float
) -> str:
    """ Handles text-only tasks using our robust "silent audio" methodology. """
    silent_audio_path = config.SILENT_AUDIO_PATH
    if not os.path.exists(silent_audio_path):
        raise FileNotFoundError(f"Silent audio file not found! Expected at: {silent_audio_path}")

    # 1. Prepare a silent audio sample.
    samples = prepare_one_sample(silent_audio_path, processor, cuda_enabled=torch.cuda.is_available())

    # 2. Prepare the text prompt, explicitly stating NO audio is present.
    prompt_template = model.cfg.prompt_template
    prompts = _convert_messages_to_salmonn_prompt(messages, prompt_template, is_audio_present=False)

    # 3. Prepare generation config.
    generate_config = {
        "max_length": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
    }

    # 4. Call generate.
    with torch.cuda.amp.autocast(dtype=torch.float16):
        response = model.generate(samples, generate_config, prompts=prompts)
    return response[0]


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
    if not choices: return ""
    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        formatted_choices.append(f"({letter}) {choice}")
    return "\n".join(formatted_choices)