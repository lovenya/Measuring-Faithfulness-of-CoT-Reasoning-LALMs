# core/salmonn_utils.py

import sys
import os
import torch
import nltk
import re
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
    # We import all necessary components based on the evidence from cli_inference.py
    from config import Config as SalmonnConfig
    from models.salmonn import SALMONN
    from transformers import WhisperFeatureExtractor
    from utils import prepare_one_sample
except ImportError as e:
    print(f"FATAL: Failed to import from SALMONN source code. Check the directory and dependencies. Error: {e}")
    sys.exit(1)


def load_model_and_tokenizer(model_path: str) -> Tuple[object, object]:
    """
    Assembles the SALMONN model from its components and loads the feature extractor.
    This implementation is based on the official cli_inference.py script.
    """
    print("Assembling SALMONN model from components...")
    
    # 1. Create a temporary, minimal config file in memory to load the model.
    # This is a robust way to use their API without relying on a physical file.
    temp_config_dict = {
        'model': {
            'llama_path': os.path.abspath(config.MODEL_PATHS['llama_path']),
            'whisper_path': os.path.abspath(config.MODEL_PATHS['whisper_path']),
            'beats_path': os.path.abspath(config.MODEL_PATHS['beats_path']),
            'ckpt': os.path.abspath(config.MODEL_PATHS['ckpt']),
            'prompt_template': "USER: {} ASSISTANT:", # A standard template
        },
        'generate': { # Default generation params
            'max_length': 512,
            'do_sample': True,
        }
    }
    
    # The SALMONN Config class expects to read from a file, so we create a dummy
    # argparse object to pass it the dictionary directly.
    class DummyArgs:
        def __init__(self, cfg_dict):
            self.cfg_path = None
            self.options = None
            self.config = cfg_dict

    cfg = SalmonnConfig(DummyArgs(temp_config_dict))

    # 2. Load the model using the correct .from_config() method.
    model = SALMONN.from_config(cfg.config.model)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 3. Load the correct WhisperFeatureExtractor.
    print("Loading Whisper feature extractor...")
    processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    
    # 4. Attach the tokenizer to the processor to conform to our "contract".
    processor.tokenizer = model.llama_tokenizer
    
    # 5. Store the config inside the model object for later use in inference.
    model.framework_config = cfg.config

    print("SALMONN model and processor loaded successfully.")
    return model, processor


def _convert_messages_to_salmonn_prompt(messages: List[Dict[str, str]], prompt_template: str, is_audio_present: bool) -> List[str]:
    """
    Correctly formats our 'messages' list into the prompt string SALMONN expects.
    """
    full_text_prompt = ""
    for msg in messages:
        content = msg["content"].replace("audio\n\n", "").strip()
        # A simple join is sufficient as the template handles roles.
        full_text_prompt += f"{content}\n"
    full_text_prompt = full_text_prompt.strip()

    audio_placeholder = "<Speech><SpeechHere></Speech> "
    
    if is_audio_present:
        final_prompt_content = audio_placeholder + full_text_prompt
    else:
        final_prompt_content = full_text_prompt

    # The API expects a list containing one formatted string.
    return [prompt_template.format(final_prompt_content)]


def run_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    audio_path: str, max_new_tokens: int, do_sample: bool,
    temperature: float, top_p: float
) -> str:
    """ Gatekeeper for all multi-modal interactions with the SALMONN model. """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # 1. Prepare audio using the official 'prepare_one_sample' utility.
    samples = prepare_one_sample(audio_path, processor, cuda_enabled=torch.cuda.is_available())

    # 2. Prepare the text prompt.
    prompt_template = model.framework_config.model.prompt_template
    prompts = _convert_messages_to_salmonn_prompt(messages, prompt_template, is_audio_present=True)

    # 3. Prepare generation config, overriding defaults with our parameters.
    generate_config = model.framework_config.generate
    generate_config['max_length'] = max_new_tokens
    generate_config['do_sample'] = do_sample
    generate_config['temperature'] = temperature
    generate_config['top_p'] = top_p

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
    prompt_template = model.framework_config.model.prompt_template
    prompts = _convert_messages_to_salmonn_prompt(messages, prompt_template, is_audio_present=False)

    # 3. Prepare generation config.
    generate_config = model.framework_config.generate
    generate_config['max_length'] = max_new_tokens
    generate_config['do_sample'] = do_sample
    generate_config['temperature'] = temperature
    generate_config['top_p'] = top_p

    # 4. Call generate.
    with torch.cuda.amp.autocast(dtype=torch.float16):
        response = model.generate(samples, generate_config, prompts=prompts)
        
    return response[0]


# --- Model-Agnostic Utility Functions ---
# These functions are part of our "contract" and are identical to the
# versions used for our other models. No changes are needed here.

def sanitize_cot(cot_text: str) -> str:
    if not cot_text: return ""
    sentences = nltk.sent_tokenize(cot_text)
    if len(sentences) > 1:
        return " ".join(sentences[:-1])
    else:
        return ""

def parse_answer(text: str) -> str | None:
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
    if not choices: return ""
    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        formatted_choices.append(f"({letter}) {choice}")
    return "\n".join(formatted_choices)