# core/salmonn_utils.py

import sys
import os
import torch
import nltk
import re
import yaml
import importlib.util  # We need this for the manual import
from typing import Tuple, List, Dict
import librosa

# We import our main config with an alias to avoid any confusion.
import config as framework_config

# --- THE CRITICAL FIX: MANUAL, UNAMBIGUOUS IMPORT ---
# This block is the definitive solution to the 'config' name collision.

# 1. Define the absolute path to the SALMONN source code and its config file.
_SALMONN_CODE_PATH = framework_config.SALMONN_COMPONENT_PATHS['source_code']
_SALMONN_CONFIG_PATH = os.path.join(_SALMONN_CODE_PATH, 'config.py')

if not os.path.exists(_SALMONN_CONFIG_PATH):
    raise FileNotFoundError(f"SALMONN's config.py not found at: {_SALMONN_CONFIG_PATH}")

# 2. Use importlib to load the SALMONN config file as a module directly from its path.
#    This completely bypasses the standard import system and its caching,
#    guaranteeing we load the correct file.
spec = importlib.util.spec_from_file_location("salmonn_config_module", _SALMONN_CONFIG_PATH)
salmonn_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(salmonn_config_module)

# 3. Now we can safely get the 'Config' class from the module we just loaded.
SalmonnConfig = salmonn_config_module.Config

# 4. With the collision resolved, we can now safely add the main source code
#    directory to the path to allow for the other, non-conflicting imports.
if _SALMONN_CODE_PATH not in sys.path:
    sys.path.append(_SALMONN_CODE_PATH)
    print(f"INFO: Temporarily added '{_SALMONN_CODE_PATH}' to Python path.")

try:
    from models.salmonn import SALMONN
    from transformers import WhisperFeatureExtractor
    from utils import prepare_one_sample
except ImportError as e:
    print(f"FATAL: Failed to import SALMONN library components. Error: {e}")
    sys.exit(1)
# --- END OF FIX ---


def load_model_and_tokenizer(model_path: str) -> Tuple[object, object]:
    """
    Loads the complex, multi-component SALMONN model.
    """
    print("--- Loading SALMONN Model System (using native config loader) ---")
    
    config_path = os.path.join(framework_config.SALMONN_COMPONENT_PATHS['source_code'], 'configs/decode_config.yaml')
    mock_args = type('Args', (), {'cfg_path': config_path, 'options': None})()
    
    # We now use the 'SalmonnConfig' class we loaded manually.
    cfg = SalmonnConfig(mock_args)

    cfg.config.model.llama_path = framework_config.SALMONN_COMPONENT_PATHS['vicuna']
    cfg.config.model.whisper_path = framework_config.SALMONN_COMPONENT_PATHS['whisper']
    cfg.config.model.beats_path = framework_config.SALMONN_COMPONENT_PATHS['beats']
    cfg.config.model.ckpt = framework_config.SALMONN_COMPONENT_PATHS['salmonn_checkpoint']

    model = SALMONN.from_config(cfg.config.model)
    model.to("cuda")
    model.eval()

    processor = WhisperFeatureExtractor.from_pretrained(framework_config.SALMONN_COMPONENT_PATHS['whisper'])
    model.custom_config = cfg.config

    print("SALMONN model and processor loaded successfully.")
    return model, processor


def _convert_messages_to_salmonn_prompt(messages: List[Dict[str, str]], model_config: dict) -> str:
    """
    A helper to translate our standard 'messages' format into the specific,
    templated string that SALMONN's generate method expects.
    """
    full_text = ""
    for msg in messages:
        content = msg.get("content", "")
        if "audio\n\n" in content:
            content = content.replace("audio\n\n", "").strip()
        full_text += content + "\n"
    full_text = full_text.strip()

    prompt_template = model_config.model.prompt_template
    wrapped_text = "<Speech><SpeechHere></Speech> " + full_text
    
    return prompt_template.format(wrapped_text)


def run_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    audio_path: str, max_new_tokens: int, do_sample: bool,
    temperature: float, top_p: float
) -> str:
    """
    Runs multi-modal inference using the SALMONN model.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    resampled_wav, sr = librosa.load(audio_path, sr=16000)

    # 2. Use the WhisperFeatureExtractor (our 'processor') to create the spectrogram.
    #    This is the step that was missing.
    spectrogram = processor(resampled_wav, sampling_rate=sr, return_tensors="pt").input_features
    
    # 3. Construct the 'samples' dictionary exactly as the model expects.
    samples = {
        "spectrogram": spectrogram.to(model.device, dtype=torch.float32),
        "raw_wav": torch.from_numpy(resampled_wav).unsqueeze(0).to(model.device, dtype=torch.float32)
    }

    prompt = _convert_messages_to_salmonn_prompt(messages, model.custom_config)
    
    generate_cfg = model.custom_config.generate
    generate_cfg['max_new_tokens'] = max_new_tokens
    generate_cfg['do_sample'] = do_sample
    generate_cfg['temperature'] = temperature
    generate_cfg['top_p'] = top_p
    generate_cfg['num_beams'] = 1

    with torch.cuda.amp.autocast(dtype=torch.float32):
        result = model.generate(samples, generate_cfg, prompts=[prompt])
        
    
    return result[0]


def run_text_only_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    max_new_tokens: int, do_sample: bool, temperature: float, top_p: float
) -> str:
    """
    Handles text-only tasks by providing a dummy silent audio input.
    """
    silent_audio_path = framework_config.SILENT_AUDIO_PATH
    if not os.path.exists(silent_audio_path):
        raise FileNotFoundError(f"Silent audio file not found at '{silent_audio_path}'")
    
    return run_inference(
        model, processor, messages, silent_audio_path,
        max_new_tokens, do_sample, temperature, top_p
    )


def sanitize_cot(cot_text: str) -> str:
    """ Model-agnostic utility to remove the final sentence from a CoT. """
    if not cot_text: return ""
    sentences = nltk.sent_tokenize(cot_text)
    if len(sentences) > 1:
        return " ".join(sentences[:-1])
    else:
        return ""


def parse_answer(text: str) -> str | None:
    """ Model-agnostic utility to find the final letter choice in the model's output. """
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
    return None


def format_choices_for_prompt(choices: List[str]) -> str:
    """ Model-agnostic utility to format choices for the prompt. """
    if not choices: return ""
    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        formatted_choices.append(f"({letter}) {choice}")
    return "\n".join(formatted_choices)