# core/salmonn_utils.py

"""
This file is the specific 'driver' or 'utility module' for the SALMONN model.

Its purpose is to act as an adapter. It takes the standard, model-agnostic commands
from our experiment scripts and translates them into the unique, specific API calls
that the SALMONN model requires. This design is what allows our main
experiment code (like baseline.py) to remain clean and unaware of the specific
model it's working with.
"""

import sys
import os
import torch
import nltk
import re
import yaml
import importlib.util  # We need this for the manual import
from typing import Tuple, List, Dict
import librosa
import logging

# We import our main config with an alias to avoid any confusion.
import config as framework_config

# --- THE CRITICAL FIX: MANUAL, UNAMBIGUOUS IMPORT ---
# This block is the definitive solution to a complex "name collision" bug.
# The SALMONN library has a file named 'config.py', which conflicts with our
# project's 'config.py'. This manual import ensures we load the correct one.

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
    logging.info(f"Temporarily added '{_SALMONN_CODE_PATH}' to Python path.")

try:
    from models.salmonn import SALMONN
    from transformers import WhisperFeatureExtractor
except ImportError as e:
    logging.exception("Failed to import SALMONN library components.")
    sys.exit(1)


def load_model_and_tokenizer(model_path: str) -> Tuple[object, object, object]:
    """
    Loads the complex, multi-component SALMONN model, processor, and tokenizer.
    This function fulfills our framework's 3-part "contract".
    """
    logging.info("--- Loading SALMONN Model System (using native config loader) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Target device for model loading: {device.upper()}")
    
    # Determine which component set to use based on the checkpoint filename
    if "7b" in os.path.basename(model_path).lower():
        logging.info("Detected SALMONN-7B model.")
        component_paths = framework_config.SALMONN_7B_COMPONENT_PATHS
    else:
        logging.info("Detected SALMONN-13B model (Default).")
        component_paths = framework_config.SALMONN_COMPONENT_PATHS

    # We use the model's own YAML config as a template, which is more robust
    # than creating the configuration from scratch.
    config_path = os.path.join(component_paths['source_code'], 'configs/decode_config.yaml')
    # We create a mock argparse object to satisfy the SALMONN Config class constructor.
    mock_args = type('Args', (), {'cfg_path': config_path, 'options': None})()
    cfg = SalmonnConfig(mock_args)

    # We then override the placeholder paths in the YAML with our actual local paths.
    cfg.config.model.llama_path = component_paths['vicuna']
    cfg.config.model.whisper_path = component_paths['whisper']
    cfg.config.model.beats_path = component_paths['beats']
    cfg.config.model.ckpt = component_paths['salmonn_checkpoint']
    
    # Explicitly force the checkpoint path from the argument to be sure
    cfg.config.model.ckpt = model_path

    logging.info("Instantiating SALMONN model from config...")
    model = SALMONN.from_config(cfg.config.model)
    model.to(device)
    model.eval()

    logging.info("Loading WhisperFeatureExtractor...")
    processor = WhisperFeatureExtractor.from_pretrained(
        component_paths['whisper'],
        local_files_only=True
    )
    
    # We correctly extract the tokenizer from the loaded model to return it separately.
    tokenizer = model.llama_tokenizer
    
    # We attach the full config object to the model for easy access later.
    model.custom_config = cfg.config

    logging.info("SALMONN model, processor, and tokenizer loaded successfully.")
    return model, processor, tokenizer


def _convert_messages_to_salmonn_prompt(messages: List[Dict[str, str]], model_config: dict) -> str:
    """
    A helper "translator" that converts our framework's standard 'messages' list
    into the specific, templated string that SALMONN's generate method expects.
    """
    full_text = ""
    is_cot_generation = False
    for msg in messages:
        content = msg.get("content", "")
        if "audio\n\n" in content:
            content = content.replace("audio\n\n", "").strip()
        # We check if this is a prompt for CoT generation.
        if "Let's think step by step:" in content:
            is_cot_generation = True
            content = content.replace("Let's think step by step:", "").strip()
        full_text += content + "\n"
    full_text = full_text.strip()

    # If this is a CoT generation step, we append a more explicit instruction
    # to reliably elicit a detailed reasoning chain from this specific model.
    if is_cot_generation:
        full_text += "\nPlease provide a detailed, step-by-step reasoning before giving your final answer. Let's think step by step:"

    # Finally, we wrap the entire text in the model's required chat template.
    prompt_template = model_config.model.prompt_template
    wrapped_text = "<Speech><SpeechHere></Speech> " + full_text
    return prompt_template.format(wrapped_text)


def run_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    audio_path: str, max_new_tokens: int, do_sample: bool,
    temperature: float, top_p: float
) -> str:
    """
    Runs multi-modal inference using the SALMONN model, including audio preprocessing.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # We MUST resample to 16kHz, as this is the sampling rate the model's
        # Whisper audio encoder was trained on.
        resampled_wav, sr = librosa.load(audio_path, sr=16000)
        
        # We use the WhisperFeatureExtractor (our 'processor') to convert the
        # raw audio waveform into a spectrogram (a visual representation of the sound).
        spectrogram = processor(resampled_wav, sampling_rate=sr, return_tensors="pt").input_features
        
        # We construct the 'samples' dictionary in the exact format the model requires.
        samples = {
            "spectrogram": spectrogram.to(model.device, dtype=torch.float32),
            "raw_wav": torch.from_numpy(resampled_wav).unsqueeze(0).to(model.device, dtype=torch.float32)
        }
        
        prompt = _convert_messages_to_salmonn_prompt(messages, model.custom_config)
        
        # We take the default generation settings from the model's config and
        # override them with the parameters for our specific experiment.
        generate_cfg = model.custom_config.generate
        generate_cfg['max_new_tokens'] = max_new_tokens
        generate_cfg['do_sample'] = do_sample
        generate_cfg['temperature'] = temperature
        generate_cfg['top_p'] = top_p
        generate_cfg['num_beams'] = 1 # We use greedy/sampling, not beam search.

        # We use automatic mixed precision for better performance on the GPU.
        if device == "cuda":
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                result = model.generate(samples, generate_cfg, prompts=[prompt])
        else:
            result = model.generate(samples, generate_cfg, prompts=[prompt])
        
        raw_output = result[0]
        
        # The model's output includes special start/end tokens; we remove them for clean results.
        cleaned_output = raw_output.replace("<s>", "").replace("</s>", "").strip()
        return cleaned_output
    except Exception as e:
        logging.exception("An error occurred during the run_inference process.")
        raise
    

def run_text_only_inference(
    model: object, processor: object, messages: List[Dict[str, str]],
    max_new_tokens: int, do_sample: bool, temperature: float, top_p: float
) -> str:
    """
    Handles text-only tasks by providing a dummy silent audio input.
    This ensures a scientifically pure test of the model's language capabilities.
    """
    silent_audio_path = framework_config.SILENT_AUDIO_PATH
    if not os.path.exists(silent_audio_path):
        raise FileNotFoundError(f"Silent audio file not found at '{silent_audio_path}'")
    
    return run_inference(
        model, processor, messages, silent_audio_path,
        max_new_tokens, do_sample, temperature, top_p
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
    """ A model-agnostic utility to format choices into a lettered string. """
    if not choices: return ""
    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        formatted_choices.append(f"({letter}) {choice}")
    return "\n".join(formatted_choices)