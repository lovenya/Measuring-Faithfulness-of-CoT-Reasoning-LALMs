# core/salmonn_utils.py

import sys
import os
import torch
import nltk
import re
import librosa
import yaml
from typing import Tuple, List, Dict

# We import our main project config to get the master paths to the model components.
import config as project_config

# --- Environment Setup for Custom SALMONN Code ---
_SALMONN_CODE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'salmonn-source-code'))
if not os.path.exists(_SALMONN_CODE_PATH):
    raise FileNotFoundError(f"SALMONN source code not found at: {_SALMONN_CODE_PATH}")
if _SALMONN_CODE_PATH not in sys.path:
    sys.path.append(_SALMONN_CODE_PATH)
    print(f"INFO: Temporarily added '{_SALMONN_CODE_PATH}' to Python path.")

try:
    from models.salmonn import SALMONN
    from transformers import WhisperFeatureExtractor
except ImportError as e:
    print(f"FATAL: Failed to import from 'salmonn-source-code'. Ensure you have run 'pip install -r requirements.txt'. Error: {e}")
    sys.exit(1)


# In core/salmonn_utils.py

def load_model_and_tokenizer(model_path: str) -> Tuple[object, object]:
    """
    Assembles the SALMONN model from its constituent components, now with
    robust, absolute pathing for its internal config files.
    """
    # 1. Load the base YAML config designed for inference.
    yaml_path = os.path.join(_SALMONN_CODE_PATH, 'configs', 'decode_config.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"SALMONN decode_config.yaml not found at: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. Programmatically override the paths in the config with our local paths.
    cfg['model']['llama_path'] = project_config.MODEL_PATHS['salmonn_vicuna']
    cfg['model']['whisper_path'] = project_config.MODEL_PATHS['salmonn_whisper']
    cfg['model']['beats_path'] = project_config.MODEL_PATHS['salmonn_beats']
    cfg['model']['ckpt'] = os.path.join(project_config.MODEL_PATHS['salmonn_checkpoint'], 'SALMONN_13B.pth')

    # --- THE CRITICAL FIX ---
    # The paths in the YAML are relative to the source code directory.
    # We must convert them to absolute paths so the model can find them
    # from our project's root execution directory.
    prompt_path = cfg['model']['prompt_path']
    test_prompt_path = cfg['model']['test_prompt_path']
    
    cfg['model']['prompt_path'] = os.path.join(_SALMONN_CODE_PATH, prompt_path)
    cfg['model']['test_prompt_path'] = os.path.join(_SALMONN_CODE_PATH, test_prompt_path)
    # --- END OF FIX ---

    print("Loading SALMONN model from config...")
    model = SALMONN.from_config(cfg['model'])
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Store the generation config and prompt template on the model object for later use.
    model.generate_cfg_template = cfg['generate']
    model.prompt_template = cfg['model']['prompt_template']

    print("Loading Whisper feature extractor...")
    processor = WhisperFeatureExtractor.from_pretrained(project_config.MODEL_PATHS['salmonn_whisper'])
    
    print("SALMONN model and processor loaded successfully.")
    return model, processor


def _convert_messages_to_salmonn_prompt(messages: List[Dict], model: object) -> str:
    """ Correctly formats our standard 'messages' list into the specific prompt string that SALMONN expects. """
    text_content = ""
    for msg in messages:
        if "audio\n\n" in msg.get("content", ""):
            content = msg["content"].replace("audio\n\n", "").strip()
            text_content += content + "\n"
        else:
            text_content += msg.get("content", "").strip() + "\n"
    
    prompt_template = model.prompt_template
    return prompt_template.format("<Speech><SpeechHere></Speech> " + text_content.strip())


def run_inference(
    model: object, processor: object, messages: List[Dict],
    audio_path: str, max_new_tokens: int, do_sample: bool,
    temperature: float, top_p: float
) -> str:
    """ Runs multi-modal inference using the SALMONN model. """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # 1. Pre-process the audio.
    wav_file, sr = librosa.load(audio_path, sr=16000)
    audio_input = processor(wav_file, sampling_rate=sr, return_tensors="pt").input_features
    samples = {'audio_input': audio_input.to(model.device)}

    # 2. Format the text prompt.
    text_prompt = [_convert_messages_to_salmonn_prompt(messages, model)]

    # --- THE FINAL, DEFINITIVE FIX ---
    # We create the 'generate_cfg' dictionary that the model's generate() method expects.
    # We start with the template from the YAML and override it with our specific parameters.
    generate_cfg = model.generate_cfg_template.copy()
    generate_cfg['max_new_tokens'] = max_new_tokens
    generate_cfg['do_sample'] = do_sample
    generate_cfg['temperature'] = temperature
    generate_cfg['top_p'] = top_p
    # --- END OF FIX ---

    # 3. Generate the response, passing the correctly formatted arguments.
    with torch.cuda.amp.autocast(dtype=torch.float16):
        response = model.generate(
            samples,
            generate_cfg=generate_cfg, # Pass the dictionary here
            prompts=text_prompt,
        )
    return response[0]


def run_text_only_inference(
    model: object, processor: object, messages: List[Dict],
    max_new_tokens: int, do_sample: bool, temperature: float, top_p: float
) -> str:
    """ Runs text-only inference using our robust 'silent audio' methodology. """
    silent_audio_path = project_config.SILENT_AUDIO_PATH
    return run_inference(
        model, processor, messages, silent_audio_path,
        max_new_tokens, do_sample, temperature, top_p
    )


def sanitize_cot(cot_text: str) -> str:
    """ Model-agnostic utility to remove the final 'spoiler' sentence from a CoT. """
    if not cot_text: return ""
    sentences = nltk.sent_tokenize(cot_text)
    if len(sentences) > 1: return " ".join(sentences[:-1])
    else: return ""


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
    refusal_keywords = ["cannot be determined", "none of the choices", "ambiguous", "not enough information", "no definitive answer"]
    if any(keyword in cleaned_text.lower() for keyword in refusal_keywords):
        return "REFUSAL"
    return None


def format_choices_for_prompt(choices: List[str]) -> str:
    """ Model-agnostic utility to format choices into a lettered string. """
    if not choices: return ""
    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        formatted_choices.append(f"({letter}) {choice}")
    return "\n".join(formatted_choices)