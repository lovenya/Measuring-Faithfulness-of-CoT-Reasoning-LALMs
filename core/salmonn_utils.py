# core/salmonn_utils.py

import sys
import os
import torch
import nltk
import re
import librosa
import yaml
from typing import Tuple, List, Dict

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
    # --- THE CRITICAL IMPORT ---
    # We now import the authors' own data preparation utility.
    from utils import prepare_one_sample
except ImportError as e:
    print(f"FATAL: Failed to import from 'salmonn-source-code'. Error: {e}")
    sys.exit(1)


def load_model_and_tokenizer(model_path: str) -> Tuple[object, object]:
    """ Assembles the SALMONN 13B model from its constituent components. """
    yaml_path = os.path.join(_SALMONN_CODE_PATH, 'configs', 'decode_config.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"SALMONN decode_config.yaml not found at: {yaml_path}")
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # We override the paths AND the model architecture to perfectly match the
    # pre-trained checkpoint we are loading.
    cfg['model']['llama_path'] = project_config.MODEL_PATHS['salmonn_vicuna']
    cfg['model']['whisper_path'] = project_config.MODEL_PATHS['salmonn_whisper']
    cfg['model']['beats_path'] = project_config.MODEL_PATHS['salmonn_beats']
    cfg['model']['ckpt'] = os.path.join(project_config.MODEL_PATHS['salmonn_checkpoint'], 'salmonn_v1.pth')
    
    # This is the critical line that fixes the size mismatch. The checkpoint
    # was trained with 1 query token, so we must build the model with 1.
    cfg['model']['num_speech_query_token'] = 1
    
    prompt_path = cfg['model']['prompt_path']
    test_prompt_path = cfg['model']['test_prompt_path']
    cfg['model']['prompt_path'] = os.path.join(_SALMONN_CODE_PATH, prompt_path)
    cfg['model']['test_prompt_path'] = os.path.join(_SALMONN_CODE_PATH, test_prompt_path)

    print("Loading SALMONN model from config...")
    model = SALMONN.from_config(cfg['model'])
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

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
    """
    Runs multi-modal inference by exactly replicating the logic from the authors'
    own cli_inference.py script.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # --- THE FINAL, DEFINITIVE FIX ---
    # 1. Use the authors' 'prepare_one_sample' utility to create the 'samples' dictionary.
    #    This guarantees the input is in the exact format the model expects.
    samples = prepare_one_sample(audio_path, processor)
    # Move all tensors in the dictionary to the correct device.
    samples = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in samples.items()}

    # 2. Format the text prompt as before.
    text_prompt = [_convert_messages_to_salmonn_prompt(messages, model)]

    # 3. Construct the 'generate_cfg' dictionary as required.
    generate_cfg = model.generate_cfg_template.copy()
    generate_cfg['max_new_tokens'] = max_new_tokens
    generate_cfg['do_sample'] = do_sample
    generate_cfg['temperature'] = temperature
    generate_cfg['top_p'] = top_p
    
    # 4. Call generate() with the three correct arguments: samples, generate_cfg, and prompts.
    with torch.cuda.amp.autocast(dtype=torch.float16):
        response = model.generate(
            samples,
            generate_cfg=generate_cfg,
            prompts=text_prompt,
        )
    # --- END OF FIX ---
    return response[0]


# ... (The rest of the file is correct and remains unchanged) ...

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