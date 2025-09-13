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
# This block adds the SALMONN source code to Python's path, allowing us to import its custom modules.
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


def load_model_and_tokenizer(model_path: str) -> Tuple[object, object]:
    """
    Assembles the SALMONN model from its constituent components using its
    purpose-built `from_config` factory method.
    """
    # 1. Create the configuration dictionary that the model expects.
    #    This structure is based on the 'config.yaml' from the SALMONN repository.
    model_config = {
        'llama_path': project_config.MODEL_PATHS['salmonn_vicuna'],
        'whisper_path': project_config.MODEL_PATHS['salmonn_whisper'],
        'beats_path': project_config.MODEL_PATHS['salmonn_beats'],
        'ckpt': os.path.join(model_path, 'salmonn_7b_v0.pth'), # model_path is the checkpoint dir
        'freeze_whisper': True,
        'freeze_beats': True,
        'use_speech_Qformer': True,
        'freeze_speech_QFormer': True, # Set to True for inference
        'window_level_Qformer': True,
        'num_speech_query_token': 1,
        'second_per_window': 0.32,
        'second_stride': 0.32,
        'prompt_template': 'USER: {}\nASSISTANT:',
        'max_txt_len': 300,
        'end_sym': '</s>',
        'lora': False # LoRA is for training, disable for inference
    }

    print("Loading SALMONN model from config...")
    # 2. Use the correct 'from_config' factory method to initialize the model.
    #    This is the critical fix that resolves our previous error.
    model = SALMONN.from_config(model_config)
    
    # The SALMONN class does not automatically store the prompt template.
    # We will store it on the model object ourselves right after creation.
    # This makes it accessible to our helper functions later.
    model.prompt_template = model_config['prompt_template']
    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Whisper feature extractor...")
    # 3. Load the correct processor, as identified in their cli_inference.py.
    processor = WhisperFeatureExtractor.from_pretrained(project_config.MODEL_PATHS['salmonn_whisper'])
    
    print("SALMONN model and processor loaded successfully.")
    return model, processor


def _convert_messages_to_salmonn_prompt(messages: List[Dict], model: object) -> str:
    """
    Correctly formats our standard 'messages' list into the specific prompt
    string that SALMONN expects, including its unique placeholder.
    """
    text_content = ""
    for msg in messages:
        if "audio\n\n" in msg.get("content", ""):
            content = msg["content"].replace("audio\n\n", "").strip()
            text_content += content + "\n"
        else:
            text_content += msg.get("content", "").strip() + "\n"
    
    # Use the model's own prompt template for robustness.
    prompt_template = model.prompt_template
    # Use the correct placeholder, as seen in their source code.
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

    # 3. Generate the response.
    with torch.cuda.amp.autocast(dtype=torch.float16):
        response = model.generate(
            samples,
            prompts=text_prompt,
            max_length=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
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