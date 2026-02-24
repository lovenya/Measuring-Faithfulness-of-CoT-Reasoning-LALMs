# core/qwen_utils.py

from typing import List
import nltk
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import re
import config

def load_model_and_tokenizer(model_path: str):
    """
    Loads the specified Qwen2-Audio model and processor.
    
    If QWEN_LOCAL_MODEL_PATH environment variable is set (for SSD optimization),
    it will load from that path instead of the provided model_path.
    """
    import os
    
    # Check for local SSD path (set by optimized submission scripts)
    local_path = os.environ.get('QWEN_LOCAL_MODEL_PATH')
    if local_path and os.path.exists(local_path):
        print(f"[OPTIMIZATION] Using local SSD model path: {local_path}")
        model_path = local_path
    
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path, 
        sampling_rate=16000 
    )
    
    tokenizer = processor.tokenizer
    
    print(f"Loading model from {model_path}...")
    # Using the latest transformers from source, we expect the library to handle
    # attention correctly without any special flags.
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("Model and processor loaded successfully!")
    return model, processor, tokenizer


def run_inference(model, processor, messages: list, audio_path: str, max_new_tokens: int, temperature: float = 1.0, top_p: float = 0.01, top_k: int = 0, do_sample: bool = True):
    """
    This is the universal, multi-modal inference function for the Qwen model.
    """
    
    conversation = []
    for message in messages:
        if message["role"] == "user" and "audio" in message.get("content", ""):
            conversation.append({
                "role": "user", 
                "content": [
                    {"type": "audio", "audio_path": audio_path},
                    {"type": "text", "text": message["content"]},
                ]
            })
        else:
            conversation.append(message)
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    audio_data, sampling_rate = librosa.load(
        audio_path, 
        sr=processor.feature_extractor.sampling_rate
    )
    
    inputs = processor(
        text=text,
        audio=audio_data,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if do_sample:
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k}
    else:
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}

    with torch.no_grad():
        generate_ids = model.generate(**inputs, **generation_kwargs)
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response


def run_text_only_inference(model, processor, messages: list, max_new_tokens: int, temperature: float = 1.0, top_p: float = 0.01, top_k: int = 0, do_sample: bool = True):
    """
    Runs inference on the model for a text-only task (no audio input).
    This is a specialized function for tasks like paraphrasing.
    """
    # This workflow is simpler: no audio processing, no multi-modal message construction.
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # The processor call only includes text.
    inputs = processor.tokenizer(text=text, return_tensors="pt", padding=True)

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if do_sample:
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k}
    else:
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}

    with torch.no_grad():
        generate_ids = model.generate(**inputs, **generation_kwargs)
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response

def sanitize_cot(cot_text: str) -> str:
    """
    Removes the final sentence from a Chain-of-Thought text.
    This is a critical step to prevent the model from simply copying the
    answer from a "spoiler" sentence like "Therefore, the answer is (A)."
    
    Args:
        cot_text (str): The original, generated Chain-of-Thought.
        
    Returns:
        str: The sanitized CoT with the final sentence removed.
    """
    if not cot_text:
        return ""
        
    # Use NLTK for robust sentence tokenization
    sentences = nltk.sent_tokenize(cot_text)
    
    # If there's more than one sentence, return all but the last one.
    # If there's only one sentence (or zero), return an empty string,
    # as that single sentence might be the spoiler.
    if len(sentences) > 1:
        return " ".join(sentences[:-1])
    else:
        return ""


def parse_answer(text: str) -> str | None:
    """
    A universal, robust parser for all experiments. It checks for answer
    patterns in a specific order to ensure correctness.

    1. Flexibly finds the ideal pattern `(X)` anywhere in the text.
    2. Checks for strict patterns like `X)` or `(X` matching the whole string.
    3. Checks for a single-letter response `X` matching the whole string.
    4. Identifies and labels refusals to answer if no choice is found.
    """
    if not text:
        return None

    cleaned_text = text.strip()

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

    # Priority 5: If no choice is found, check for a refusal.
    refusal_keywords = [
        "cannot be determined", "none of the choices", "ambiguous",
        "not enough information", "no definitive answer", "not have enough information", "cannot determine"
    ]
    if any(keyword in cleaned_text.lower() for keyword in refusal_keywords):
        return "REFUSAL"

    # Final Fallback: If no pattern matches, the output is unparsable.
    return None


def format_choices_for_prompt(choices: List[str]) -> str:
    """
    Model-agnostic function to format a list of choices into a string.
    e.g., ["cat", "dog"] -> "(A) cat\n(B) dog"
    """
    if not choices:
        return ""
    
    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        formatted_choices.append(f"({letter}) {choice}")
        
    return "\n".join(formatted_choices)