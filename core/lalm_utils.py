# core/lalm_utils.py

import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import re


def load_model_and_tokenizer(model_path: str):
    """
    Loads the specified Qwen2-Audio model and processor.
    """
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path, 
        sampling_rate=16000 
    )
    
    print(f"Loading model from {model_path}...")
    # Using the latest transformers from source, we expect the library to handle
    # attention correctly without any special flags.
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("Model and processor loaded successfully!")
    return model, processor


def run_inference(model, processor, messages: list, audio_path: str, max_new_tokens: int, temperature: float = 0.6, top_p: float = 0.9, do_sample: bool = True):
    """
    Runs inference on the model with a given chat history and audio input.
    """
    # Create conversation with audio
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
    
    audio_data, _ = librosa.load(
        audio_path, 
        sr=processor.feature_extractor.sampling_rate
    )
    
    # --- THE CRITICAL FIX ---
    # The processor expects the keyword 'audios' (plural), not 'audio'.
    # This aligns our code with the official Qwen documentation and resolves the
    # underlying cause of the downstream 'cache_position' error.
    inputs = processor(
        text=text,
        audios=[audio_data], # <-- Corrected keyword from 'audio' to 'audios'
        return_tensors="pt",
        padding=True
    )
    # --- END OF FIX ---

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
    }

    if do_sample:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        generate_ids = model.generate(**inputs, **generation_kwargs)
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response


def parse_answer(text: str) -> str | None:
    """
    Parses the model's text output to find the multiple-choice answer.
    """
    if not text:
        return None

    cleaned_text = text.strip()

    match = re.search(r'^\(([A-Z])\)$', cleaned_text)
    if match:
        return match.group(1)

    match = re.search(r'^([A-Z])\)$', cleaned_text)
    if match:
        return match.group(1)

    match = re.search(r'^\(([A-Z])', cleaned_text)
    if match:
        return match.group(1)

    match = re.search(r'answer is\s+([A-Z])', cleaned_text, re.IGNORECASE)
    if match:
        return match.group(1)

    if len(cleaned_text) == 1 and 'A' <= cleaned_text <= 'Z':
        return cleaned_text

    return None
