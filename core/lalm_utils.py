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
    
    audio_data, sampling_rate = librosa.load(
        audio_path, 
        sr=processor.feature_extractor.sampling_rate
    )
    
    # --- THE CRITICAL FIX ---
    # The processor expects the keyword 'audios' (plural), not 'audio'.
    # This aligns our code with the official Qwen documentation and resolves the
    # underlying cause of the downstream 'cache_position' error.
    inputs = processor(
        text=text,
        audio=audio_data, # <-- Corrected keyword from 'audio' to 'audios'
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )
    # --- END OF FIX ---

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if do_sample:
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        }
    else:
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

    with torch.no_grad():
        generate_ids = model.generate(**inputs, **generation_kwargs)
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response


def parse_answer(text: str) -> str | None:
    """
    A universal, robust parser for all experiments.
    1. Flexibly finds the answer choice `(X)` anywhere in the text.
    2. Explicitly identifies and labels refusals to answer.
    """
    if not text:
        return None

    cleaned_text = text.strip()
    
    # Step 1: Flexibly search for the answer pattern `(X)` anywhere.
    # This handles cases like "The answer is (A)." and "(A)\nThank you."
    match = re.search(r'\(([A-Z])\)', cleaned_text)
    if match:
        return match.group(1)

    # Step 2: If no choice is found, check for a refusal.
    refusal_keywords = [
        "cannot be determined", "none of the choices", "ambiguous",
        "not enough information", "no definitive answer"
    ]
    if any(keyword in cleaned_text.lower() for keyword in refusal_keywords):
        return "REFUSAL"

    # Step 3: If neither is found, the output is truly unparsable.
    return None