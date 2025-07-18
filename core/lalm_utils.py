import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import re


def load_model_and_tokenizer(model_path: str):
    """
    Loads the specified Qwen2-Audio model and processor.

    Args:
        model_path (str): The path to the model directory.

    Returns:
        tuple: A tuple containing the loaded model and processor.
    """
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    print(f"Loading model from {model_path}...")
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

    Args:
        model: The loaded model.
        processor: The loaded processor.
        messages (list): The list of messages in the chat history.
        audio_path (str): Path to the audio file.
        max_new_tokens (int): The maximum number of new tokens to generate.
        temperature (float): The temperature for sampling.
        top_p (float): The top_p for sampling.
        do_sample (bool): Whether to use sampling. Set to False for deterministic output.

    Returns:
        str: The generated text response.
    """
    # Create conversation with audio
    conversation = []
    for message in messages:
        if message["role"] == "user" and "audio" in message.get("content", ""):
            # Add audio to the first user message
            conversation.append({
                "role": "user", 
                "content": [
                    {"type": "audio", "audio_path": audio_path},
                    {"type": "text", "text": message["content"]},
                ]
            })
        else:
            conversation.append(message)
    
    # Apply chat template
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    # Load audio
    audio_data, _ = librosa.load(
        audio_path, 
        sr=processor.feature_extractor.sampling_rate
    )
    
    # Build the BatchEncoding
    inputs = processor(
        text=text,
        audio=[audio_data],
        return_tensors="pt",
        padding=True
    )

    # Move all tensors to the same device as the model
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Start with the generation arguments that are always used
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
    }

    # Conditionally add sampling parameters only if do_sample is True
    if do_sample:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["do_sample"] = False

    # Generate response
    with torch.no_grad():
        generate_ids = model.generate(**inputs, **generation_kwargs)
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response


def parse_answer(text: str) -> str | None:
    """
    Parses the model's text output to find the multiple-choice answer.
    This version is robust enough to handle all observed formats including (A), A, and A).
    """
    if not text:
        return None

    # Clean the input string thoroughly once at the beginning
    cleaned_text = text.strip()

    # Pattern 1: (A) - Most specific, check first.
    match = re.search(r'^\(([A-Z])\)$', cleaned_text)
    if match:
        return match.group(1)

    # Pattern 2: A) - The one I repeatedly missed.
    # The ^ and $ ensure it matches the entire string and not a substring.
    match = re.search(r'^([A-Z])\)$', cleaned_text)
    if match:
        return match.group(1)

    # Pattern 3: "answer is A" - Verbose but clear.
    # This pattern is less likely now but good to keep for robustness.
    match = re.search(r'answer is\s+([A-Z])', cleaned_text, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern 4: A - The most minimal case.
    if len(cleaned_text) == 1 and 'A' <= cleaned_text <= 'Z':
        return cleaned_text

    # If all other checks fail, return None.
    return None