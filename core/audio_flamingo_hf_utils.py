# core/audio_flamingo_hf_utils.py

"""
Hugging Face / transformers backend adapter for Audio Flamingo 3.

This module implements the same framework contract as other model utility
modules while enforcing strict local-path model loading.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import nltk
import torch
from peft import PeftModel
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

import config as framework_config


def _strip_audio_token(text: str) -> str:
    """Remove framework's leading audio marker from prompt text."""
    if not text:
        return ""
    return text.replace("audio\n\n", "", 1).strip()


def load_model_and_tokenizer(model_path: str) -> Tuple[object, object, object]:
    """Load AF3 HF model from a strict local path with think adapter."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            "AF3 HF local model directory not found. Expected: "
            f"{model_path}. Clone it locally before running."
        )

    think_dir = os.path.join(model_path, "think")
    adapter_path = os.path.join(think_dir, "adapter_model.safetensors")
    non_lora_path = os.path.join(think_dir, "non_lora_trainables.bin")

    if not os.path.isfile(adapter_path) or not os.path.isfile(non_lora_path):
        raise FileNotFoundError(
            "AF3 HF think adapter files are missing. Expected files:\n"
            f"  - {adapter_path}\n"
            f"  - {non_lora_path}"
        )

    processor = AutoProcessor.from_pretrained(model_path)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
    model.load_state_dict(non_lora_trainables, strict=False)

    model = PeftModel.from_pretrained(model, model_path, subfolder="think")
    model.eval()

    tokenizer = getattr(processor, "tokenizer", processor)
    return model, processor, tokenizer


def _build_conversation(messages: List[Dict[str, str]], audio_path: str) -> List[Dict[str, object]]:
    """Translate framework chat messages into AF3 HF chat template format."""
    conversation: List[Dict[str, object]] = []
    audio_attached = False

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        has_audio_marker = "audio\n\n" in content
        clean_text = _strip_audio_token(content)

        if role == "user" and has_audio_marker and not audio_attached:
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": clean_text},
                        {"type": "audio", "path": os.path.abspath(audio_path)},
                    ],
                }
            )
            audio_attached = True
        else:
            conversation.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": clean_text if clean_text else content}],
                }
            )

    if not audio_attached:
        injected = False
        for idx, msg in enumerate(conversation):
            if msg.get("role") == "user":
                text_chunks = msg.get("content", [])
                if isinstance(text_chunks, list):
                    text_chunks.append({"type": "audio", "path": os.path.abspath(audio_path)})
                    msg["content"] = text_chunks
                    injected = True
                    break
        if not injected:
            conversation.insert(
                0,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please answer the following question about the audio."},
                        {"type": "audio", "path": os.path.abspath(audio_path)},
                    ],
                },
            )

    return conversation


def run_inference(
    model: object,
    processor: object,
    messages: List[Dict[str, str]],
    audio_path: str,
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    """Run multimodal inference for AF3 HF."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    conversation = _build_conversation(messages, audio_path)
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    inputs = inputs.to(model.device)

    generation_kwargs = {"max_new_tokens": max_new_tokens}
    if do_sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
    else:
        generation_kwargs.update({"do_sample": False})

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    generated_ids = outputs[:, inputs.input_ids.shape[1] :]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0] if decoded else ""


def run_text_only_inference(
    model: object,
    processor: object,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    """Run text-only tasks by pairing prompts with framework silent audio."""
    silent_audio_path = framework_config.SILENT_AUDIO_PATH
    if not os.path.exists(silent_audio_path):
        raise FileNotFoundError(f"Silent audio file not found at: {silent_audio_path}")

    return run_inference(
        model=model,
        processor=processor,
        messages=messages,
        audio_path=silent_audio_path,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def sanitize_cot(cot_text: str) -> str:
    """Remove the final sentence to reduce answer leakage in follow-up prompts."""
    if not cot_text:
        return ""

    sentences = nltk.sent_tokenize(cot_text)
    if len(sentences) > 1:
        return " ".join(sentences[:-1])
    return ""


def parse_answer(text: str) -> str | None:
    """Robust parser for multiple answer formats."""
    if not text:
        return None

    cleaned = text.strip().strip(".").strip()

    match = re.search(r"\(([a-zA-Z])\)", cleaned)
    if match:
        return match.group(1).upper()

    match = re.search(r"([a-zA-Z])\)", cleaned)
    if match:
        return match.group(1).upper()

    match = re.search(r"\(([a-zA-Z])", cleaned)
    if match:
        return match.group(1).upper()

    match = re.search(r"(?:choice|option|answer|prediction|conclusion)\s*[:\-]?\s*([A-D])\b", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    matches = re.findall(r"(?:^|[^a-zA-Z])([A-D])(?:[^a-zA-Z]|$)", cleaned, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    if len(cleaned) == 1 and cleaned.isalpha():
        return cleaned.upper()

    refusal_keywords = [
        "cannot be determined",
        "none of the choices",
        "ambiguous",
        "not enough information",
        "no definitive answer",
    ]
    if any(keyword in cleaned.lower() for keyword in refusal_keywords):
        return "REFUSAL"

    return None


def format_choices_for_prompt(choices: List[str]) -> str:
    """Format answer options as (A)...(B)... lines."""
    if not choices:
        return ""

    formatted = []
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        formatted.append(f"({letter}) {choice}")
    return "\n".join(formatted)
