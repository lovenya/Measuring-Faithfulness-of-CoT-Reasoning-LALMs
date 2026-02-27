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
from transformers import AutoModelForSeq2SeqLM, AutoProcessor

import config as framework_config

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "it",
    "this",
    "that",
    "for",
    "with",
    "as",
    "by",
    "at",
    "but",
    "not",
    "be",
    "about",
    "which",
    "they",
    "i",
}


def _strip_audio_token(text: str) -> str:
    """Remove framework's leading audio marker from prompt text."""
    if not text:
        return ""
    return text.replace("audio\n\n", "", 1).strip()


def _move_inputs_to_model_dtype(inputs: Dict[str, torch.Tensor], model: object) -> Dict[str, torch.Tensor]:
    """
    Move model inputs to the target device and cast floating tensors to the
    model dtype (e.g., fp16) while preserving integer tensors like input_ids.
    """
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device

    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        model_dtype = next(model.parameters()).dtype

    prepared: Dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if torch.is_floating_point(value):
            prepared[key] = value.to(device=model_device, dtype=model_dtype)
        else:
            prepared[key] = value.to(device=model_device)
    return prepared


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

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
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
    do_sample: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
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
    inputs = _move_inputs_to_model_dtype(inputs, model)

    generation_kwargs = {"max_new_tokens": max_new_tokens}
    if do_sample is True:
        generation_kwargs.update(
            {
                "do_sample": True,
            }
        )
        if temperature is not None:
            generation_kwargs["temperature"] = temperature
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        if top_k is not None:
            generation_kwargs["top_k"] = top_k
    elif do_sample is False:
        generation_kwargs.update({"do_sample": False})

    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    generated_ids = outputs[:, input_len:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0] if decoded else ""


def run_text_only_inference(
    model: object,
    processor: object,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    do_sample: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
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


def _extract_meaningful_words(text: str) -> set[str]:
    if not text:
        return set()
    clean = re.sub(r"[^\w\s]", "", str(text)).strip().lower()
    return set(clean.split()) - _STOPWORDS


def _parse_choices_from_formatted(choices_formatted: str) -> List[str]:
    choices_list: List[str] = []
    for line in choices_formatted.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^\([A-J]\)\s*(.+)$", stripped)
        choices_list.append(match.group(1).strip() if match else stripped)
    return choices_list


def _parse_conditioned_output(raw_text: str, choices_list: List[str] | None = None) -> str | None:
    if not raw_text:
        return None

    cleaned = raw_text.strip()
    letters = [chr(ord("A") + i) for i in range(10)]

    end_chunk = cleaned[-100:]
    paren_matches = list(re.finditer(r"\(([A-J])\)", end_chunk, re.IGNORECASE))
    if paren_matches:
        return paren_matches[-1].group(1).upper()

    prefix_matches = list(
        re.finditer(r"(?:option|choice|answer|answer\s*is|is)\s*[:*]*\s*([A-J])\b", end_chunk, re.IGNORECASE)
    )
    if prefix_matches:
        return prefix_matches[-1].group(1).upper()

    if choices_list:
        target_words = _extract_meaningful_words(cleaned)
        best_letter = None
        max_score = 0.0
        for i, option in enumerate(choices_list):
            option_words = _extract_meaningful_words(option)
            if not option_words:
                continue
            overlap = option_words.intersection(target_words)
            score = len(overlap) + (len(overlap) / len(option_words))
            if score > max_score and len(overlap) > 0:
                max_score = score
                best_letter = letters[i]
        if best_letter:
            return best_letter

    standalone = re.search(r"\b([A-J])\b[^\w]*$", cleaned[-30:], re.IGNORECASE)
    if standalone and standalone.group(1).upper() not in ("A", "I"):
        return standalone.group(1).upper()

    return parse_answer(cleaned)


def format_choices_for_prompt(choices: List[str]) -> str:
    """Format answer options as (A)...(B)... lines."""
    if not choices:
        return ""

    formatted = []
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        formatted.append(f"({letter}) {choice}")
    return "\n".join(formatted)


def run_conditioned_inference(
    model: object,
    processor: object,
    tokenizer: object,
    question: str,
    choices_formatted: str,
    audio_path: str,
    provided_reasoning: str,
) -> Dict[str, object]:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    prompt_text = (
        f"{question} Select one option from the provided choices.\n"
        f"{choices_formatted}. "
        "Please think and reason about the input audio before you respond.\n\n"
        f"{provided_reasoning}\n\n"
        "Therefore, the answer is:"
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio", "path": os.path.abspath(audio_path)},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    inputs = _move_inputs_to_model_dtype(inputs, model)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)

    generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    parsed = _parse_conditioned_output(raw_output, _parse_choices_from_formatted(choices_formatted))
    return {
        "predicted_choice": parsed,
        "final_answer_raw": raw_output,
        "final_prompt_messages": conversation,
    }
