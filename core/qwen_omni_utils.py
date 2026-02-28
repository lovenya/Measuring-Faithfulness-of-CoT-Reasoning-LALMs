# core/qwen_omni_utils.py

"""
Transformers backend adapter for Qwen2.5-Omni.

This module follows the framework utility contract while keeping prompting,
parsing, and generation behavior aligned with Pooneh's Qwen Omni wrappers.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import torch

import config as framework_config

_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

_SINGLE_TURN_SUFFIX = (
    "Please think step-by-step about the audio and the choices provided. "
    "At the very end of your response, explicitly state your final prediction "
    "using only the single letter of the correct choice (e.g., A, B, C, or D)."
)

_PROCESS_MM_INFO = None


def _strip_audio_token(text: str) -> str:
    if not text:
        return ""
    return text.replace("audio\n\n", "", 1).strip()


def _extract_instruction_from_messages(messages: List[Dict[str, str]]) -> str:
    user_texts: List[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        cleaned = _strip_audio_token(msg.get("content", ""))
        if cleaned:
            user_texts.append(cleaned)

    if not user_texts:
        return ""

    selected = next(
        (text for text in user_texts if "Question:" in text and "Choices:" in text),
        user_texts[0],
    )
    return selected.replace(_SINGLE_TURN_SUFFIX, "").strip()


def _build_xml_prompt_text(instruction: str) -> str:
    return (
        f"{instruction}\n\n"
        "You must analyze the audio and provide your answer strictly following the template below. "
        "Do not include any other text outside of these tags.\n\n"
        "Template:\n"
        "<Reasoning>\n"
        "[Your step-by-step thinking here]\n"
        "</Reasoning>\n"
        "<Conclusion>\n"
        "[Single letter choice here, e.g., A]\n"
        "</Conclusion>"
    )


def _build_no_reasoning_xml_prompt_text(instruction: str) -> str:
    """Build XML prompt that asks for a direct answer without reasoning."""
    return (
        f"{instruction}\n\n"
        "Based on the audio, provide your answer directly using the template below. "
        "Do not include any reasoning or analysis.\n\n"
        "Template:\n"
        "<Conclusion>\n"
        "[Single letter choice here, e.g., A]\n"
        "</Conclusion>"
    )


def _build_conditioned_xml_prompt_text(instruction: str, provided_reasoning: str) -> str:
    return (
        f"{instruction}\n\n"
        "You must analyze the audio and provide your answer strictly following the template below. "
        "The analysis has been provided for you; use it to reach the conclusion.\n\n"
        "Template:\n"
        "<Reasoning>\n"
        f"{provided_reasoning}\n"
        "</Reasoning>\n"
        "<Conclusion>\n"
        "[Single letter choice here, e.g., A]\n"
        "</Conclusion>"
    )


def _parse_model_output(raw_text: str) -> Dict[str, str | None]:
    reasoning = ""
    predicted_choice = None

    reason_match = re.search(r"<Reason.*?>(.*?)</Reason.*?>", raw_text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    conclusion_match = re.search(r"<Conclu.*?>\s*([A-Za-z])\s*</Conclu.*?>", raw_text, re.IGNORECASE)
    if conclusion_match:
        predicted_choice = conclusion_match.group(1).upper()

    if not predicted_choice:
        cleaned_text = raw_text.strip()
        fallback_match = re.search(r"(?:[^a-zA-Z]|^)([A-D])(?:[^a-zA-Z]*)$", cleaned_text, re.IGNORECASE)
        if fallback_match:
            predicted_choice = fallback_match.group(1).upper()
            if not reasoning:
                reasoning = cleaned_text[: fallback_match.start()].strip()

    return {
        "raw_output": raw_text,
        "reasoning": reasoning if reasoning else raw_text,
        "predicted_choice": predicted_choice,
    }


def _ensure_process_mm_info():
    global _PROCESS_MM_INFO
    if _PROCESS_MM_INFO is not None:
        return _PROCESS_MM_INFO

    try:
        from qwen_omni_utils import process_mm_info  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Missing qwen_omni_utils/process_mm_info dependency. "
            "Install torchvision and qwen_omni_utils in qwen_omni_env."
        ) from exc

    _PROCESS_MM_INFO = process_mm_info
    return _PROCESS_MM_INFO


def load_model_and_tokenizer(model_path: str) -> Tuple[object, object, object]:
    local_path = os.environ.get("QWEN_OMNI_LOCAL_MODEL_PATH")
    if local_path and os.path.isdir(local_path):
        model_path = local_path
        print(f"[OPTIMIZATION] Using local Qwen Omni model path: {model_path}")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            "Qwen2.5-Omni local model directory not found. Expected: "
            f"{model_path}"
        )

    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    except Exception as exc:
        raise ImportError(
            "Qwen Omni transformers classes are unavailable in this environment. "
            "Install a transformers build that includes Qwen2.5-Omni support."
        ) from exc

    _ensure_process_mm_info()

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    tokenizer = getattr(processor, "tokenizer", processor)
    return model, processor, tokenizer


def _build_conversation(messages: List[Dict[str, str]], audio_path: str) -> List[Dict[str, object]]:
    instruction = _extract_instruction_from_messages(messages)
    prompt_text = _build_xml_prompt_text(instruction)

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": _DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio", "audio": os.path.abspath(audio_path)},
            ],
        },
    ]


def _build_no_reasoning_conversation(
    question: str,
    choices_formatted: str,
    audio_path: str,
) -> List[Dict[str, object]]:
    """Build conversation for no-reasoning inference (direct answer only)."""
    instruction = f"Question: {question}\nChoices:\n{choices_formatted}"
    prompt_text = _build_no_reasoning_xml_prompt_text(instruction)

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": _DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio", "audio": os.path.abspath(audio_path)},
            ],
        },
    ]


def _build_conditioned_conversation(
    question: str,
    choices_formatted: str,
    provided_reasoning: str,
    audio_path: str,
) -> List[Dict[str, object]]:
    instruction = f"Question: {question}\nChoices:\n{choices_formatted}"
    prompt_text = _build_conditioned_xml_prompt_text(instruction, provided_reasoning)

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": _DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio", "audio": os.path.abspath(audio_path)},
            ],
        },
    ]


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
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    conversation = _build_conversation(messages, audio_path)

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    process_mm_info = _ensure_process_mm_info()
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Keep generation behavior aligned with Pooneh's wrapper.
    with torch.no_grad():
        text_ids, _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_audio_in_video=False,
        )

    generated_ids = text_ids[:, inputs.input_ids.shape[1] :]

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0] if decoded else ""


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

    conversation = _build_conditioned_conversation(
        question=question,
        choices_formatted=choices_formatted,
        provided_reasoning=provided_reasoning,
        audio_path=audio_path,
    )

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    process_mm_info = _ensure_process_mm_info()
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        text_ids, _ = model.generate(
            **inputs,
            max_new_tokens=32,
            use_audio_in_video=False,
        )

    generated_ids = text_ids[:, inputs.input_ids.shape[1] :]
    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {
        "predicted_choice": parse_answer(raw_output),
        "final_answer_raw": raw_output,
        "final_prompt_messages": conversation,
    }


def run_no_reasoning_inference(
    model: object,
    processor: object,
    tokenizer: object,
    question: str,
    choices_formatted: str,
    audio_path: str,
) -> Dict[str, object]:
    """Run inference with no-reasoning prompt (direct answer only)."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    conversation = _build_no_reasoning_conversation(
        question=question,
        choices_formatted=choices_formatted,
        audio_path=audio_path,
    )

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    process_mm_info = _ensure_process_mm_info()
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        text_ids, _ = model.generate(
            **inputs,
            max_new_tokens=32,
            use_audio_in_video=False,
        )

    generated_ids = text_ids[:, inputs.input_ids.shape[1] :]
    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {
        "predicted_choice": parse_answer(raw_output),
        "final_answer_raw": raw_output,
        "final_prompt_messages": conversation,
    }


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
    if not cot_text:
        return ""
    parsed = _parse_model_output(cot_text)
    reasoning = parsed.get("reasoning")
    return str(reasoning) if reasoning is not None else cot_text


def parse_answer(text: str) -> str | None:
    if not text:
        return None
    parsed = _parse_model_output(text)
    predicted = parsed.get("predicted_choice")
    return str(predicted) if predicted is not None else None


def format_choices_for_prompt(choices: List[str]) -> str:
    if not choices:
        return ""

    formatted = []
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        formatted.append(f"({letter}) {choice}")
    return "\n".join(formatted)
