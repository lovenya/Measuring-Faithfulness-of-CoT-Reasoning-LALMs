# core/prompt_strategies.py

"""
Prompt strategy builders and execution helpers for reasoning-based experiments.

This module keeps prompting logic centralized so experiments remain focused on
trial orchestration and result bookkeeping.
"""

from __future__ import annotations

from typing import Dict, List

DEFAULT_PROMPT_STRATEGY = "two_turn_sanitized_cot"
VALID_PROMPT_STRATEGIES = ("two_turn_sanitized_cot", "single_turn_explicit_letter")
DEPRECATED_STRATEGY_ALIASES = {
    "legacy_two_turn": "two_turn_sanitized_cot",
    "pooneh_single_turn": "single_turn_explicit_letter",
}


def get_prompt_strategy(config) -> str:
    """Resolve and validate prompt strategy from runtime config."""
    strategy = getattr(config, "PROMPT_STRATEGY", DEFAULT_PROMPT_STRATEGY)
    strategy = DEPRECATED_STRATEGY_ALIASES.get(strategy, strategy)
    if strategy not in VALID_PROMPT_STRATEGIES:
        raise ValueError(
            f"Invalid prompt strategy '{strategy}'. "
            f"Valid options: {VALID_PROMPT_STRATEGIES}"
        )
    return strategy


def _legacy_cot_prompt(question: str, choices: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": "Let's think step by step:"},
    ]


def _legacy_final_prompt(question: str, choices: str, sanitized_cot: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": sanitized_cot},
        {
            "role": "user",
            "content": (
                "Given the reasoning above, what is the single, most likely answer? "
                "Please respond with only the letter of the correct choice in "
                "parentheses, and nothing else."
            ),
        },
    ]


def _single_turn_explicit_letter_prompt(question: str, choices: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                f"audio\n\nQuestion: {question}\nChoices:\n{choices}\n\n"
                "Please think step-by-step about the audio and the choices provided. "
                "At the very end of your response, explicitly state your final prediction "
                "using only the single letter of the correct choice (e.g., A, B, C, or D)."
            ),
        }
    ]


def run_reasoning_trial(
    model,
    processor,
    model_utils,
    question: str,
    choices: str,
    audio_path: str,
    strategy: str,
) -> Dict[str, object]:
    """
    Execute the configured prompting flow and return normalized trial artifacts.

    Dispatch order:
    1. If model_utils has ``run_reasoning_inference`` → use it (model-specific default).
    2. Otherwise fall back to legacy strategies (backward compatibility only).
    """
    # --- Model-specific dispatch (preferred) ---
    if hasattr(model_utils, "run_reasoning_inference"):
        tokenizer = getattr(model_utils, "tokenizer", None)
        result = model_utils.run_reasoning_inference(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            question=question,
            choices_formatted=choices,
            audio_path=audio_path,
        )
        # Ensure we parse if not already parsed
        if "predicted_choice" not in result or result["predicted_choice"] is None:
            result["predicted_choice"] = model_utils.parse_answer(
                result.get("final_answer_raw", "")
            )
        return result

    # --- Legacy fallback strategies (backward compatibility) ---
    strategy = DEPRECATED_STRATEGY_ALIASES.get(strategy, strategy)

    if strategy == "two_turn_sanitized_cot":
        cot_prompt_messages = _legacy_cot_prompt(question, choices)
        generated_cot = model_utils.run_inference(
            model,
            processor,
            cot_prompt_messages,
            audio_path,
            max_new_tokens=768,
        )
        sanitized_cot = model_utils.sanitize_cot(generated_cot)

        final_answer_prompt_messages = _legacy_final_prompt(question, choices, sanitized_cot)
        final_answer_text = model_utils.run_inference(
            model,
            processor,
            final_answer_prompt_messages,
            audio_path,
            max_new_tokens=50,
        )

        return {
            "generated_cot": generated_cot,
            "sanitized_cot": sanitized_cot,
            "final_answer_raw": final_answer_text,
            "final_prompt_messages": final_answer_prompt_messages,
        }

    if strategy == "single_turn_explicit_letter":
        single_turn_messages = _single_turn_explicit_letter_prompt(question, choices)
        response_text = model_utils.run_inference(
            model,
            processor,
            single_turn_messages,
            audio_path,
            max_new_tokens=1024,
        )

        return {
            "generated_cot": response_text,
            "sanitized_cot": model_utils.sanitize_cot(response_text),
            "final_answer_raw": response_text,
            "final_prompt_messages": single_turn_messages,
        }

    raise ValueError(
        f"Unknown prompt strategy '{strategy}'. "
        f"Valid options: {VALID_PROMPT_STRATEGIES}. "
        f"Deprecated aliases still accepted: {tuple(DEPRECATED_STRATEGY_ALIASES.keys())}"
    )


# ---------------------------------------------------------------------------
# Centralized delegation helpers for intervention experiments.
#
# These let experiment modules call a single function without caring whether
# the underlying model backend has a specialised implementation or not.
# ---------------------------------------------------------------------------


def run_conditioned_trial(
    model,
    processor,
    tokenizer,
    model_utils,
    question: str,
    choices: str,
    audio_path: str,
    provided_reasoning: str,
) -> Dict[str, object]:
    """Run inference with pre-filled reasoning and ask for conclusion only.

    If the model backend exposes ``run_conditioned_inference`` (e.g. Qwen Omni
    with XML tags, AF3 HF with "Therefore, the answer is:" pattern) it is used
    directly.  Otherwise the legacy two-turn prompt is constructed and fed
    through ``run_inference``.
    """
    if hasattr(model_utils, "run_conditioned_inference"):
        return model_utils.run_conditioned_inference(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            question=question,
            choices_formatted=choices,
            audio_path=audio_path,
            provided_reasoning=provided_reasoning,
        )

    # Legacy fallback – works for qwen / salmonn / flamingo
    final_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": provided_reasoning},
        {
            "role": "user",
            "content": (
                "Given the reasoning above, what is the single, most likely answer? "
                "Please respond with only the letter of the correct choice in parentheses, and nothing else."
            ),
        },
    ]
    final_answer_text = model_utils.run_inference(
        model, processor, final_prompt_messages, audio_path,
        max_new_tokens=10,
    )
    return {
        "predicted_choice": model_utils.parse_answer(final_answer_text),
        "final_answer_raw": final_answer_text,
        "final_prompt_messages": final_prompt_messages,
    }


def run_no_reasoning_trial(
    model,
    processor,
    tokenizer,
    model_utils,
    question: str,
    choices: str,
    audio_path: str,
) -> Dict[str, object]:
    """Run inference with a no-reasoning prompt (direct answer, no CoT cue).

    If the model backend exposes ``run_no_reasoning_inference`` it is used.
    Otherwise a minimal single-turn prompt is constructed.
    """
    if hasattr(model_utils, "run_no_reasoning_inference"):
        return model_utils.run_no_reasoning_inference(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            question=question,
            choices_formatted=choices,
            audio_path=audio_path,
        )

    # Legacy fallback
    no_reasoning_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
    ]
    final_answer_text = model_utils.run_inference(
        model, processor, no_reasoning_messages, audio_path,
        max_new_tokens=50,
    )
    return {
        "predicted_choice": model_utils.parse_answer(final_answer_text),
        "final_answer_raw": final_answer_text,
        "final_prompt_messages": no_reasoning_messages,
    }
