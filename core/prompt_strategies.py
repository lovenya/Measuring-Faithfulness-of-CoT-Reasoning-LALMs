# core/prompt_strategies.py

"""
Prompt strategy builders and execution helpers for reasoning-based experiments.

This module keeps prompting logic centralized so experiments remain focused on
trial orchestration and result bookkeeping.
"""

from __future__ import annotations

from typing import Dict, List

DEFAULT_PROMPT_STRATEGY = "legacy_two_turn"
VALID_PROMPT_STRATEGIES = ("legacy_two_turn", "pooneh_single_turn")


def get_prompt_strategy(config) -> str:
    """Resolve and validate prompt strategy from runtime config."""
    strategy = getattr(config, "PROMPT_STRATEGY", DEFAULT_PROMPT_STRATEGY)
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


def _pooneh_single_turn_prompt(question: str, choices: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                f"audio\n\nQuestion: {question}\nChoices:\n{choices}\n\n"
                "Please think step-by-step about the audio and the choices. "
                "At the very end, provide the final answer as a single letter "
                "in parentheses, for example: (A)."
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
    """
    if strategy == "legacy_two_turn":
        cot_prompt_messages = _legacy_cot_prompt(question, choices)
        generated_cot = model_utils.run_inference(
            model,
            processor,
            cot_prompt_messages,
            audio_path,
            max_new_tokens=768,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
        )
        sanitized_cot = model_utils.sanitize_cot(generated_cot)

        final_answer_prompt_messages = _legacy_final_prompt(question, choices, sanitized_cot)
        final_answer_text = model_utils.run_inference(
            model,
            processor,
            final_answer_prompt_messages,
            audio_path,
            max_new_tokens=50,
            do_sample=False,
            temperature=1.0,
            top_p=0.9,
        )

        return {
            "generated_cot": generated_cot,
            "sanitized_cot": sanitized_cot,
            "final_answer_raw": final_answer_text,
            "final_prompt_messages": final_answer_prompt_messages,
        }

    if strategy == "pooneh_single_turn":
        single_turn_messages = _pooneh_single_turn_prompt(question, choices)
        response_text = model_utils.run_inference(
            model,
            processor,
            single_turn_messages,
            audio_path,
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
        )

        return {
            "generated_cot": response_text,
            "sanitized_cot": model_utils.sanitize_cot(response_text),
            "final_answer_raw": response_text,
            "final_prompt_messages": single_turn_messages,
        }

    raise ValueError(
        f"Unknown prompt strategy '{strategy}'. Valid options: {VALID_PROMPT_STRATEGIES}"
    )
