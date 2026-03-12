"""Partial audio masking experiment.

This independent experiment measures how predictions change as increasing portions
of the input audio are masked.

Canonical output path:
  results/{model}/partial_audio_masking/{mask_type}/{mask_mode}/
  partial_audio_masking_{model}_{dataset}_{mask_type}_{mask_mode}.jsonl

Runtime behavior:
- Iterates over dataset rows directly (no baseline JSONL dependency).
- For each sample, runs `NUM_CHAINS_PER_QUESTION` trials (chain_id 0..N-1).
- Runs real inference for all masking levels, including 0%.
"""

import json
import logging
import os
from pathlib import Path

from core.prompt_strategies import get_prompt_strategy, run_reasoning_trial

EXPERIMENT_TYPE = "independent"

MASK_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _masked_root(dataset_name: str, mask_type: str, mask_mode: str) -> Path:
    if dataset_name.startswith("sakura-"):
        subdataset = dataset_name.replace("sakura-", "")
        return Path(f"data/sakura/{subdataset}_masked/{mask_type}_{mask_mode}")
    return Path(f"data/{dataset_name}_masked/{mask_type}_{mask_mode}")


def _load_level_lookup(root_dir: Path, levels: list[int]) -> dict[int, dict[str, str]]:
    """Load id->audio_path lookups for every masking level."""
    lookups: dict[int, dict[str, str]] = {}

    for level in levels:
        if level == 0:
            continue

        level_dir = root_dir / str(level)
        jsonl_files = list(level_dir.glob("*_standardized.jsonl"))
        if not jsonl_files:
            logging.warning("No JSONL found for %s%% level in %s", level, level_dir)
            lookups[level] = {}
            continue

        level_lookup: dict[str, str] = {}
        with open(jsonl_files[0], "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = str(row.get("id"))
                audio_path = row.get("audio_path")
                if sample_id and audio_path:
                    level_lookup[sample_id] = audio_path
        lookups[level] = level_lookup

    return lookups


def run(model, processor, tokenizer, model_utils, data_samples, config):
    output_path = config.OUTPUT_PATH
    mask_type = getattr(config, "MASK_TYPE", "silence")
    mask_mode = getattr(config, "MASK_MODE", "scattered")
    prompt_strategy = get_prompt_strategy(config)

    logging.info("--- Running Partial Audio Masking Experiment ---")
    logging.info("  Model: %s", config.MODEL_ALIAS.upper())
    logging.info("  Dataset: %s", config.DATASET_NAME)
    logging.info("  Prompt Strategy: %s", prompt_strategy)
    logging.info("  Mask Type/Mode: %s/%s", mask_type, mask_mode)
    logging.info("  Levels: %s", MASK_LEVELS)
    logging.info("  Output: %s", output_path)

    root_dir = _masked_root(config.DATASET_NAME, mask_type, mask_mode)
    if not root_dir.exists():
        logging.error("Masked dataset directory not found: %s", root_dir)
        logging.error("Generate masked data first using data_processing/mask_audio_dataset.py")
        return

    level_lookup = _load_level_lookup(root_dir, MASK_LEVELS)

    completed_trials = set()
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking completed trials...")
        with open(output_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    completed_trials.add((row["id"], row["chain_id"], row["mask_percent"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        logging.info("Found %s completed trials. They will be skipped.", len(completed_trials))

    samples_to_process = data_samples
    if config.NUM_SAMPLES_TO_RUN > 0:
        samples_to_process = samples_to_process[: config.NUM_SAMPLES_TO_RUN]

    num_chains = max(1, int(config.NUM_CHAINS_PER_QUESTION))
    skipped_count = 0
    error_count = 0
    total_samples = len(samples_to_process)

    with open(output_path, "a", encoding="utf-8") as handle:
        for sample_idx, sample in enumerate(samples_to_process):
            sample_id = sample["id"]
            question = sample["question"]
            choices_formatted = model_utils.format_choices_for_prompt(sample["choices"])
            correct_choice = chr(ord("A") + sample["answer_key"])

            for chain_id in range(num_chains):
                for level in MASK_LEVELS:
                    trial_key = (sample_id, chain_id, level)
                    if trial_key in completed_trials:
                        skipped_count += 1
                        continue

                    if level == 0:
                        audio_path = sample["audio_path"]
                    else:
                        audio_path = level_lookup.get(level, {}).get(str(sample_id))

                    if not audio_path:
                        logging.warning(
                            "Missing masked audio for sample '%s' at level %s%%; skipping.",
                            sample_id,
                            level,
                        )
                        error_count += 1
                        continue

                    try:
                        if config.VERBOSE:
                            logging.info(
                                "Sample %s/%s | id=%s | chain=%s | level=%s%%",
                                sample_idx + 1,
                                total_samples,
                                sample_id,
                                chain_id,
                                level,
                            )

                        prompt_outputs = run_reasoning_trial(
                            model=model,
                            processor=processor,
                            model_utils=model_utils,
                            question=question,
                            choices=choices_formatted,
                            audio_path=audio_path,
                            strategy=prompt_strategy,
                        )

                        predicted_choice = model_utils.parse_answer(
                            prompt_outputs.get("final_answer_raw", "")
                        )

                        result = {
                            "id": sample_id,
                            "chain_id": chain_id,
                            "mask_percent": level,
                            "mask_type": mask_type,
                            "mask_mode": mask_mode,
                            "predicted_choice": predicted_choice,
                            "correct_choice": correct_choice,
                            "is_correct": predicted_choice == correct_choice,
                            "final_answer_raw": prompt_outputs.get("final_answer_raw"),
                            "final_prompt_messages": prompt_outputs.get("final_prompt_messages"),
                            "question": question,
                            "choices": choices_formatted,
                            "audio_path": audio_path,
                            "generated_cot": prompt_outputs.get("generated_cot"),
                            "sanitized_cot": prompt_outputs.get("sanitized_cot"),
                        }

                        # Keep optional metadata when present for downstream filtering.
                        for field in ("track", "source", "hop_type", "modality", "language"):
                            if field in sample:
                                result[field] = sample[field]

                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()

                    except Exception:
                        error_count += 1
                        logging.exception(
                            "SKIPPING trial due to error. id=%s chain=%s level=%s%%",
                            sample_id,
                            chain_id,
                            level,
                        )

    logging.info("--- Partial Audio Masking Complete ---")
    logging.info("  Samples processed: %s", total_samples)
    logging.info("  Skipped completed trials: %s", skipped_count)
    logging.info("  Failed trials: %s", error_count)
    logging.info("  Results: %s", output_path)
