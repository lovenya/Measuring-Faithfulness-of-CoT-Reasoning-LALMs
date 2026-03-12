"""SNR robustness experiment.

This independent experiment evaluates prediction stability under additive noise.
It iterates over dataset samples directly (no baseline JSONL dependency), runs
`NUM_CHAINS_PER_QUESTION` trials per sample, and writes real model outputs for
clean and noisy conditions.
"""

import json
import logging
import os
import time
from pathlib import Path

from core.prompt_strategies import get_prompt_strategy, run_reasoning_trial

EXPERIMENT_TYPE = "independent"

# SNR levels to test (dB), from cleanest to noisiest.
SNR_LEVELS = [20, 10, 5, 0, -5, -10]


def get_noisy_audio_path(original_audio_path: str, snr_db: int, dataset_name: str) -> str:
    """Construct path to pre-generated noisy audio."""
    original = Path(original_audio_path)
    stem = original.stem
    noisy_filename = f"{stem}_snr_{snr_db}db.wav"

    if dataset_name.startswith("sakura-"):
        track = dataset_name.replace("sakura-", "")
        noisy_dir = Path(f"data/sakura_noisy/{track}/audio")
    else:
        noisy_dir = Path(f"data/{dataset_name}_noisy/audio")

    return str(noisy_dir / noisy_filename)


def run(model, processor, tokenizer, model_utils, data_samples, config):
    output_path = config.OUTPUT_PATH
    dataset_name = config.DATASET_NAME
    prompt_strategy = get_prompt_strategy(config)

    logging.info("--- Running SNR Robustness Experiment ---")
    logging.info("  Model: %s", config.MODEL_ALIAS.upper())
    logging.info("  Dataset: %s", dataset_name)
    logging.info("  Prompt Strategy: %s", prompt_strategy)
    logging.info("  SNR Levels: %s dB", SNR_LEVELS)
    logging.info("  Output: %s", output_path)

    completed_trials = set()
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking completed work...")
        with open(output_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    completed_trials.add((row["id"], row["chain_id"], row["snr_db"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        logging.info("Found %s completed trials. They will be skipped.", len(completed_trials))

    samples_to_process = data_samples
    if config.NUM_SAMPLES_TO_RUN > 0:
        samples_to_process = samples_to_process[: config.NUM_SAMPLES_TO_RUN]

    num_chains = max(1, int(config.NUM_CHAINS_PER_QUESTION))
    total_samples = len(samples_to_process)
    skipped_count = 0
    new_count = 0
    error_count = 0
    start_time = time.time()

    logging.info("  Samples: %s", total_samples)
    logging.info(
        "  Expected entries per sample: %s (clean + %s noisy SNR levels) × %s chains",
        1 + len(SNR_LEVELS),
        len(SNR_LEVELS),
        num_chains,
    )

    with open(output_path, "a", encoding="utf-8") as handle:
        for sample_idx, sample in enumerate(samples_to_process):
            sample_id = sample["id"]
            question = sample["question"]
            choices_formatted = model_utils.format_choices_for_prompt(sample["choices"])
            original_audio_path = sample["audio_path"]
            correct_choice = chr(ord("A") + sample["answer_key"])

            for chain_id in range(num_chains):
                condition_audio_paths = [("clean", original_audio_path)]
                condition_audio_paths.extend(
                    (snr_db, get_noisy_audio_path(original_audio_path, snr_db, dataset_name))
                    for snr_db in SNR_LEVELS
                )

                for snr_db, audio_path in condition_audio_paths:
                    trial_key = (sample_id, chain_id, snr_db)
                    if trial_key in completed_trials:
                        skipped_count += 1
                        continue

                    if snr_db != "clean" and not os.path.exists(audio_path):
                        logging.warning(
                            "Noisy audio not found: %s (id=%s, snr=%s). Skipping.",
                            audio_path,
                            sample_id,
                            snr_db,
                        )
                        error_count += 1
                        continue

                    try:
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
                            "snr_db": snr_db,
                            "predicted_choice": predicted_choice,
                            "correct_choice": correct_choice,
                            "is_correct": predicted_choice == correct_choice,
                            "audio_path": audio_path,
                            "final_answer_raw": prompt_outputs.get("final_answer_raw"),
                            "generated_cot": prompt_outputs.get("generated_cot"),
                            "sanitized_cot": prompt_outputs.get("sanitized_cot"),
                            "final_prompt_messages": prompt_outputs.get("final_prompt_messages"),
                            "question": question,
                            "choices": choices_formatted,
                        }

                        for field in ("track", "source", "hop_type", "modality", "language"):
                            if field in sample:
                                result[field] = sample[field]

                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()
                        new_count += 1
                    except Exception as exc:
                        logging.error(
                            "Error on id=%s chain=%s snr=%s: %s",
                            sample_id,
                            chain_id,
                            snr_db,
                            exc,
                        )
                        error_count += 1

            if (sample_idx + 1) % 5 == 0 or sample_idx == total_samples - 1:
                elapsed = time.time() - start_time
                avg_per_sample = elapsed / (sample_idx + 1)
                remaining = avg_per_sample * (total_samples - sample_idx - 1)
                logging.info(
                    "Progress: %s/%s samples | New: %s | Skipped: %s | Errors: %s | "
                    "Elapsed: %.1fm | ETA: %.1fm",
                    sample_idx + 1,
                    total_samples,
                    new_count,
                    skipped_count,
                    error_count,
                    elapsed / 60,
                    remaining / 60,
                )

    total_elapsed = time.time() - start_time
    logging.info("--- SNR Robustness Experiment Complete ---")
    logging.info("  Total samples: %s", total_samples)
    logging.info("  New trials: %s", new_count)
    logging.info("  Skipped: %s", skipped_count)
    logging.info("  Errors: %s", error_count)
    logging.info("  Total time: %.1f minutes", total_elapsed / 60)
    logging.info("  Results: %s", output_path)
