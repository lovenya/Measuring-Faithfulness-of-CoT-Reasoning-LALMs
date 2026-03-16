# experiments/partial_filler_text.py

"""Unified partial filler text experiment.

This is the canonical, parameterized experiment for word-level CoT corruption.
Mode is controlled via config.PARTIAL_FILLER_MODE:
- start
- end
- random

Legacy experiment names are mapped in main.py/config.py:
- flipped_partial_filler_text -> partial_filler_text (mode=end)
- random_partial_filler_text  -> partial_filler_text (mode=random)
"""

import collections
import json
import logging
import os

from core.filler_text_utils import (
    create_word_level_masked_cot,
    load_lorem_token_pool,
    run_filler_trial,
)

EXPERIMENT_TYPE = "dependent"

VALID_MODES = {"start", "end", "random"}
POSITION_LABELS = {
    "start": "Start",
    "end": "End",
    "random": "Random",
}


def _resolve_mode(config) -> tuple[str, str]:
    mode = getattr(config, "PARTIAL_FILLER_MODE", "start")
    if mode not in VALID_MODES:
        logging.warning(
            "Unknown PARTIAL_FILLER_MODE '%s'. Falling back to 'start'.",
            mode,
        )
        mode = "start"
    return mode, POSITION_LABELS[mode]


def run(model, processor, tokenizer, model_utils, config):
    mode, position_label = _resolve_mode(config)

    baseline_results_path = config.BASELINE_RESULTS_PATH
    if not os.path.exists(baseline_results_path):
        logging.error(
            "Baseline results file not found at the path provided by main.py: '%s'",
            baseline_results_path,
        )
        return

    logging.info(
        "Reading baseline data for model '%s' from '%s'...",
        config.MODEL_ALIAS,
        baseline_results_path,
    )
    all_baseline_trials = []
    with open(baseline_results_path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            try:
                all_baseline_trials.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning("Skipping corrupted line %s in %s", line_num, baseline_results_path)
                continue

    if config.NUM_CHAINS_PER_QUESTION > 0:
        logging.info(
            "Filtering to include only the first %s chains for each question.",
            config.NUM_CHAINS_PER_QUESTION,
        )
        all_baseline_trials = [
            trial for trial in all_baseline_trials if trial["chain_id"] < config.NUM_CHAINS_PER_QUESTION
        ]

    if config.NUM_SAMPLES_TO_RUN > 0:
        trials_by_question = collections.defaultdict(list)
        for trial in all_baseline_trials:
            trials_by_question[trial["id"]].append(trial)
        unique_question_ids = list(trials_by_question.keys())[: config.NUM_SAMPLES_TO_RUN]
        samples_to_process = [trial for qid in unique_question_ids for trial in trials_by_question[qid]]
    else:
        samples_to_process = all_baseline_trials

    output_path = config.OUTPUT_PATH
    completed_steps = set()
    completed_chains = set()
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking for completed work...")
        progress_counts = collections.defaultdict(set)
        with open(output_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    step_key = (row["id"], row["chain_id"], int(row["percent_replaced"]))
                    completed_steps.add(step_key)
                    progress_counts[(row["id"], row["chain_id"])].add(int(row["percent_replaced"]))
                except (json.JSONDecodeError, KeyError):
                    continue

        for chain_key, steps in progress_counts.items():
            # 21 percent points from 0 to 100 inclusive in steps of 5.
            if len(steps) >= 21:
                completed_chains.add(chain_key)

    if completed_chains:
        logging.info("Found %s fully completed chains. They will be skipped.", len(completed_chains))

    lorem_pool = None
    if getattr(config, "FILLER_TYPE", "lorem") == "lorem":
        lorem_pool = load_lorem_token_pool()
        logging.info("Loaded lorem pool with %s unique words.", len(lorem_pool))

    logging.info(
        "Running WORD-LEVEL Partial Filler (%s) Experiment (mode=%s, model=%s): saving to %s",
        position_label,
        mode,
        config.MODEL_ALIAS.upper(),
        output_path,
    )

    skipped_trials_count = 0
    with open(output_path, "a", encoding="utf-8") as handle:
        for idx, baseline_trial in enumerate(samples_to_process):
            try:
                q_id, chain_id = baseline_trial["id"], baseline_trial["chain_id"]
                if (q_id, chain_id) in completed_chains:
                    continue

                if config.VERBOSE:
                    logging.info(
                        "Processing trial %s/%s: ID %s, Chain %s",
                        idx + 1,
                        len(samples_to_process),
                        q_id,
                        chain_id,
                    )

                choices_formatted = baseline_trial["choices"]
                sanitized_cot = baseline_trial["sanitized_cot"]

                for percentile in range(0, 101, 5):
                    step_key = (q_id, chain_id, percentile)
                    if step_key in completed_steps:
                        continue
                    if config.VERBOSE:
                        logging.info("  - Processing %s%% replacement...", percentile)

                    modified_cot = create_word_level_masked_cot(
                        sanitized_cot,
                        percentile,
                        mode=mode,
                        filler_type=config.FILLER_TYPE,
                        lorem_pool=lorem_pool,
                    )

                    trial_result = run_filler_trial(
                        model,
                        processor,
                        tokenizer,
                        model_utils,
                        baseline_trial["question"],
                        choices_formatted,
                        baseline_trial["audio_path"],
                        modified_cot,
                    )

                    baseline_final_choice = baseline_trial["predicted_choice"]
                    final_ordered_result = {
                        "id": q_id,
                        "chain_id": chain_id,
                        "percent_replaced": percentile,
                        "predicted_choice": trial_result["predicted_choice"],
                        "correct_choice": baseline_trial["correct_choice"],
                        "is_correct": trial_result["predicted_choice"] == baseline_trial["correct_choice"],
                        "corresponding_baseline_predicted_choice": baseline_final_choice,
                        "is_consistent_with_baseline": trial_result["predicted_choice"] == baseline_final_choice,
                        "modified_reasoning_chain": modified_cot,
                        "final_answer_raw": trial_result["final_answer_raw"],
                        "final_prompt_messages": trial_result["final_prompt_messages"],
                        "question": trial_result["question"],
                        "choices": trial_result["choices"],
                        "audio_path": trial_result["audio_path"],
                        "partial_filler_mode": mode,
                    }

                    handle.write(json.dumps(final_ordered_result, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed_steps.add(step_key)

            except Exception:
                skipped_trials_count += 1
                logging.exception(
                    "SKIPPING ENTIRE TRIAL due to unhandled error. ID: %s, Chain: %s",
                    baseline_trial.get("id", "N/A"),
                    baseline_trial.get("chain_id", "N/A"),
                )
                continue

    total_processed_in_this_run = len(samples_to_process) - len(completed_chains)
    logging.info(
        "--- WORD-LEVEL Partial Filler (%s, mode=%s) experiment for %s complete. ---",
        position_label,
        mode,
        config.MODEL_ALIAS.upper(),
    )
    logging.info("\n" + "=" * 25 + " RUN SUMMARY " + "=" * 25)
    logging.info("Total trials in dataset: %s", len(samples_to_process))
    logging.info("Trials already complete: %s", len(completed_chains))
    logging.info(
        "Trials processed in this run: %s",
        total_processed_in_this_run - skipped_trials_count,
    )
    logging.info("Skipped trials due to errors in this run: %s", skipped_trials_count)
    logging.info("Results saved to: %s", output_path)
    logging.info("=" * 65)
