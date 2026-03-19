#!/usr/bin/env python3
# data_processing/verify_parallel_completeness.py

"""
Validate completeness of parallel experiment output files.

Primary target is audio_masking runs where each baseline trial should produce
a fixed number of entries per sample (default: 11 for mask levels 0..100).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

FILLER_EXPERIMENTS = {
    "filler_text",
    "partial_filler_text",
    "random_partial_filler_text",
    "flipped_partial_filler_text",
}

PARTIAL_FILLER_MODE_BY_EXPERIMENT = {
    "partial_filler_text": None,
    "random_partial_filler_text": "random",
    "flipped_partial_filler_text": "end",
}


def _normalize_experiment_name(experiment: str) -> str:
    if experiment == "audio_masking":
        return "partial_audio_masking"
    if experiment in PARTIAL_FILLER_MODE_BY_EXPERIMENT:
        return "partial_filler_text"
    return experiment


def build_output_location(
    model: str,
    experiment: str,
    dataset: str,
    results_dir: str,
    restricted: bool,
    perturbation_source: str = "self",
    filler_type: str | None = None,
    mask_type: str | None = None,
    mask_mode: str | None = None,
    partial_filler_mode: str | None = None,
    external_llm: str | None = None,
) -> tuple[str, str]:
    """
    Build (search_dir, base_filename) for outputs.
    """
    canonical_experiment = _normalize_experiment_name(experiment)
    resolved_partial_filler_mode = PARTIAL_FILLER_MODE_BY_EXPERIMENT.get(experiment, partial_filler_mode)

    if external_llm:
        # e.g. results/external_llm_perturbations/mistral/qwen_omni/mmau/raw/mistakes.jsonl
        search_dir = os.path.join(results_dir, "external_llm_perturbations", external_llm, model, dataset, "raw")
        base_filename = experiment # mistakes or paraphrased
        if restricted:
            base_filename += "-restricted"
        return search_dir, base_filename

    if canonical_experiment == "partial_audio_masking":
        if not mask_type or not mask_mode:
            raise ValueError(
                "partial_audio_masking requires --mask-type and --mask-mode for path resolution."
            )
        search_dir = os.path.join(results_dir, model, canonical_experiment, mask_type, mask_mode)
    elif canonical_experiment == "partial_filler_text":
        if not resolved_partial_filler_mode:
            raise ValueError(
                "partial_filler_text requires --partial-filler-mode for path resolution."
            )
        search_dir = os.path.join(results_dir, model, canonical_experiment, resolved_partial_filler_mode)
    else:
        search_dir = os.path.join(results_dir, model, canonical_experiment)

    base_filename = f"{canonical_experiment}_{model}_{dataset}"
    if restricted:
        base_filename += "-restricted"

    if perturbation_source == "mistral":
        base_filename += "-mistral"

    if experiment in FILLER_EXPERIMENTS and filler_type:
        base_filename += f"-{filler_type}"

    if canonical_experiment == "partial_audio_masking":
        base_filename += f"_{mask_type}_{mask_mode}"
    elif canonical_experiment == "partial_filler_text":
        base_filename += f"_{resolved_partial_filler_mode}"

    return search_dir, base_filename


def build_output_part_path(search_dir: str, base_filename: str, part_num: int | None) -> str:
    if part_num is None:
        return os.path.join(search_dir, f"{base_filename}.jsonl")
    return os.path.join(search_dir, f"{base_filename}.part_{part_num}.jsonl")


def build_baseline_part_path(
    model: str,
    dataset: str,
    results_dir: str,
    restricted: bool,
    part_num: int | None,
) -> str:
    base = f"baseline_{model}_{dataset}"
    if restricted:
        base += "-restricted"
    if part_num is None:
        return os.path.join(results_dir, model, "baseline", f"{base}.jsonl")
    return os.path.join(results_dir, model, "baseline", f"{base}.part_{part_num}.jsonl")


def count_lines(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_baseline_trials(baseline_part_path: str, num_chains: int) -> tuple[int, int]:
    """
    Returns (trial_count, malformed_lines).
    A trial is counted when chain_id exists and chain_id < num_chains.
    """
    trial_count = 0
    malformed_lines = 0

    with open(baseline_part_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                chain_id = data.get("chain_id")
                if isinstance(chain_id, int) and chain_id < num_chains:
                    trial_count += 1
            except json.JSONDecodeError:
                malformed_lines += 1

    return trial_count, malformed_lines


def extract_ids(filepath: str, num_chains: int) -> set[tuple]:
    """
    Extract unique (id, chain_id) tuples from a JSONL file.
    Only includes entries where chain_id < num_chains.
    """
    ids: set[tuple] = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                sample_id = data.get("id")
                chain_id = data.get("chain_id")
                if sample_id is not None and isinstance(chain_id, int) and chain_id < num_chains:
                    ids.add((sample_id, chain_id))
            except json.JSONDecodeError:
                pass
    return ids


def validate_parallel_completeness(
    model: str,
    experiment: str,
    dataset: str,
    results_dir: str,
    restricted: bool,
    expected_parts: int | None,
    num_chains: int,
    expected_entries_per_sample: int | None = None,
    perturbation_source: str = "self",
    filler_type: str | None = None,
    mask_type: str | None = None,
    mask_mode: str | None = None,
    partial_filler_mode: str | None = None,
    external_llm: str | None = None,
) -> dict[str, Any]:
    """
    Validate output files. Check parts [1..expected_parts] if provided, else check single file.
    """
    if expected_parts is not None and expected_parts <= 0:
        raise ValueError("--expected-parts must be a positive integer.")
    if num_chains <= 0:
        raise ValueError("--num-chains must be a positive integer.")

    if _normalize_experiment_name(experiment) == "partial_audio_masking" and expected_entries_per_sample is None:
        expected_entries_per_sample = 11

    search_dir, base_filename = build_output_location(
        model=model,
        experiment=experiment,
        dataset=dataset,
        results_dir=results_dir,
        restricted=restricted,
        perturbation_source=perturbation_source,
        filler_type=filler_type,
        mask_type=mask_type,
        mask_mode=mask_mode,
        partial_filler_mode=partial_filler_mode,
        external_llm=external_llm,
    )

    part_results: list[dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    missing_count = 0
    expected_total = 0
    actual_total = 0

    total_baseline_ids = 0
    total_output_ids = 0
    total_missing_ids = 0

    parts_to_check = [None] if expected_parts is None else list(range(1, expected_parts + 1))

    for part_num in parts_to_check:
        output_part_path = build_output_part_path(search_dir, base_filename, part_num)
        baseline_part_path = build_baseline_part_path(
            model=model,
            dataset=dataset,
            results_dir=results_dir,
            restricted=restricted,
            part_num=part_num,
        )

        row: dict[str, Any] = {
            "part": part_num if part_num is not None else "ALL",
            "output_part_path": output_part_path,
            "baseline_part_path": baseline_part_path,
            "baseline_trials": None,
            "baseline_ids": 0,
            "output_ids": 0,
            "missing_ids": 0,
            "missing_id_list": [],
            "expected_lines": None,
            "actual_lines": None,
            "status": "FAIL",
            "reason": "",
        }

        if not os.path.exists(baseline_part_path):
            row["reason"] = "missing_baseline_part"
            missing_count += 1
            fail_count += 1
            part_results.append(row)
            continue

        baseline_trials, malformed_baseline = count_baseline_trials(
            baseline_part_path, num_chains
        )
        row["baseline_trials"] = baseline_trials

        if malformed_baseline > 0:
            row["reason"] = f"malformed_baseline_lines={malformed_baseline}"
            fail_count += 1
            part_results.append(row)
            continue

        # Extract baseline IDs for this part
        baseline_id_set = extract_ids(baseline_part_path, num_chains)
        row["baseline_ids"] = len(baseline_id_set)
        total_baseline_ids += len(baseline_id_set)

        if expected_entries_per_sample is not None:
            expected_lines = baseline_trials * expected_entries_per_sample
            row["expected_lines"] = expected_lines
            expected_total += expected_lines

        if not os.path.exists(output_part_path):
            row["reason"] = "missing_output_part"
            row["missing_ids"] = len(baseline_id_set)
            row["missing_id_list"] = sorted(baseline_id_set)
            total_missing_ids += len(baseline_id_set)
            missing_count += 1
            fail_count += 1
            part_results.append(row)
            continue

        actual_lines = count_lines(output_part_path)
        row["actual_lines"] = actual_lines
        actual_total += actual_lines

        # ID coverage check (always-on, determines pass/fail)
        output_id_set = extract_ids(output_part_path, num_chains)
        row["output_ids"] = len(output_id_set)
        total_output_ids += len(output_id_set)

        missing = baseline_id_set - output_id_set
        row["missing_ids"] = len(missing)
        total_missing_ids += len(missing)
        if missing:
            row["missing_id_list"] = sorted(missing)

        # Supplementary line-count check (advisory only)
        line_count_note = ""
        if expected_entries_per_sample is not None and actual_lines != row["expected_lines"]:
            line_count_note = f" (line_count: {actual_lines}/{row['expected_lines']})"

        if len(missing) == 0:
            row["status"] = "PASS"
            row["reason"] = f"all_ids_covered{line_count_note}"
            pass_count += 1
        else:
            row["reason"] = f"missing_{len(missing)}_ids{line_count_note}"
            fail_count += 1

        part_results.append(row)

    return {
        "config": {
            "model": model,
            "experiment": experiment,
            "dataset": dataset,
            "results_dir": results_dir,
            "restricted": restricted,
            "expected_parts": expected_parts,
            "num_chains": num_chains,
            "expected_entries_per_sample": expected_entries_per_sample,
            "perturbation_source": perturbation_source,
            "filler_type": filler_type,
            "mask_type": mask_type,
            "mask_mode": mask_mode,
            "partial_filler_mode": partial_filler_mode,
            "search_dir": search_dir,
            "base_filename": base_filename,
        },
        "parts": part_results,
        "summary": {
            "total_parts": expected_parts,
            "pass_parts": pass_count,
            "fail_parts": fail_count,
            "missing_parts": missing_count,
            "total_baseline_ids": total_baseline_ids,
            "total_output_ids": total_output_ids,
            "total_missing_ids": total_missing_ids,
            "expected_total_lines": expected_total,
            "actual_total_lines": actual_total,
            "all_pass": fail_count == 0,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    cfg = report["config"]
    summary = report["summary"]
    part_rows = report["parts"]

    print("\n--- Experiment Completeness Verification ---")
    print(f"  - Model: {cfg['model'].upper()}")
    print(f"  - Experiment: {cfg['experiment'].upper()}")
    print(f"  - Dataset: {cfg['dataset'].upper()}")
    print(f"  - Run Mode: {'RESTRICTED' if cfg['restricted'] else 'FULL DATASET'}")
    if _normalize_experiment_name(cfg["experiment"]) == "partial_audio_masking":
        print(f"  - Mask Type/Mode: {cfg['mask_type']}/{cfg['mask_mode']}")
    if _normalize_experiment_name(cfg["experiment"]) == "partial_filler_text":
        print(f"  - Partial Filler Mode: {cfg['partial_filler_mode']}")
    print(f"  - Expected Parts: {cfg['expected_parts'] if cfg['expected_parts'] is not None else 'N/A (Single File)'}")
    print(f"  - Num Chains: {cfg['num_chains']}")
    if cfg["expected_entries_per_sample"] is not None:
        print(f"  - Expected Entries/Sample: {cfg['expected_entries_per_sample']}")
    if cfg.get("filler_type"):
        print(f"  - Filler Type: {cfg['filler_type']}")
    print(f"  - Search Dir: {cfg['search_dir']}")
    print(f"  - Base Filename: {cfg['base_filename']}")

    # Determine if we have line-count info to show
    has_line_counts = cfg["expected_entries_per_sample"] is not None

    if has_line_counts:
        header = (
            f"\n{'part':>4}  {'baseline_ids':>12}  {'output_ids':>10}  {'missing':>7}  "
            f"{'exp_lines':>9}  {'act_lines':>9}  {'status':>6}  reason"
        )
    else:
        header = (
            f"\n{'part':>4}  {'baseline_ids':>12}  {'output_ids':>10}  {'missing':>7}  "
            f"{'act_lines':>9}  {'status':>6}  reason"
        )
    print(header)
    print("-" * len(header))

    for row in part_rows:
        bi = row.get("baseline_ids", "-")
        oi = row.get("output_ids", "-")
        mi = row.get("missing_ids", "-")
        ac = row["actual_lines"] if row["actual_lines"] is not None else "-"
        if has_line_counts:
            ex = row["expected_lines"] if row["expected_lines"] is not None else "-"
            print(
                f"{row['part']:>4}  {str(bi):>12}  {str(oi):>10}  {str(mi):>7}  "
                f"{str(ex):>9}  {str(ac):>9}  {row['status']:>6}  {row['reason']}"
            )
        else:
            print(
                f"{row['part']:>4}  {str(bi):>12}  {str(oi):>10}  {str(mi):>7}  "
                f"{str(ac):>9}  {row['status']:>6}  {row['reason']}"
            )

    # Show missing IDs if any
    for row in part_rows:
        if row.get("missing_id_list"):
            print(f"\n  Part {row['part']} missing IDs (first 10):")
            for mid in row["missing_id_list"][:10]:
                print(f"    id={mid[0]}, chain_id={mid[1]}")
            if len(row["missing_id_list"]) > 10:
                print(f"    ... and {len(row['missing_id_list']) - 10} more")

    print("\nSummary:")
    print(f"  - Total parts: {summary['total_parts']}")
    print(f"  - PASS: {summary['pass_parts']}")
    print(f"  - FAIL: {summary['fail_parts']}")
    print(f"  - Missing: {summary['missing_parts']}")
    print(f"  - ID Coverage: {summary['total_output_ids']}/{summary['total_baseline_ids']} "
          f"({summary['total_baseline_ids'] - summary['total_missing_ids']}/{summary['total_baseline_ids']} covered)")
    if has_line_counts:
        print(f"  - Expected lines (sum): {summary['expected_total_lines']}")
        print(f"  - Actual lines (sum):   {summary['actual_total_lines']}")
    print(f"  - Overall: {'PASS' if summary['all_pass'] else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify completeness of experiment output files (both single file and parallel parts)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--restricted", action="store_true")
    parser.add_argument(
        "--expected-parts",
        type=int,
        default=None,
        help="Number of parts if parallel. If omitted, checks standard single output location.",
    )
    parser.add_argument("--num-chains", type=int, required=True)
    parser.add_argument("--expected-entries-per-sample", type=int, default=None)
    parser.add_argument(
        "--perturbation-source",
        type=str,
        default="self",
        choices=["self", "mistral"],
    )
    parser.add_argument(
        "--filler-type",
        type=str,
        default=None,
        choices=["dots", "lorem"],
        help="Filler type (mandatory for filler text experiments): 'dots' or 'lorem'.",
    )
    parser.add_argument("--mask-type", type=str, default=None)
    parser.add_argument("--mask-mode", type=str, default=None)
    parser.add_argument(
        "--partial-filler-mode",
        type=str,
        default=None,
        choices=["start", "end", "random"],
    )
    parser.add_argument(
        "--external-llm",
        type=str,
        default=None,
        help="External LLM name (e.g. mistral) if verifying external generations. "
             "If provided, experiment name should be 'mistakes' or 'paraphrased' "
             "and output paths will point to the raw perturbation directories."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report to stdout after the text report.",
    )
    args = parser.parse_args()

    if _normalize_experiment_name(args.experiment) == "partial_audio_masking":
        if not args.mask_type or not args.mask_mode:
            parser.error("partial_audio_masking requires --mask-type and --mask-mode.")

    if _normalize_experiment_name(args.experiment) == "partial_filler_text":
        resolved_mode = PARTIAL_FILLER_MODE_BY_EXPERIMENT.get(args.experiment, args.partial_filler_mode)
        if not resolved_mode:
            parser.error("partial_filler_text requires --partial-filler-mode.")

    # Filler type is mandatory for filler text experiments
    if "filler" in args.experiment and not args.filler_type:
        parser.error(
            f"--filler-type is mandatory for filler text experiments "
            f"(experiment='{args.experiment}'). Use --filler-type dots or --filler-type lorem."
        )

    datasets = ["mmar", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language", "mmau", "jasco"] if args.dataset == "all" else [args.dataset]

    all_pass = True
    for ds in datasets:
        if args.dataset == "all":
            print(f"\n{'='*60}\nVERIFYING DATASET: {ds.upper()}\n{'='*60}")
        
        try:
            report = validate_parallel_completeness(
                model=args.model,
                experiment=args.experiment,
                dataset=ds,
                results_dir=args.results_dir,
                restricted=args.restricted,
                expected_parts=args.expected_parts,
                num_chains=args.num_chains,
                expected_entries_per_sample=args.expected_entries_per_sample,
                perturbation_source=args.perturbation_source,
                filler_type=args.filler_type,
                mask_type=args.mask_type,
                mask_mode=args.mask_mode,
                partial_filler_mode=args.partial_filler_mode,
                external_llm=args.external_llm,
            )

            print_report(report)
            if args.json:
                print("\nJSON report:")
                print(json.dumps(report, indent=2))
            
            if not report["summary"]["all_pass"]:
                all_pass = False
        except Exception as e:
            print(f"\nError verifying dataset {ds}: {e}")
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
