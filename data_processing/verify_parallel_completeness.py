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


def build_output_location(
    model: str,
    experiment: str,
    dataset: str,
    results_dir: str,
    restricted: bool,
    perturbation_source: str = "self",
    mask_type: str | None = None,
    mask_mode: str | None = None,
) -> tuple[str, str]:
    """
    Build (search_dir, base_filename) for parallel outputs.
    """
    if experiment == "audio_masking":
        if not mask_type or not mask_mode:
            raise ValueError(
                "audio_masking requires --mask-type and --mask-mode for path resolution."
            )
        search_dir = os.path.join(results_dir, model, experiment, mask_type, mask_mode)
    else:
        search_dir = os.path.join(results_dir, model, experiment)

    base_filename = f"{experiment}_{model}_{dataset}"
    if restricted:
        base_filename += "-restricted"

    if perturbation_source == "mistral":
        base_filename += "-mistral"

    if experiment == "audio_masking":
        base_filename += f"_{mask_type}_{mask_mode}"

    return search_dir, base_filename


def build_output_part_path(search_dir: str, base_filename: str, part_num: int) -> str:
    return os.path.join(search_dir, f"{base_filename}.part_{part_num}.jsonl")


def build_baseline_part_path(
    model: str,
    dataset: str,
    results_dir: str,
    restricted: bool,
    part_num: int,
) -> str:
    base = f"baseline_{model}_{dataset}"
    if restricted:
        base += "-restricted"
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


def validate_parallel_completeness(
    model: str,
    experiment: str,
    dataset: str,
    results_dir: str,
    restricted: bool,
    expected_parts: int,
    num_chains: int,
    expected_entries_per_sample: int | None = None,
    perturbation_source: str = "self",
    mask_type: str | None = None,
    mask_mode: str | None = None,
) -> dict[str, Any]:
    """
    Validate expected parts [1..expected_parts] and per-part line counts.
    """
    if expected_parts <= 0:
        raise ValueError("--expected-parts must be a positive integer.")
    if num_chains <= 0:
        raise ValueError("--num-chains must be a positive integer.")

    if experiment == "audio_masking" and expected_entries_per_sample is None:
        expected_entries_per_sample = 11

    search_dir, base_filename = build_output_location(
        model=model,
        experiment=experiment,
        dataset=dataset,
        results_dir=results_dir,
        restricted=restricted,
        perturbation_source=perturbation_source,
        mask_type=mask_type,
        mask_mode=mask_mode,
    )

    part_results: list[dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    missing_count = 0
    expected_total = 0
    actual_total = 0

    for part_num in range(1, expected_parts + 1):
        output_part_path = build_output_part_path(search_dir, base_filename, part_num)
        baseline_part_path = build_baseline_part_path(
            model=model,
            dataset=dataset,
            results_dir=results_dir,
            restricted=restricted,
            part_num=part_num,
        )

        row: dict[str, Any] = {
            "part": part_num,
            "output_part_path": output_part_path,
            "baseline_part_path": baseline_part_path,
            "baseline_trials": None,
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

        if expected_entries_per_sample is not None:
            expected_lines = baseline_trials * expected_entries_per_sample
            row["expected_lines"] = expected_lines
            expected_total += expected_lines

        if not os.path.exists(output_part_path):
            row["reason"] = "missing_output_part"
            missing_count += 1
            fail_count += 1
            part_results.append(row)
            continue

        actual_lines = count_lines(output_part_path)
        row["actual_lines"] = actual_lines
        actual_total += actual_lines

        if expected_entries_per_sample is None:
            row["status"] = "PASS"
            row["reason"] = "exists"
            pass_count += 1
        else:
            if actual_lines == row["expected_lines"]:
                row["status"] = "PASS"
                row["reason"] = "counts_match"
                pass_count += 1
            else:
                row["reason"] = "count_mismatch"
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
            "mask_type": mask_type,
            "mask_mode": mask_mode,
            "search_dir": search_dir,
            "base_filename": base_filename,
        },
        "parts": part_results,
        "summary": {
            "total_parts": expected_parts,
            "pass_parts": pass_count,
            "fail_parts": fail_count,
            "missing_parts": missing_count,
            "expected_total_lines": expected_total,
            "actual_total_lines": actual_total,
            "all_pass": fail_count == 0,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    cfg = report["config"]
    summary = report["summary"]
    part_rows = report["parts"]

    print("\n--- Parallel Completeness Verification ---")
    print(f"  - Model: {cfg['model'].upper()}")
    print(f"  - Experiment: {cfg['experiment'].upper()}")
    print(f"  - Dataset: {cfg['dataset'].upper()}")
    print(f"  - Run Mode: {'RESTRICTED' if cfg['restricted'] else 'FULL DATASET'}")
    if cfg["experiment"] == "audio_masking":
        print(f"  - Mask Type/Mode: {cfg['mask_type']}/{cfg['mask_mode']}")
    print(f"  - Expected Parts: {cfg['expected_parts']}")
    print(f"  - Num Chains: {cfg['num_chains']}")
    if cfg["expected_entries_per_sample"] is not None:
        print(f"  - Expected Entries/Sample: {cfg['expected_entries_per_sample']}")
    print(f"  - Search Dir: {cfg['search_dir']}")
    print(f"  - Base Filename: {cfg['base_filename']}")

    header = (
        f"\n{'part':>4}  {'baseline_trials':>15}  {'expected_lines':>14}  "
        f"{'actual_lines':>12}  {'status':>6}  reason"
    )
    print(header)
    print("-" * len(header))
    for row in part_rows:
        bt = row["baseline_trials"] if row["baseline_trials"] is not None else "-"
        ex = row["expected_lines"] if row["expected_lines"] is not None else "-"
        ac = row["actual_lines"] if row["actual_lines"] is not None else "-"
        print(
            f"{row['part']:>4}  {str(bt):>15}  {str(ex):>14}  "
            f"{str(ac):>12}  {row['status']:>6}  {row['reason']}"
        )

    print("\nSummary:")
    print(f"  - Total parts: {summary['total_parts']}")
    print(f"  - PASS: {summary['pass_parts']}")
    print(f"  - FAIL: {summary['fail_parts']}")
    print(f"  - Missing: {summary['missing_parts']}")
    if cfg["expected_entries_per_sample"] is not None:
        print(f"  - Expected lines (sum): {summary['expected_total_lines']}")
        print(f"  - Actual lines (sum):   {summary['actual_total_lines']}")
    print(f"  - Overall: {'PASS' if summary['all_pass'] else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify completeness of parallel output files by part index and "
            "(optionally) expected line counts."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--restricted", action="store_true")
    parser.add_argument("--expected-parts", type=int, required=True)
    parser.add_argument("--num-chains", type=int, required=True)
    parser.add_argument("--expected-entries-per-sample", type=int, default=None)
    parser.add_argument(
        "--perturbation-source",
        type=str,
        default="self",
        choices=["self", "mistral"],
    )
    parser.add_argument("--mask-type", type=str, default=None)
    parser.add_argument("--mask-mode", type=str, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report to stdout after the text report.",
    )
    args = parser.parse_args()

    if args.experiment == "audio_masking":
        if not args.mask_type or not args.mask_mode:
            parser.error("audio_masking requires --mask-type and --mask-mode.")

    report = validate_parallel_completeness(
        model=args.model,
        experiment=args.experiment,
        dataset=args.dataset,
        results_dir=args.results_dir,
        restricted=args.restricted,
        expected_parts=args.expected_parts,
        num_chains=args.num_chains,
        expected_entries_per_sample=args.expected_entries_per_sample,
        perturbation_source=args.perturbation_source,
        mask_type=args.mask_type,
        mask_mode=args.mask_mode,
    )

    print_report(report)
    if args.json:
        print("\nJSON report:")
        print(json.dumps(report, indent=2))

    return 0 if report["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
