# data_processing/merge_parallel_results.py

"""
GATHER step for parallel processing workflow with strict completeness checks.

By default, merge is refused when chunk verification fails (missing/incomplete).
Use --force-merge to override.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

try:
    # Works when executed as: python data_processing/merge_parallel_results.py
    from verify_parallel_completeness import (
        build_output_location,
        validate_parallel_completeness,
        print_report,
    )
except ImportError:
    # Works when executed from environments that expose project root imports.
    from data_processing.verify_parallel_completeness import (  # type: ignore
        build_output_location,
        validate_parallel_completeness,
        print_report,
    )


def merge_part_files(
    model: str,
    experiment: str,
    dataset: str,
    results_dir: str,
    restricted: bool,
    perturbation_source: str = "self",
    expected_parts: int | None = None,
    num_chains: int | None = None,
    expected_entries_per_sample: int | None = None,
    force_merge: bool = False,
    mask_type: str | None = None,
    mask_mode: str | None = None,
) -> int:
    """
    Validate parallel parts and merge to a single .jsonl file.

    Returns process exit code:
      0 on success,
      1 when validation fails without --force-merge.
    """
    print("\n--- Merging Parallel Results ---")
    print(f"  - Model: {model.upper()}")
    print(f"  - Experiment: {experiment.upper()}")
    print(f"  - Dataset: {dataset.upper()}")
    print(f"  - Run Mode: {'RESTRICTED' if restricted else 'FULL DATASET'}")
    if perturbation_source != "self":
        print(f"  - Perturbation Source: {perturbation_source.upper()}")
    if experiment == "audio_masking":
        print(f"  - Mask Type/Mode: {mask_type}/{mask_mode}")

    if expected_parts is None:
        print(
            "\nFATAL: --expected-parts is required for strict completeness validation.\n"
            "       Use --force-merge to bypass this check (not recommended)."
        )
        return 1

    if num_chains is None:
        print(
            "\nFATAL: --num-chains is required for completeness validation.\n"
            "       It must match the value used when running chunk jobs."
        )
        return 1

    report = validate_parallel_completeness(
        model=model,
        experiment=experiment,
        dataset=dataset,
        results_dir=results_dir,
        restricted=restricted,
        expected_parts=expected_parts,
        num_chains=num_chains,
        expected_entries_per_sample=expected_entries_per_sample,
        perturbation_source=perturbation_source,
        mask_type=mask_type,
        mask_mode=mask_mode,
    )
    print_report(report)

    all_pass = report["summary"]["all_pass"]
    if not all_pass and not force_merge:
        print(
            "\nFATAL: Refusing merge because one or more chunks are missing/incomplete.\n"
            "       Fix failed chunks and rerun, or use --force-merge to override."
        )
        return 1

    if not all_pass and force_merge:
        print(
            "\nWARNING: --force-merge is enabled. Proceeding with incomplete chunk set."
        )

    cfg: dict[str, Any] = report["config"]
    search_dir = cfg["search_dir"]
    base_filename = cfg["base_filename"]

    part_files: list[str] = []
    for row in report["parts"]:
        part_path = row["output_part_path"]
        if os.path.exists(part_path):
            part_files.append(part_path)

    if not part_files:
        print("\nFATAL: No output part files exist to merge.")
        return 1

    final_output_path = os.path.join(search_dir, f"{base_filename}.jsonl")
    os.makedirs(search_dir, exist_ok=True)

    print(f"\nMerging {len(part_files)} part files into: {final_output_path}")
    total_lines_written = 0
    with open(final_output_path, "w", encoding="utf-8") as f_out:
        for part_file in part_files:
            with open(part_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    f_out.write(line)
                    total_lines_written += 1

    print("\n--- Merge complete. ---")
    print(f"  - Total lines merged: {total_lines_written}")
    print(f"  - Final output: {final_output_path}")
    if not all_pass:
        print("  - NOTE: Merge used incomplete parts due to --force-merge.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge parallel output files into one JSONL with strict completeness "
            "checks (refuse by default on missing/incomplete chunks)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Model alias.")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset alias.")
    parser.add_argument(
        "--results-dir", type=str, default="./results", help="Root results directory."
    )
    parser.add_argument(
        "--restricted", action="store_true", help="Use restricted naming convention."
    )
    parser.add_argument(
        "--perturbation-source",
        type=str,
        default="self",
        choices=["self", "mistral"],
        help="Source of perturbations.",
    )
    parser.add_argument(
        "--expected-parts",
        type=int,
        default=None,
        help="Expected number of chunk files (required for strict validation).",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=None,
        help="Chain count used during chunk runs (required for validation).",
    )
    parser.add_argument(
        "--expected-entries-per-sample",
        type=int,
        default=None,
        help="Expected entries per baseline trial (audio_masking defaults to 11).",
    )
    parser.add_argument(
        "--force-merge",
        action="store_true",
        help="Override failed completeness checks and merge existing parts.",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default=None,
        choices=["silence", "noise"],
        help="Mask type (required for audio_masking).",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default=None,
        choices=["random", "start", "end", "scattered"],
        help="Mask mode (required for audio_masking).",
    )
    args = parser.parse_args()

    if args.experiment == "audio_masking":
        if not args.mask_type or not args.mask_mode:
            parser.error("audio_masking requires --mask-type and --mask-mode.")
    else:
        # Keep path behavior consistent for non-audio experiments.
        if args.mask_type or args.mask_mode:
            print(
                "WARNING: --mask-type/--mask-mode ignored for non-audio_masking experiments."
            )

    # Validate location resolution early for clearer errors.
    build_output_location(
        model=args.model,
        experiment=args.experiment,
        dataset=args.dataset,
        results_dir=args.results_dir,
        restricted=args.restricted,
        perturbation_source=args.perturbation_source,
        mask_type=args.mask_type,
        mask_mode=args.mask_mode,
    )

    return merge_part_files(
        model=args.model,
        experiment=args.experiment,
        dataset=args.dataset,
        results_dir=args.results_dir,
        restricted=args.restricted,
        perturbation_source=args.perturbation_source,
        expected_parts=args.expected_parts,
        num_chains=args.num_chains,
        expected_entries_per_sample=args.expected_entries_per_sample,
        force_merge=args.force_merge,
        mask_type=args.mask_type,
        mask_mode=args.mask_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
