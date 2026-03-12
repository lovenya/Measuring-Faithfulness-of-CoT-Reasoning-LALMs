#!/usr/bin/env python3
"""Evaluate SNR robustness results.

Consistency is computed at analysis time by joining SNR outputs with baseline
predictions on (id, chain_id).
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SNR_LEVELS = ["clean", 20, 10, 5, 0, -5, -10]
ALL_DATASETS = ["mmar", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]


def load_results(results_dir: str, model: str, dataset: str) -> list[dict]:
    filepath = os.path.join(results_dir, model, "snr_robustness", f"snr_robustness_{model}_{dataset}.jsonl")
    if not os.path.exists(filepath):
        return []

    entries = []
    with open(filepath, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_baseline_lookup(results_dir: str, model: str, dataset: str) -> dict[tuple[str, str], str | None]:
    filepath = os.path.join(results_dir, model, "baseline", f"baseline_{model}_{dataset}.jsonl")
    if not os.path.exists(filepath):
        return {}

    lookup = {}
    with open(filepath, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            lookup[(str(row.get("id")), str(row.get("chain_id")))] = row.get("predicted_choice")
    return lookup


def get_hop_type(entry: dict) -> str | None:
    hop = entry.get("hop_type")
    if hop:
        return hop
    sample_id = entry.get("id", "")
    if str(sample_id).endswith("_single"):
        return "single"
    if str(sample_id).endswith("_multi"):
        return "multi"
    return None


def filter_by_hop_type(entries: list[dict], hop_type: str) -> list[dict]:
    if hop_type == "merged":
        return entries
    return [entry for entry in entries if get_hop_type(entry) == hop_type]


def compute_metrics(entries: list[dict], baseline_lookup: dict[tuple[str, str], str | None]) -> tuple[dict, int]:
    """Return metrics per SNR level and total missing baseline joins."""
    by_snr = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "consistency_total": 0,
            "consistent": 0,
        }
    )
    missing_baseline_count = 0

    for entry in entries:
        snr = entry.get("snr_db")
        if snr not in SNR_LEVELS:
            continue

        by_snr[snr]["total"] += 1
        if entry.get("is_correct"):
            by_snr[snr]["correct"] += 1

        baseline_pred = baseline_lookup.get((str(entry.get("id")), str(entry.get("chain_id"))))
        if baseline_pred is None:
            missing_baseline_count += 1
            continue

        by_snr[snr]["consistency_total"] += 1
        if entry.get("predicted_choice") == baseline_pred:
            by_snr[snr]["consistent"] += 1

    metrics = {}
    for snr in SNR_LEVELS:
        data = by_snr.get(
            snr,
            {
                "total": 0,
                "correct": 0,
                "consistency_total": 0,
                "consistent": 0,
            },
        )
        total = data["total"]
        if total == 0:
            metrics[snr] = None
            continue

        consistency_total = data["consistency_total"]
        consistency_pct = None
        if consistency_total > 0:
            consistency_pct = data["consistent"] / consistency_total * 100

        metrics[snr] = {
            "total": total,
            "correct": data["correct"],
            "accuracy": data["correct"] / total * 100,
            "consistency_total": consistency_total,
            "consistent": data["consistent"],
            "consistency": consistency_pct,
        }

    return metrics, missing_baseline_count


def run_analysis(args, hop_type_label: str, hop_filter: str) -> list[dict]:
    print(f"\n{'=' * 78}")
    print(f"SNR Robustness Analysis — {args.model.upper()} — Hop: {hop_type_label.upper()}")
    print(f"{'=' * 78}")

    all_rows = []

    for dataset in args.datasets:
        entries = load_results(args.results_dir, args.model, dataset)

        if dataset.startswith("sakura-") and hop_filter != "merged":
            entries = filter_by_hop_type(entries, hop_filter)

        if not entries:
            print(f"\n--- {dataset.upper()} ---")
            print("  No results found.")
            continue

        baseline_lookup = load_baseline_lookup(args.results_dir, args.model, dataset)
        metrics, missing_count = compute_metrics(entries, baseline_lookup)

        print(f"\n--- {dataset.upper()} ({len(entries)} entries) ---")
        print(f"  {'SNR':>8} {'Total':>7} {'Acc':>8} {'Consist':>13} {'Consist %':>10}")
        print(f"  {'-' * 60}")

        for snr in SNR_LEVELS:
            stat = metrics.get(snr)
            if not stat:
                continue

            snr_label = "clean" if snr == "clean" else f"{snr}dB"
            consist_den = stat["consistency_total"]
            if consist_den > 0:
                consist_str = f"{stat['consistent']}/{consist_den}"
                consist_pct = f"{stat['consistency']:.1f}%"
            else:
                consist_str = "n/a"
                consist_pct = "n/a"

            print(
                f"  {snr_label:>8} {stat['total']:>7} {stat['accuracy']:>7.1f}% "
                f"{consist_str:>13} {consist_pct:>10}"
            )

            all_rows.append(
                {
                    "hop_type": hop_type_label,
                    "model": args.model,
                    "dataset": dataset,
                    "snr_db": snr,
                    "total": stat["total"],
                    "correct": stat["correct"],
                    "accuracy": round(stat["accuracy"], 2),
                    "consistency_total": stat["consistency_total"],
                    "consistent": stat["consistent"],
                    "consistency": round(stat["consistency"], 2) if stat["consistency"] is not None else None,
                    "missing_baseline_joins": missing_count,
                }
            )

        if missing_count > 0:
            print(
                f"  WARNING: {missing_count}/{len(entries)} entries missing baseline join; "
                "excluded from consistency denominator."
            )

    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SNR robustness experiment results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Model alias (e.g., 'qwen_audio_2').")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, help="Datasets to evaluate (default: all).")
    parser.add_argument("--results-dir", type=str, default="results", help="Base results directory.")
    parser.add_argument(
        "--hop-type",
        type=str,
        default="merged",
        choices=["merged", "single", "multi", "all"],
        help=(
            "Hop type filter for Sakura datasets.\n"
            "  merged = all data together (default)\n"
            "  single = single-hop only\n"
            "  multi  = multi-hop only\n"
            "  all    = run both single and multi separately"
        ),
    )
    parser.add_argument("--output-csv", type=str, default=None, help="Optional: save summary to CSV.")
    args = parser.parse_args()

    hop_runs = [("single", "single"), ("multi", "multi")] if args.hop_type == "all" else [(args.hop_type, args.hop_type)]

    all_rows = []
    for hop_label, hop_filter in hop_runs:
        all_rows.extend(run_analysis(args, hop_label, hop_filter))

    if args.output_csv and all_rows:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSummary saved to: {args.output_csv}")

    print(f"\n{'=' * 78}")


if __name__ == "__main__":
    main()
