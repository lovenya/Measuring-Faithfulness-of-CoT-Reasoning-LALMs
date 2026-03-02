#!/usr/bin/env python3
"""
Generate standalone SLURM submission scripts for:
  1. early_answering (flamingo_hf) — 5 datasets × rrg-ravanelm × 3.5h × 40GB MIG
  2. random_partial_filler_text (flamingo_hf) — 5 datasets × rrg-csubakan × 3.5h × 40GB MIG
  3. Mistral perturbations (mistakes + paraphrase) — 5 datasets × 2 modes × def-csubakan-ab rotated × 2.5h × full H100

Directory structure: submission_scripts/{model}/{experiment}/
"""

import os

DATASETS = ['mmar', 'sakura-animal', 'sakura-emotion', 'sakura-gender', 'sakura-language']

# ─── Template for AF3 CoT experiments (40GB MIG) ────────────────────────────
COT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --output=logs/{model}/{experiment}/%x-%j.out
#SBATCH --error=logs/{model}/{experiment}/%x-%j.err

set -euo pipefail
START_TIME=$(date +%s)
module load StdEnv/2023 cuda rust gcc arrow
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs
source af3_new_hf_env/bin/activate

while sleep 1200; do
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║              RESOURCE USAGE at $(date +%Y-%m-%d\\ %H:%M:%S)              ║"
  echo "╠══════════════════════════════════════════════════════════════════╣"
  free -h | awk 'NR==2 {{printf "║ RAM: Total=%s, Used=%s\\n", $2, $3}}'
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk -F',' '{{printf "║ %s | Util: %s%% | VRAM: %s/%s MiB | Temp: %s°C\\n", $1, $2, $3, $4, $5}}'
  echo "╚══════════════════════════════════════════════════════════════════╝"
done &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

{cmd}

END_TIME=$(date +%s)
echo "Duration: $(($END_TIME - $START_TIME))s"
"""

# ─── Template for Mistral perturbation gen (full H100) ──────────────────────
MISTRAL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/mistral_perturbations/%x-%j.out
#SBATCH --error=logs/mistral_perturbations/%x-%j.err

set -euo pipefail
START_TIME=$(date +%s)
module load StdEnv/2023 cuda rust gcc arrow opencv
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs
source mistral_env/bin/activate

while sleep 1200; do
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║              RESOURCE USAGE at $(date +%Y-%m-%d\\ %H:%M:%S)              ║"
  echo "╠══════════════════════════════════════════════════════════════════╣"
  free -h | awk 'NR==2 {{printf "║ RAM: Total=%s, Used=%s\\n", $2, $3}}'
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk -F',' '{{printf "║ %s | Util: %s%% | VRAM: %s/%s MiB | Temp: %s°C\\n", $1, $2, $3, $4, $5}}'
  echo "╚══════════════════════════════════════════════════════════════════╝"
done &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

{cmd}

END_TIME=$(date +%s)
echo "Duration: $(($END_TIME - $START_TIME))s"
"""

ACCOUNTS_ROTATION = ['rrg-ravanelm', 'rrg-csubakan', 'def-csubakan-ab']
rot_idx = 0

def make_short_dat(dataset):
    if dataset == 'mmar':
        return 'm'
    return dataset.replace("sakura-", "s-")[:3]

# ─── 1. Early Answering (flamingo_hf) ──────────────────────────────────────
for dataset in DATASETS:
    outdir = "submission_scripts/flamingo_hf/early_answering"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs("logs/flamingo_hf/early_answering", exist_ok=True)

    cmd = (
        f"python main.py \\\n"
        f"  --model flamingo_hf \\\n"
        f"  --experiment early_answering \\\n"
        f"  --dataset {dataset} \\\n"
        f"  --num-chains 1 \\\n"
        f"  --verbose"
    )
    content = COT_TEMPLATE.format(
        job_name=f"ea-af-{make_short_dat(dataset)}",
        account="rrg-ravanelm",
        time_limit="03:30:00",
        model="flamingo_hf",
        experiment="early_answering",
        cmd=cmd,
    )
    path = os.path.join(outdir, f"run_early_answering_{dataset}.sh")
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, 0o755)

# ─── 2. Random Partial Filler Text (flamingo_hf) ───────────────────────────
for dataset in DATASETS:
    outdir = "submission_scripts/flamingo_hf/random_partial_filler_text"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs("logs/flamingo_hf/random_partial_filler_text", exist_ok=True)

    cmd = (
        f"python main.py \\\n"
        f"  --model flamingo_hf \\\n"
        f"  --experiment random_partial_filler_text \\\n"
        f"  --dataset {dataset} \\\n"
        f"  --filler-type lorem \\\n"
        f"  --num-chains 1 \\\n"
        f"  --verbose"
    )
    content = COT_TEMPLATE.format(
        job_name=f"fill-af-{make_short_dat(dataset)}",
        account="rrg-csubakan",
        time_limit="03:30:00",
        model="flamingo_hf",
        experiment="random_partial_filler_text",
        cmd=cmd,
    )
    path = os.path.join(outdir, f"run_random_partial_filler_text_{dataset}.sh")
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, 0o755)

# ─── 3. Mistral Perturbations (mistakes + paraphrase) ──────────────────────
os.makedirs("submission_scripts/mistral_perturbations", exist_ok=True)
os.makedirs("logs/mistral_perturbations", exist_ok=True)
os.makedirs("results/mistral_perturbations", exist_ok=True)

for mode in ['mistakes', 'paraphrase']:
    for dataset in DATASETS:
        account = ACCOUNTS_ROTATION[rot_idx % len(ACCOUNTS_ROTATION)]
        rot_idx += 1

        short_mode = "mis" if mode == "mistakes" else "par"
        short_dat = make_short_dat(dataset)
        job_name = f"mp-{short_mode}-{short_dat}"

        baseline_path = f"results/flamingo_hf/baseline/baseline_flamingo_hf_{dataset}.jsonl"
        output_path = f"results/mistral_perturbations/flamingo_hf_{dataset}_{mode}.jsonl"

        cmd = (
            f"python scripts/generate_perturbations.py \\\n"
            f"  --baseline-results {baseline_path} \\\n"
            f"  --output {output_path} \\\n"
            f"  --mode {mode}"
        )
        content = MISTRAL_TEMPLATE.format(
            job_name=job_name,
            account=account,
            time_limit="02:30:00",
            cmd=cmd,
        )
        path = f"submission_scripts/mistral_perturbations/run_perturbation_{mode}_{dataset}.sh"
        with open(path, 'w') as f:
            f.write(content)
        os.chmod(path, 0o755)

print("Created submission scripts:")
print(f"  - 5 early_answering (flamingo_hf)")
print(f"  - 5 random_partial_filler_text (flamingo_hf)")
print(f"  - 10 mistral_perturbations (5 mistakes + 5 paraphrase)")
print(f"  Total: 20 scripts")
