---
description: Generate and submit SLURM experiment scripts with proper account rotation, logging, and resource monitoring
---

# Submit Experiments Workflow

## Pre-Flight Checklist

Before generating scripts, confirm with the user:

1. **Which experiments** (e.g., `baseline`, `early_answering`, `random_partial_filler_text`, `adding_mistakes`, `paraphrasing`)
2. **Which models** (e.g., `flamingo_hf`, `qwen_omni`)
3. **Which datasets** (e.g., `mmar`, `sakura-animal`, `sakura-emotion`, `sakura-gender`, `sakura-language`)
4. **Experiment-specific flags**:
   - `baseline`: `--num-chains N`
   - `early_answering`: `--num-chains 1 --restricted`
   - `random_partial_filler_text`: `--filler-type lorem --num-chains 1` (+ `--part P --total-parts T` if parallel)
   - `adding_mistakes` / `paraphrasing`: `--num-chains 1` (+ `--use-external-perturbations` if Mistral)
   - `audio_masking`: `--mask-type TYPE --mask-mode MODE --num-chains 1`
5. **Number of parallel chunks** (if any)
6. **`--num-samples`** (for demo/test runs, e.g., `--num-samples 100`)

## Account Rotation Strategy

| Priority | Account           | Notes                |
| -------- | ----------------- | -------------------- |
| 1        | `rrg-ravanelm`    | Most quota           |
| 2        | `rrg-csubakan`    | Second most          |
| 3        | `def-csubakan-ab` | Use when others busy |

Round-robin across accounts when submitting multiple scripts.

## Time Limits & GPU Per Model

| Model                   | Time Limit | GPU                                          |
| ----------------------- | ---------- | -------------------------------------------- |
| `flamingo_hf`           | `04:00:00` | `nvidia_h100_80gb_hbm3_3g.40gb:1` (40GB MIG) |
| `qwen_omni`             | `12:00:00` | `nvidia_h100_80gb_hbm3_3g.40gb:1` (40GB MIG) |
| Mistral (perturbations) | `04:00:00` | `h100:1` (full 80GB)                         |

## Environment Activation

Refer to the `/activate_env` workflow for the full list. In SLURM scripts:

| Task                                  | Module Loading                                       | Env Activation                       |
| ------------------------------------- | ---------------------------------------------------- | ------------------------------------ |
| **CoT experiments** (flamingo_hf)     | `module load StdEnv/2023 cuda rust gcc arrow`        | `source af3_new_hf_env/bin/activate` |
| **CoT experiments** (qwen_omni)       | `module load StdEnv/2023 cuda rust gcc arrow`        | `source qwen_omni_env/bin/activate`  |
| **Perturbation generation** (Mistral) | `module load StdEnv/2023 cuda rust gcc arrow opencv` | `source mistral_env/bin/activate`    |
| **Analysis / CPU tasks**              | Use interactive node with `def-csubakan-ab_cpu`      | `source analysis_env/bin/activate`   |

## Log Directory Structure

Logs follow results folder structure: `logs/{model}/{experiment}/`

```bash
#SBATCH --output=logs/{model}/{experiment}/%x-%j.out
#SBATCH --error=logs/{model}/{experiment}/%x-%j.err
```

Create dirs before submitting: `mkdir -p logs/{model}/{experiment}`

## Required SLURM Template Features

### 1. Module Loading + Environment

```bash
module load StdEnv/2023 cuda rust gcc arrow
source {env}/bin/activate
```

### 2. Resource Monitoring (every 20 minutes)

```bash
while sleep 1200; do
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║              RESOURCE USAGE at $(date +%Y-%m-%d\ %H:%M:%S)              ║"
  echo "╠══════════════════════════════════════════════════════════════════╣"
  echo "║ MEMORY (RAM):"
  free -h | awk 'NR==2 {printf "║   Total: %s | Used: %s | Available: %s\n", $2, $3, $7}'
  echo "║ CPU: Cores=${SLURM_CPUS_PER_TASK:-N/A}"
  CPU_USAGE=$(ps -u $USER -o %cpu= | awk '{s+=$1} END {printf "%.1f", s}')
  echo "║   Total CPU Usage: ${CPU_USAGE}%"
  echo "║ GPU:"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r NAME UTIL MEM_USED MEM_TOTAL TEMP; do
    echo "║   $(echo $NAME | xargs) | Util: $(echo $UTIL | xargs)% | VRAM: $(echo $MEM_USED | xargs)/${MEM_TOTAL} MiB | Temp: $(echo $TEMP | xargs)°C"
  done
  echo "╚══════════════════════════════════════════════════════════════════╝"
done &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT
```

### 3. Runtime Summary (at end)

```bash
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Duration: $((DURATION/3600))h $(((DURATION%3600)/60))m $((DURATION%60))s"
```

## Submission Process

// turbo-all

1. Generate scripts (manually or via `scripts/generate_submission_scripts.py`)
2. Create log directories: `mkdir -p logs/{model}/{experiment}`
3. Review one script before batch submission
4. Submit: `for s in submission_scripts/{folder}/*.sh; do sbatch "$s"; done`
5. Verify: `squeue -u $USER`
