import os

MODELS = ['qwen_omni', 'flamingo_hf']
DATASETS = ['mmar', 'sakura-animal', 'sakura-emotion', 'sakura-gender', 'sakura-language']

TEMPLATE = """#!/bin/bash
#==================================================================
# SBATCH Script for {experiment}: {dataset} / {model}
#==================================================================
#SBATCH --time={time_limit}
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --account=rrg-csubakan
#SBATCH --job-name={short_exp}-{short_mod}-{short_dat}
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#==================================================================
# Job Environment Setup
#==================================================================
START_TIME=$(date +%s)
echo "## Job Started: $(date) | Job: ${{SLURM_JOB_NAME}} | Job ID: ${{SLURM_JOB_ID}} ##"
echo "--> Project Directory: $(pwd)"

echo "--> Loading system-level modules..."
module load StdEnv/2023 cuda rust gcc arrow

echo "--> Activating Python virtual environment..."
{env_activate}

#==================================================================
# Resource Monitoring (runs in background every 20 minutes)
#==================================================================
while sleep 1200; do
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║              RESOURCE USAGE at $(date +%Y-%m-%d\\ %H:%M:%S)              ║"
  echo "╠══════════════════════════════════════════════════════════════════╣"
  echo "║ MEMORY (RAM):"
  free -h | awk 'NR==2 {{printf "║   Total: %s | Used: %s | Available: %s\\n", $2, $3, $7}}'
  echo "║"
  echo "║ CPU:"
  echo "║   Cores Allocated: ${{SLURM_CPUS_PER_TASK:-N/A}}"
  CPU_USAGE=$(ps -u $USER -o %cpu= | awk '{{s+=$1}} END {{printf "%.1f", s}}')
  echo "║   Total CPU Usage: ${{CPU_USAGE}}% (across all your processes)"
  echo "║"
  echo "║ GPU:"
  GPU_INFO=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
  if [ -n "$GPU_INFO" ]; then
    echo "$GPU_INFO" | while IFS=',' read -r NAME UTIL MEM_USED MEM_TOTAL TEMP; do
      NAME=$(echo "$NAME" | xargs)
      UTIL=$(echo "$UTIL" | xargs)
      MEM_USED=$(echo "$MEM_USED" | xargs)
      MEM_TOTAL=$(echo "$MEM_TOTAL" | xargs)
      TEMP=$(echo "$TEMP" | xargs)
      echo "║   Model: $NAME"
      if [ "$UTIL" = "[N/A]" ] || [ "$UTIL" = "N/A" ]; then
        MIG_UTIL=$(nvidia-smi --query-compute-apps=gpu_uuid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$MIG_UTIL" ]; then
          echo "║   GPU Utilization: MIG mode (per-GPU util unavailable)"
        else
          echo "║   GPU Utilization: N/A (MIG mode - no active compute apps)"
        fi
      else
        echo "║   GPU Utilization: ${{UTIL}}% (how busy the GPU cores are)"
      fi
      echo "║   Memory Used: ${{MEM_USED}} MiB / ${{MEM_TOTAL}} MiB"
      echo "║   Temperature: ${{TEMP}}°C"
    done
  else
    echo "║   No GPU available"
  fi
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""
done &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

echo "--> Verifying Initial GPU Status..."
nvidia-smi

#==================================================================
# Main Experiment
#==================================================================
echo "--> Starting {experiment} experiment for {model} on {dataset}..."

{cmd}

echo "--> Python script finished."

#==================================================================
# Runtime Summary
#==================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                         JOB COMPLETE                           ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║ Job Name: ${{SLURM_JOB_NAME}}"
echo "║ Job ID:   ${{SLURM_JOB_ID}}"
echo "║ Duration: ${{HOURS}}h ${{MINUTES}}m ${{SECONDS}}s"
echo "║ Finished: $(date)"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "## Job Finished: $(date) ##"
"""

for exp in ['baseline', 'early_answering']:
    for model in MODELS:
        for dataset in DATASETS:
            # Environment and time limit based on model
            if model == 'qwen_omni':
                env_activate = "source qwen_omni_env/bin/activate"
                time_limit = "12:00:00"
            else:
                env_activate = "source af3_new_hf_env/bin/activate"
                time_limit = "04:00:00"
            
            # Short names for SLURM job name
            short_exp = "b" if exp == 'baseline' else "ea"
            short_mod = "qo" if model == 'qwen_omni' else "af"
            short_dat = "m" if dataset == 'mmar' else dataset.replace("sakura-", "s-")[:3]
            job_name = f"{short_exp}-{short_mod}-{short_dat}"
            
            # Command based on experiment
            if exp == 'baseline':
                cmd = f"python main.py \\\n  --model {model} \\\n  --experiment baseline \\\n  --dataset {dataset} \\\n  --num-chains 3 \\\n  --verbose"
            else:
                cmd = f"python main.py \\\n  --model {model} \\\n  --experiment early_answering \\\n  --dataset {dataset} \\\n  --restricted \\\n  --num-chains 1 \\\n  --verbose"
            
            content = TEMPLATE.format(
                experiment=exp,
                dataset=dataset,
                model=model,
                time_limit=time_limit,
                env_activate=env_activate,
                cmd=cmd,
                short_exp=short_exp,
                short_mod=short_mod,
                short_dat=short_dat
            )
            
            # Save file
            path = f"submission_scripts/{exp}/{model}/run_{exp}_{model}_{dataset.replace('-', '_')}.sh"
            with open(path, 'w') as f:
                f.write(content)
            os.chmod(path, 0o755)

print("Created 20 submission scripts successfully.")
