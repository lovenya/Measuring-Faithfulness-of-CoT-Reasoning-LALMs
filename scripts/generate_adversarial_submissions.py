#!/usr/bin/env python3
"""
Generates 16 sbatch submission scripts for the adversarial experiment.
4 tracks × 2 aug modes × 2 variants = 16 scripts.

Account allocation:
- correct + overlay → rrg-ravanelm
- correct + concat → rrg-csubakan  
- wrong + overlay → rrg-csubakan
- wrong + concat → rrg-csubakan
"""

import os

TRACKS = ['animal', 'emotion', 'gender', 'language']
AUG_MODES = ['concat', 'overlay']
VARIANTS = ['correct', 'wrong']

# Account mapping
def get_account(aug, variant):
    if variant == 'correct' and aug == 'overlay':
        return 'rrg-ravanelm'
    else:
        return 'rrg-csubakan'

TEMPLATE = '''#!/bin/bash
#==================================================================
# SBATCH Script for adversarial experiment: {track} / {aug} / {variant}
#==================================================================
#SBATCH --time=02:30:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --account={account}
#SBATCH --job-name=adv-{track_short}-{aug_short}-{variant_short}
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#==================================================================
# Job Environment Setup
#==================================================================
echo "## Job Started: $(date) | Job: ${{SLURM_JOB_NAME}} | Job ID: ${{SLURM_JOB_ID}} ##"
echo "--> Project Directory: $(pwd)"

echo "--> Loading system-level modules..."
module load StdEnv/2023 cuda rust gcc arrow

echo "--> Activating Python virtual environment..."
source qwen_new_env/bin/activate

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
echo "--> Starting adversarial experiment: {track} / {aug} / {variant}..."

python main.py \\
  --model qwen \\
  --experiment adversarial \\
  --dataset sakura-{track} \\
  --adversarial-aug {aug} \\
  --adversarial-variant {variant} \\
  --verbose

echo "--> Python script finished."

#==================================================================
# Runtime Summary
#==================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                         JOB COMPLETE                           ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║ Job Name: ${{SLURM_JOB_NAME}}"
echo "║ Job ID:   ${{SLURM_JOB_ID}}"
echo "║ Finished: $(date)"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "## Job Finished: $(date) ##"
'''

def main():
    base_dir = 'submission_scripts/adversarial'
    
    for aug in AUG_MODES:
        aug_dir = os.path.join(base_dir, aug)
        os.makedirs(aug_dir, exist_ok=True)
        
        for variant in VARIANTS:
            for track in TRACKS:
                account = get_account(aug, variant)
                track_short = track[:3]  # ani, emo, gen, lan
                aug_short = aug[:3]      # con, ove
                variant_short = variant[0]  # c, w
                
                filename = f"run_adversarial_{track}_{aug}_{variant}.sh"
                filepath = os.path.join(aug_dir, filename)
                
                content = TEMPLATE.format(
                    track=track,
                    aug=aug,
                    variant=variant,
                    account=account,
                    track_short=track_short,
                    aug_short=aug_short,
                    variant_short=variant_short,
                )
                
                with open(filepath, 'w') as f:
                    f.write(content)
                os.chmod(filepath, 0o755)
                
                print(f"  ✓ {filepath} (account: {account})")
    
    print(f"\nGenerated {len(TRACKS) * len(AUG_MODES) * len(VARIANTS)} submission scripts.")

if __name__ == '__main__':
    main()
