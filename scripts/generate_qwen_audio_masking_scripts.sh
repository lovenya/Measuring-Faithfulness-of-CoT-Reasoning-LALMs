#!/bin/bash
# generate_qwen_audio_masking_scripts_v2.sh
# 
# Generates all Qwen audio masking sbatch scripts with:
# - Hierarchical folder structure: qwen/audio_masking/{mask_type}/{mode}/
# - Resource monitoring every 20 minutes
# - Corresponding log folder structure

set -e

BASE_DIR="submission_scripts/qwen/audio_masking"
LOG_BASE="logs/qwen/audio_masking"

MASK_TYPES="silence noise"
MODES="random start end"
DATASETS="mmar sakura-animal sakura-emotion sakura-gender sakura-language"

echo "Generating Qwen audio masking scripts with hierarchical structure..."

for MASK_TYPE in $MASK_TYPES; do
    for MODE in $MODES; do
        # Create directories
        SCRIPT_DIR="${BASE_DIR}/${MASK_TYPE}/${MODE}"
        LOG_DIR="${LOG_BASE}/${MASK_TYPE}/${MODE}"
        mkdir -p "$SCRIPT_DIR" "$LOG_DIR"
        
        for DATASET in $DATASETS; do
            DATASET_SAFE="${DATASET//-/_}"  # sakura-animal -> sakura_animal
            SCRIPT_NAME="run_qwen_${DATASET_SAFE}.sh"
            JOB_NAME="qwen-${MASK_TYPE:0:3}-${MODE:0:3}-${DATASET}"
            
            cat > "${SCRIPT_DIR}/${SCRIPT_NAME}" << 'SCRIPT_HEADER'
#!/bin/bash
#==================================================================
SCRIPT_HEADER
            cat >> "${SCRIPT_DIR}/${SCRIPT_NAME}" << EOF
# SBATCH Script for Qwen audio_masking (${MASK_TYPE} + ${MODE}) on ${DATASET}
#==================================================================
#SBATCH --time=15:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=rrg-ravanelm
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x-%j.out
#SBATCH --error=${LOG_DIR}/%x-%j.err

echo "## Job Started: \$(date) | Job: \${SLURM_JOB_NAME} | Job ID: \${SLURM_JOB_ID} ##"
echo "Running on host: \$(hostname)"

module load StdEnv/2023 cuda/12.2 rust gcc arrow

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

source qwen_env/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs

#==================================================================
# Resource Monitoring (runs in background every 20 minutes)
#==================================================================
while sleep 1200; do
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  RESOURCE USAGE at \$(date +%Y-%m-%d\ %H:%M:%S)"
  echo "╠══════════════════════════════════════════════════════════╣"
  echo "║ RAM Usage:"
  free -h | awk 'NR==2 {printf "║   Total: %s | Used: %s | Free: %s\\n", \$2, \$3, \$4}'
  echo "║"
  echo "║ CPU Cores Allocated: \${SLURM_CPUS_PER_TASK:-N/A}"
  echo "║ CPU Utilization:"
  top -b -n1 | head -3 | tail -1 | awk '{printf "║   %s\\n", \$0}'
  echo "║"
  echo "║ GPU Status:"
  nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | while read line; do
    echo "║   \$line"
  done || echo "║   No GPU available"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""
done &
MONITOR_PID=\$!
trap "kill \$MONITOR_PID 2>/dev/null" EXIT

nvidia-smi

echo ""
echo "Starting audio_masking experiment..."
python main.py --model qwen --experiment audio_masking --dataset ${DATASET} --verbose --mask-type ${MASK_TYPE} --mask-mode ${MODE}
    
# --- Calculate and Print Total Runtime ---
ELAPSED_SECONDS=\$SECONDS
HOURS=\$((ELAPSED_SECONDS / 3600))
MINS=\$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECS=\$((ELAPSED_SECONDS % 60 ))
echo "--> Total Job Runtime: \${HOURS}h \${MINS}m \${SECS}s"

echo "## Job Finished: \$(date) ##"
EOF
            echo "Created: ${SCRIPT_DIR}/${SCRIPT_NAME}"
        done
    done
done

echo ""
echo "=============================================="
echo "Generated scripts with structure:"
echo "  submission_scripts/qwen/audio_masking/{silence,noise}/{random,start,end}/"
echo "  logs/qwen/audio_masking/{silence,noise}/{random,start,end}/"
echo ""
echo "Total scripts: $(find ${BASE_DIR} -name '*.sh' | wc -l)"
echo "=============================================="
