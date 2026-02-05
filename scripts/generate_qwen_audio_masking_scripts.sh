#!/bin/bash
# generate_qwen_audio_masking_scripts.sh
# 
# Generates all Qwen audio masking sbatch scripts for all combinations:
# - 2 mask types (silence, noise)
# - 3 modes (random, start, end)
# - 5 datasets (mmar, sakura-animal, sakura-emotion, sakura-gender, sakura-language)
# Total: 30 scripts

OUTPUT_DIR="submission_scripts/qwen/audio_masking"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs/qwen/audio_masking

MASK_TYPES="silence noise"
MODES="random start end"
DATASETS="mmar sakura-animal sakura-emotion sakura-gender sakura-language"

for MASK_TYPE in $MASK_TYPES; do
    for MODE in $MODES; do
        for DATASET in $DATASETS; do
            SCRIPT_NAME="run_qwen_audio_masking_${MASK_TYPE}_${MODE}_${DATASET//-/_}.sh"
            JOB_NAME="qwen-mask-${MASK_TYPE:0:3}-${MODE:0:3}-${DATASET}"
            
            cat > "${OUTPUT_DIR}/${SCRIPT_NAME}" << EOF
#!/bin/bash
#==================================================================
# SBATCH Script for Qwen audio_masking (${MASK_TYPE} + ${MODE}) on ${DATASET}
#==================================================================
#SBATCH --time=15:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=rrg-ravanelm
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/qwen/audio_masking/%x-%j.out
#SBATCH --error=logs/qwen/audio_masking/%x-%j.err

echo "## Job Started: \$(date) | Job: \${SLURM_JOB_NAME} | Job ID: \${SLURM_JOB_ID} ##"

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

nvidia-smi

python main.py --model qwen --experiment audio_masking --dataset ${DATASET} --verbose --restricted --mask-type ${MASK_TYPE} --mask-mode ${MODE}
    
echo "## Job Finished: \$(date) ##"
EOF
            echo "Created: ${SCRIPT_NAME}"
        done
    done
done

echo ""
echo "Generated $(ls -1 ${OUTPUT_DIR}/*.sh | wc -l) scripts in ${OUTPUT_DIR}/"
