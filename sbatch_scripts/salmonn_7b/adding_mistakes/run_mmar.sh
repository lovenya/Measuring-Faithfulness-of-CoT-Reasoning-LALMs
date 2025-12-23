#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --account=rrg-ravanelm
#SBATCH --job-name=salmonn7b-mistakes-mmar
#SBATCH --output=logs/salmonn_7b/adding_mistakes/%x-%A_%a.out
#SBATCH --error=logs/salmonn_7b/adding_mistakes/%x-%A_%a.err
#SBATCH --array=1-20

echo "## Job Started: $(date) | Job: ${SLURM_JOB_NAME} | Job ID: ${SLURM_ARRAY_JOB_ID} | Task ID: ${SLURM_ARRAY_TASK_ID} ##"
echo "--> SLURM_TMPDIR: ${SLURM_TMPDIR}"

module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow python/3.11

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/salmonn_env/bin/activate
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs

# ============================================================================
# OPTIMIZATION: Copy model weights to local SSD
# ============================================================================
echo "--> [OPTIMIZATION] Copying model components to local SSD..."
COPY_START=$(date +%s)

LOCAL_MODEL_DIR="${SLURM_TMPDIR}/model_components"
mkdir -p "${LOCAL_MODEL_DIR}"

# Copy Qwen2-Audio-7B-Instruct
cp -r model_components/Qwen2-Audio-7B-Instruct "${LOCAL_MODEL_DIR}/"

COPY_END=$(date +%s)
echo "--> [OPTIMIZATION] Model copy completed in $((COPY_END - COPY_START)) seconds"

export SALMONN_7B_LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR}"

echo "--> Verifying GPU Status..."
nvidia-smi

# Resource Monitoring
(
    while true; do
        echo "================================="
        echo "=== Timestamp: $(date) ==="
        echo "=== GPU Status ==="
        nvidia-smi
        echo "=== Memory Status ==="
        free -h
        echo "=== CPU/Top Processes ==="
        top -b -n 1 | head -n 20
        echo "================================="
        sleep 300
    done
) &
MONITOR_PID=$!

# ============================================================================
# Run Experiment (Task ID ${SLURM_ARRAY_TASK_ID})
# ============================================================================
echo "--> Starting SALMONN-7B adding_mistakes - mmar - PART ${SLURM_ARRAY_TASK_ID}..."

python main.py --model salmonn_7b --experiment adding_mistakes --dataset mmar --restricted \
    --part ${SLURM_ARRAY_TASK_ID} --total-parts 20 \
    --verbose

echo "--> Python script finished."

kill $MONITOR_PID 2>/dev/null

OUTPUT_FILE="results/salmonn_7b/adding_mistakes/adding_mistakes_salmonn_7b_mmar-restricted.part_${SLURM_ARRAY_TASK_ID}.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    LINES=$(wc -l < "$OUTPUT_FILE")
    echo "--> OUTPUT VERIFIED: $OUTPUT_FILE has $LINES lines"
else
    echo "--> OUTPUT NOT FOUND: $OUTPUT_FILE"
fi

ELAPSED=$SECONDS
echo "--> Total Job Runtime: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "## Job Finished: $(date) ##"
