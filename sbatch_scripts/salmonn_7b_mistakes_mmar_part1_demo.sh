#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --account=rrg-ravanelm
#SBATCH --job-name=salmonn-7b-mistakes-mmar-demo
#SBATCH --output=logs/experiment_mistral/salmonn_7b/adding_mistakes/%x-%j.out
#SBATCH --error=logs/experiment_mistral/salmonn_7b/adding_mistakes/%x-%j.err

echo "## Job Started: $(date) | Job: ${SLURM_JOB_NAME} | Job ID: ${SLURM_JOB_ID} ##"
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
echo "--> [OPTIMIZATION] Copying SALMONN-7B model components to local SSD..."
COPY_START=$(date +%s)

LOCAL_MODEL_DIR="${SLURM_TMPDIR}/model_components"
mkdir -p "${LOCAL_MODEL_DIR}"

# Copy Vicuna-7B (Different from 13B)
echo "    --> Copying Vicuna-7B..."
cp -r model_components/vicuna-7b-v1.5 "${LOCAL_MODEL_DIR}/"

# Copy SALMONN-7B Checkpoint (Different from 13B)
echo "    --> Copying SALMONN-7B checkpoint..."
mkdir -p "${LOCAL_MODEL_DIR}/salmonn-7b-checkpoint"
cp model_components/salmonn-7b-checkpoint/salmonn_7b_v0.pth "${LOCAL_MODEL_DIR}/salmonn-7b-checkpoint/"

# Copy Shared Components (Whisper, BEATs, BERT)
echo "    --> Copying shared components (Whisper, BEATs, BERT)..."
cp -r model_components/whisper-large-v2 "${LOCAL_MODEL_DIR}/"

mkdir -p "${LOCAL_MODEL_DIR}/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2"
cp model_components/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt "${LOCAL_MODEL_DIR}/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/"

cp -r model_components/bert-base-uncased "${LOCAL_MODEL_DIR}/"

COPY_END=$(date +%s)
echo "--> [OPTIMIZATION] Model copy completed in $((COPY_END - COPY_START)) seconds"

export SALMONN_LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR}"

echo "--> Verifying GPU Status..."
nvidia-smi

# ============================================================================
# Run Experiment
# ============================================================================
echo "--> Starting SALMONN-7B Adding Mistakes - MMAR - PART 1..."

# Note: Using --model salmonn_7b alias
python main.py --model salmonn_7b --experiment adding_mistakes --dataset mmar --restricted \
    --use-external-perturbations \
    --perturbation-file results/combined/salmonn_mmar-restricted_adding_mistakes_combined.jsonl \
    --part 1 --total-parts 20 \
    --verbose

echo "--> Python script finished."

OUTPUT_FILE="results/salmonn/adding_mistakes/adding_mistakes_salmonn_7b_mmar-restricted-mistral.part_1.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    LINES=$(wc -l < "$OUTPUT_FILE")
    echo "--> OUTPUT VERIFIED: $OUTPUT_FILE has $LINES lines"
else
    echo "--> OUTPUT NOT FOUND: $OUTPUT_FILE"
fi

echo "## Job Finished: $(date) ##"
