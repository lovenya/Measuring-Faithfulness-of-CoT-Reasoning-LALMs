#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --account=rrg-ravanelm
#SBATCH --job-name=salmonn7b-baseline-{dataset}
#SBATCH --output=logs/salmonn_7b/baseline/%x-%j.out
#SBATCH --error=logs/salmonn_7b/baseline/%x-%j.err

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

# Copy Vicuna-7B
cp -r model_components/vicuna-7b-v1.5 "${LOCAL_MODEL_DIR}/"

# Copy SALMONN-7B Checkpoint
mkdir -p "${LOCAL_MODEL_DIR}/salmonn-7b-checkpoint"
cp model_components/salmonn-7b-checkpoint/salmonn_7b_v0.pth "${LOCAL_MODEL_DIR}/salmonn-7b-checkpoint/"

# Copy Shared Components
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
# Run SALMONN-7B Baseline
# ============================================================================
echo "--> Starting SALMONN-7B Baseline - {dataset}..."

python main.py --model salmonn_7b --experiment baseline --dataset {dataset} --num-chains 3 --verbose

echo "--> Python script finished."

ELAPSED=$SECONDS
echo "--> Total Job Runtime: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "## Job Finished: $(date) ##"
