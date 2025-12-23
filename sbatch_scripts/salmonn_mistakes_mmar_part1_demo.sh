#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=rrg-csubakan
#SBATCH --job-name=salmonn-mistakes-mmar-part1-demo
#SBATCH --output=logs/experiment_mistral/salmonn/adding_mistakes/%x-%j.out
#SBATCH --error=logs/experiment_mistral/salmonn/adding_mistakes/%x-%j.err

# DEMO: Run SALMONN Adding Mistakes Part 1 (~249 trials) to verify workflow

echo "## Job Started: $(date) | Job: ${SLURM_JOB_NAME} | Job ID: ${SLURM_JOB_ID} ##"
echo "--> DEMO: Testing SALMONN parallel workflow - Part 1 only"
echo "--> SLURM_TMPDIR (Local SSD): ${SLURM_TMPDIR}"

echo "--> Loading system-level modules..."
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow python/3.11

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "--> Activating SALMONN Python virtual environment..."
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/salmonn_env/bin/activate

cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs

# ============================================================================
# OPTIMIZATION: Copy model weights to local SSD
# ============================================================================
echo "--> [OPTIMIZATION] Copying model components to local SSD..."
COPY_START=$(date +%s)

LOCAL_MODEL_DIR="${SLURM_TMPDIR}/model_components"
mkdir -p "${LOCAL_MODEL_DIR}"

echo "    --> Copying Vicuna-13B (~26GB)..."
cp -r model_components/vicuna-13b-v1.1 "${LOCAL_MODEL_DIR}/"

echo "    --> Copying Whisper-large-v2 (~3GB)..."
cp -r model_components/whisper-large-v2 "${LOCAL_MODEL_DIR}/"

echo "    --> Copying BEATs checkpoint..."
mkdir -p "${LOCAL_MODEL_DIR}/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2"
cp model_components/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt "${LOCAL_MODEL_DIR}/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/"

echo "    --> Copying SALMONN checkpoint (~14GB)..."
mkdir -p "${LOCAL_MODEL_DIR}/salmonn-13b-checkpoint"
cp model_components/salmonn-13b-checkpoint/salmonn_v1.pth "${LOCAL_MODEL_DIR}/salmonn-13b-checkpoint/"

echo "    --> Copying bert-base-uncased..."
cp -r model_components/bert-base-uncased "${LOCAL_MODEL_DIR}/"

COPY_END=$(date +%s)
echo "--> [OPTIMIZATION] Model copy completed in $((COPY_END - COPY_START)) seconds"

du -sh "${LOCAL_MODEL_DIR}"/*

export SALMONN_LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR}"

echo "--> Verifying GPU Status..."
nvidia-smi

# ============================================================================
# Run Part 1 (~249 trials)
# ============================================================================
echo "--> Starting SALMONN Adding Mistakes - PART 1 (~249 trials)..."
python main.py --model salmonn --experiment adding_mistakes --dataset mmar --restricted \
    --use-external-perturbations \
    --perturbation-file results/combined/salmonn_mmar-restricted_adding_mistakes_combined.jsonl \
    --part 1 --total-parts 20 \
    --verbose

echo "--> Python script finished."

# Verify output
OUTPUT_FILE="results/salmonn/adding_mistakes/adding_mistakes_salmonn_mmar-restricted-mistral.part_1.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    LINES=$(wc -l < "$OUTPUT_FILE")
    echo "--> OUTPUT VERIFIED: $OUTPUT_FILE has $LINES lines"
else
    echo "--> OUTPUT NOT FOUND: $OUTPUT_FILE"
fi

ELAPSED=$SECONDS
echo "--> Total Job Runtime: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "## Job Finished: $(date) ##"
