#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --account=def-csubakan-ab
#SBATCH --job-name=salmonn7b-filler-gender
#SBATCH --output=logs/salmonn_7b/random_partial_filler_text/%x-%j.out
#SBATCH --error=logs/salmonn_7b/random_partial_filler_text/%x-%j.err

echo "## Job Started: $(date) | Job: ${SLURM_JOB_NAME} | Job ID: ${SLURM_JOB_ID} ##"
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow python/3.11
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/salmonn_env/bin/activate
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs

LOCAL_MODEL_DIR="${SLURM_TMPDIR}/model_components"
mkdir -p "${LOCAL_MODEL_DIR}"
cp -r model_components/Qwen2-Audio-7B-Instruct "${LOCAL_MODEL_DIR}/"
export SALMONN_7B_LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR}"

nvidia-smi
( while true; do nvidia-smi; free -h; sleep 300; done ) &
MONITOR_PID=$!

python main.py --model salmonn_7b --experiment random_partial_filler_text --dataset sakura-gender --restricted --verbose

kill $MONITOR_PID 2>/dev/null
ELAPSED=$SECONDS
echo "--> Total Runtime: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "## Job Finished: $(date) ##"
