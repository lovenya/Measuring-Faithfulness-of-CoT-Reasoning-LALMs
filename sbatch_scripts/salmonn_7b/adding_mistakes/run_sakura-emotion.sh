#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --account=rrg-ravanelm
#SBATCH --job-name=salmonn7b-mistakes-emotion
#SBATCH --output=logs/salmonn_7b/adding_mistakes/%x-%A_%a.out
#SBATCH --error=logs/salmonn_7b/adding_mistakes/%x-%A_%a.err
#SBATCH --array=1-20

echo "## Job Started: $(date) | Job: ${SLURM_JOB_NAME} | Job ID: ${SLURM_ARRAY_JOB_ID} | Task ID: ${SLURM_ARRAY_TASK_ID} ##"
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

python main.py --model salmonn_7b --experiment adding_mistakes --dataset sakura-emotion --restricted \
    --part ${SLURM_ARRAY_TASK_ID} --total-parts 20 --verbose

kill $MONITOR_PID 2>/dev/null
echo "## Job Finished: $(date) ##"
