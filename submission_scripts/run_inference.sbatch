#!/bin/bash
#==================================================================
# SBATCH Directives for Qwen-Audio Inference Test
#==================================================================
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=rrg-csubakan
#SBATCH --job-name=qwen-audio-inference
#SBATCH --output=slurm_logs/%x-%j.out

#==================================================================
# Job Environment Setup
#==================================================================
echo "## Job Started: $(date) | Job: ${SLURM_JOB_NAME} | Job ID: ${SLURM_JOB_ID} ##"

echo "--> Loading system-level modules..."
module load StdEnv/2023 python/3.11 gcc/12.3 cuda/12.2

echo "--> Activating Python virtual environment..."
source ~/projects/def-csubakan-ab/lovenya/Faithfulness-of-Audio-CoT/audio-faith/bin/activate

echo "--> Navigating to the main project directory..."
cd ~/projects/def-csubakan-ab/lovenya/Faithfulness-of-Audio-CoT

#==================================================================
# GPU Monitoring & Scientific Logic
#==================================================================
while sleep 600; do echo "--- Mid-run GPU Status at $(date) ---"; nvidia-smi; done &

echo "--> Verifying Initial GPU Status..."
nvidia-smi

echo "--> Starting Python inference script..."
python run_inference.py
echo "--> Python script finished."

echo "## Job Finished: $(date) ##"