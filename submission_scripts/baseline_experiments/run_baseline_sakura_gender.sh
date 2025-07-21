#!/bin/bash
#==================================================================
# SBATCH Script for 'baseline.py' on 'sakura-gender'
#==================================================================
#SBATCH --time=12:00:00                                   # Wall-clock time limit (12 hours - a safe estimate)
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1            # Requesting a 40GB H100 MIG instance
#SBATCH --cpus-per-task=8                                 # CPUs per task (ideal for H100 nodes)
#SBATCH --mem=64G                                         # Requesting 64 GB of system RAM
#SBATCH --account=rrg-csubakan                            # High-priority allocation account
#SBATCH --job-name=bline-sakura-gender-500s-10c           # Descriptive job name for monitoring
#SBATCH --output=logs/%x-%j.out                           # Standard output log file
#SBATCH --error=logs/%x-%j.err                            # Standard error log file

#==================================================================
# Job Environment Setup
#==================================================================
echo "## Job Started: $(date) | Job: ${SLURM_JOB_NAME} | Job ID: ${SLURM_JOB_ID} ##"
echo "--> Project Directory: $(pwd)"

echo "--> Loading system-level modules..."
# Rationale: Loading specific, tested versions for reproducibility.
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow

echo "--> Activating Python virtual environment..."
# Rationale: Activates our project-specific Python environment.
source /project/rrg-csubakan/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/audio-faithful/bin/activate

echo "--> Navigating to the main project directory..."
# Rationale: Ensures the script runs from the correct location for local model/data paths.
cd /project/rrg-csubakan/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs

#==================================================================
# GPU Monitoring & Scientific Logic
#==================================================================
# --- GPU Monitoring Loop (runs in the background every 20 mins) ---
while sleep 1200; do
  echo "--- Mid-run GPU Status at $(date) ---"
  nvidia-smi
done &

echo "--> Verifying Initial GPU Status..."
nvidia-smi

echo "--> Starting Python experiment script for baseline on sakura-gender..."
# Rationale: Core scientific logic with 500 samples and 10 chains.
python main.py --experiment baseline \
    --dataset sakura-gender \
    --num-samples 500 \
    --num-chains 10
    
echo "--> Python script finished."

echo "## Job Finished: $(date) ##"