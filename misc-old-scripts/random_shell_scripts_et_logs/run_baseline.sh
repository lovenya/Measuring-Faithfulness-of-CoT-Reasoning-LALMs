#!/bin/bash
#==================================================================
# SBATCH for Baseline LALM Experiment on SAKURA Emotion
#==================================================================
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=rrg-csubakan
#SBATCH --job-name=baseline-sakura-emotion
#SBATCH --output=slurm_logs/%x-%j.out

#==================================================================
# Job Environment Setup
#==================================================================
echo "## Job Started: $(date) | Job: ${SLURM_JOB_NAME} | Job ID: ${SLURM_JOB_ID} ##"

# Load the necessary system modules
module load StdEnv/2023 arrow

# Define project directory and activate Python environment
PROJECT_DIR="/project/def-csubakan-ab/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs"
source "${PROJECT_DIR}/audio-env/bin/activate"
cd "${PROJECT_DIR}"

# Set the audio backend, just in case a sub-dependency needs it. Good practice.
export HF_DATASETS_AUDIO_BACKEND="soundfile"

#==================================================================
# Run the Experiment
#==================================================================
echo "--> Starting Python experiment script..."

# Note the use of backslashes `\` to make the command readable across multiple lines
python main.py \
    --dataset ./data/sakura/emotion/sakura_emotion_test_standardized.jsonl \
    --experiment baseline \
    --num-samples 10 \
    --num-chains 1

echo "--> Python script finished."
echo "## Job Finished: $(date) ##"