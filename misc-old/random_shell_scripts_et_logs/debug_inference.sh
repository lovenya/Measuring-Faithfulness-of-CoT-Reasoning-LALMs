#!/bin/bash
#
# Slurm batch script to run a test inference job.
# This script is designed to be submitted from the project's root directory.

# --- Slurm Directives ---
#SBATCH --account=rrg-csubakan   # REQUIRED: Replace with your Compute Canada account name
#SBATCH --job-name=debug_inference         # Job name for identification
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --cpus-per-task=4                 # Request 4 CPU cores (good for data loading)
#SBATCH --mem=32G                         # Request 32GB of RAM (safe for a 7B model)
#SBATCH --time=0-00:30                    # Request 15 minutes (should be ample time for 2 inferences)
#SBATCH --output=logs/debug_inference_%j.out # Standard output log file in the 'logs' directory
#SBATCH --error=logs/debug_inference_%j.err  # Standard error log file in the 'logs' directory

# --- Job Execution ---

echo "========================================================"
echo "Starting Slurm Job: Testing debug inference Code"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Running on host: $(hostname)"
echo "Executing in directory: $(pwd)"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"
echo "========================================================"

# 1. Load HPC environment modules
# Rationale: This sets up the system-level environment (Python, CUDA) required by our tools.
# It must be done before activating the Python virtual environment.
echo "Loading modules..."
module load StdEnv/2023
module load arrow 

# 2. Activate your Python virtual environment
# Rationale: This isolates our project's Python dependencies.
echo "Activating Python environment..."
source ./audio-env/bin/activate

# 3. Execute the Python inference script
# Rationale: This runs our actual test. We pass the --data-root argument as required by your script.
# The script is expected to be in the same directory from which sbatch is run.
echo "Running the Python inference script..."
python debug_inference.py

# The exit code of the Python script will be the exit code of the job
EXIT_CODE=$?
echo "Python script finished with exit code: $EXIT_CODE"

# 4. Deactivate environment (good practice)
deactivate
echo "Python environment deactivated."

echo "========================================================"
echo "Slurm Job Finished."
echo "========================================================"

exit $EXIT_CODE