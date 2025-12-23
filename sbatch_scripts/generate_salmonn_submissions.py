
import os

account_mistakes = "rrg-csubakan"
account_paraphrasing = "rrg-ravanelm"

datasets_mistakes = ["sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]
datasets_paraphrasing = ["mmar", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]

experiments = [
    {
        "name": "adding_mistakes",
        "datasets": datasets_mistakes,
        "account": account_mistakes,
        "short": "mistakes",
        "time": "1:00:00"  # 1 hour
    },
    {
        "name": "paraphrasing",
        "datasets": datasets_paraphrasing,
        "account": account_paraphrasing,
        "short": "paraphrase",
        "time": "0:15:00"  # 15 minutes
    }
]

template = r"""#!/bin/bash
#SBATCH --time={time}
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --account={account}
#SBATCH --job-name=salmonn-{exp_short}-{dataset}
#SBATCH --output=logs/experiment_mistral/salmonn/{experiment}/%x-%A_%a.out
#SBATCH --error=logs/experiment_mistral/salmonn/{experiment}/%x-%A_%a.err
#SBATCH --array=1-20

echo "## Job Started: $(date) | Job: ${{SLURM_JOB_NAME}} | Job ID: ${{SLURM_ARRAY_JOB_ID}} | Task ID: ${{SLURM_ARRAY_TASK_ID}} ##"
echo "--> SLURM_TMPDIR: ${{SLURM_TMPDIR}}"

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

LOCAL_MODEL_DIR="${{SLURM_TMPDIR}}/model_components"
mkdir -p "${{LOCAL_MODEL_DIR}}"

# Copy Vicuna-13B
cp -r model_components/vicuna-13b-v1.1 "${{LOCAL_MODEL_DIR}}/"

# Copy Whisper-large-v2
cp -r model_components/whisper-large-v2 "${{LOCAL_MODEL_DIR}}/"

# Copy BEATs checkpoint
mkdir -p "${{LOCAL_MODEL_DIR}}/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2"
cp model_components/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt "${{LOCAL_MODEL_DIR}}/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/"

# Copy SALMONN checkpoint
mkdir -p "${{LOCAL_MODEL_DIR}}/salmonn-13b-checkpoint"
cp model_components/salmonn-13b-checkpoint/salmonn_v1.pth "${{LOCAL_MODEL_DIR}}/salmonn-13b-checkpoint/"

# Copy bert-base-uncased
cp -r model_components/bert-base-uncased "${{LOCAL_MODEL_DIR}}/"

COPY_END=$(date +%s)
echo "--> [OPTIMIZATION] Model copy completed in $((COPY_END - COPY_START)) seconds"

export SALMONN_LOCAL_MODEL_DIR="${{LOCAL_MODEL_DIR}}"

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
# Run Experiment (Task ID ${{SLURM_ARRAY_TASK_ID}})
# ============================================================================
echo "--> Starting SALMONN {experiment} - {dataset} - PART ${{SLURM_ARRAY_TASK_ID}}..."

python main.py --model salmonn --experiment {experiment} --dataset {dataset} --restricted \
    --use-external-perturbations \
    --perturbation-file results/combined/salmonn_{dataset}-restricted_{experiment}_combined.jsonl \
    --part ${{SLURM_ARRAY_TASK_ID}} --total-parts 20 \
    --verbose

echo "--> Python script finished."

kill $MONITOR_PID 2>/dev/null

OUTPUT_FILE="results/salmonn/{experiment}/{experiment}_salmonn_{dataset}-restricted-mistral.part_${{SLURM_ARRAY_TASK_ID}}.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    LINES=$(wc -l < "$OUTPUT_FILE")
    echo "--> OUTPUT VERIFIED: $OUTPUT_FILE has $LINES lines"
else
    echo "--> OUTPUT NOT FOUND: $OUTPUT_FILE"
fi

ELAPSED=$SECONDS
echo "--> Total Job Runtime: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "## Job Finished: $(date) ##"
"""

scripts = []
for exp in experiments:
    for dataset in exp['datasets']:
        script_content = template.format(
            account=exp['account'],
            exp_short=exp['short'],
            dataset=dataset,
            experiment=exp['name'],
            time=exp['time']
        )
        
        filename = f"sbatch_scripts/salmonn_{exp['name']}_{dataset}_array.sh"
        with open(filename, 'w') as f:
            f.write(script_content)
        
        scripts.append(filename)
        print(f"Generated {filename}")

with open("submit_all_salmonn_jobs.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("echo 'Submitting 9 SALMONN array jobs...'\n")
    for script in scripts:
        f.write(f"sbatch {script}\n")

print("\nRun 'bash submit_all_salmonn_jobs.sh' to submit all.")
