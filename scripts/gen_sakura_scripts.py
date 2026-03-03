#!/usr/bin/env python3
"""Generate SLURM submission scripts for sakura experiments (both models)."""

import os

PROJECT = "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs"
SAKURA = ["sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]
MODELS = ["flamingo_hf", "qwen_omni"]

# ── Environment configs ──
ENV = {
    "flamingo_hf": "source af3_new_hf_env/bin/activate",
    "qwen_omni":   "source qwen_omni_env/bin/activate",
    "mistral":     "source mistral_env/bin/activate",
}
MODULES = {
    "default": "module load StdEnv/2023 cuda rust gcc arrow",
    "mistral": "module load StdEnv/2023 cuda rust gcc arrow opencv",
}
SHORT_MODEL = {"flamingo_hf": "af", "qwen_omni": "qo"}
SHORT_DATASET = {
    "sakura-animal": "s-ani", "sakura-emotion": "s-emo",
    "sakura-gender": "s-gen", "sakura-language": "s-lan",
}


def make_template(*, job_name, time_limit, gpu, account, log_dir, modules, env_activate, description, command):
    return f"""#!/bin/bash
#==================================================================
# {description}
#==================================================================
#SBATCH --time={time_limit}
#SBATCH --gpus={gpu}
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --account={account}
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err

#==================================================================
# Job Environment Setup
#==================================================================
START_TIME=$(date +%s)
START_TZ=$(date +"%Y-%m-%d %H:%M:%S %Z")
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                        JOB STARTED                             ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║ Job Name : ${{SLURM_JOB_NAME}}"
echo "║ Job ID   : ${{SLURM_JOB_ID}}"
echo "║ Node     : $(hostname)"
echo "║ Start    : ${{START_TZ}}"
echo "║ Account  : {account}"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

cd {PROJECT}

echo "--> Loading system-level modules..."
{modules}

echo "--> Activating Python virtual environment..."
{env_activate}

echo "--> Python: $(which python)"
echo "--> PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"

#==================================================================
# Resource Monitoring (runs in background every 20 minutes)
#==================================================================
monitor_resources() {{
  while sleep 1200; do
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           RESOURCE SNAPSHOT at $(date +"%Y-%m-%d %H:%M:%S %Z")          ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║"
    echo "║ ── RAM ──"
    free -h | awk 'NR==2 {{printf "║   Total: %s | Used: %s | Free: %s | Available: %s\\n", $2, $3, $4, $7}}'
    echo "║"
    echo "║ ── CPU ──"
    echo "║   Allocated Cores : ${{SLURM_CPUS_PER_TASK:-N/A}}"
    LOAD=$(cat /proc/loadavg | awk '{{print $1, $2, $3}}')
    echo "║   Load Average    : $LOAD (1m 5m 15m)"
    CPU_USAGE=$(ps -u $USER -o %cpu= 2>/dev/null | awk '{{s+=$1}} END {{printf "%.1f", s}}')
    echo "║   Process CPU     : ${{CPU_USAGE}}%"
    echo "║"
    echo "║ ── GPU ──"
    GPU_INFO=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$GPU_INFO" ]; then
      echo "$GPU_INFO" | while IFS=',' read -r NAME UTIL MEM_USED MEM_TOTAL TEMP POWER; do
        NAME=$(echo "$NAME" | xargs)
        UTIL=$(echo "$UTIL" | xargs)
        MEM_USED=$(echo "$MEM_USED" | xargs)
        MEM_TOTAL=$(echo "$MEM_TOTAL" | xargs)
        TEMP=$(echo "$TEMP" | xargs)
        POWER=$(echo "$POWER" | xargs)
        echo "║   Device     : $NAME"
        if [ "$UTIL" = "[N/A]" ] || [ "$UTIL" = "N/A" ]; then
          echo "║   Utilization: MIG slice (per-GPU util unavailable)"
        else
          echo "║   Utilization: ${{UTIL}}%"
        fi
        echo "║   VRAM       : ${{MEM_USED}} MiB / ${{MEM_TOTAL}} MiB"
        echo "║   Temperature: ${{TEMP}}°C"
        if [ "$POWER" != "[N/A]" ] && [ "$POWER" != "N/A" ]; then
          echo "║   Power Draw : ${{POWER}} W"
        fi
      done
    else
      echo "║   No GPU detected"
    fi
    echo "║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
  done
}}
monitor_resources &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

echo "--> Initial GPU status:"
nvidia-smi

#==================================================================
# Main Experiment
#==================================================================
echo ""
echo "--> Starting: {description}"
echo ""

{command}

EXIT_CODE=$?
echo ""
echo "--> Script finished with exit code: $EXIT_CODE"

#==================================================================
# Runtime Summary
#==================================================================
END_TIME=$(date +%s)
END_TZ=$(date +"%Y-%m-%d %H:%M:%S %Z")
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                         JOB COMPLETE                           ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║ Job Name  : ${{SLURM_JOB_NAME}}"
echo "║ Job ID    : ${{SLURM_JOB_ID}}"
echo "║ Exit Code : $EXIT_CODE"
echo "║ Started   : ${{START_TZ}}"
echo "║ Finished  : ${{END_TZ}}"
echo "║ Duration  : $(printf '%02d:%02d:%02d' $HOURS $MINUTES $SECONDS)"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
"""


def gen_early_answering():
    """Single job per dataset, per model."""
    account = "def-csubakan-ab"
    time_limit = "01:20:00"
    gpu = "nvidia_h100_80gb_hbm3_3g.40gb:1"

    for model in MODELS:
        for ds in SAKURA:
            sm, sd = SHORT_MODEL[model], SHORT_DATASET[ds]
            job_name = f"ea-{sm}-{sd}"
            log_dir = f"logs/{model}/early_answering"
            desc = f"Early Answering: {model} / {ds}"
            cmd = (
                f"python main.py \\\n"
                f"  --model {model} \\\n"
                f"  --experiment early_answering \\\n"
                f"  --dataset {ds} \\\n"
                f"  --num-chains 1 \\\n"
                f"  --verbose"
            )
            script = make_template(
                job_name=job_name, time_limit=time_limit, gpu=gpu,
                account=account, log_dir=log_dir, modules=MODULES["default"],
                env_activate=ENV[model], description=desc, command=cmd,
            )
            path = f"submission_scripts/early_answering/{model}/run_ea_{model}_{ds.replace('-','_')}.sh"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(script)
            os.chmod(path, 0o755)
            print(f"  ✓ {path}")


def gen_random_partial_filler():
    """5 parts per dataset, per model. Uses SLURM array 1-5."""
    account = "rrg-ravanelm"
    time_limit = "01:00:00"
    gpu = "nvidia_h100_80gb_hbm3_3g.40gb:1"

    for model in MODELS:
        for ds in SAKURA:
            sm, sd = SHORT_MODEL[model], SHORT_DATASET[ds]
            job_name = f"rpf-{sm}-{sd}"
            log_dir = f"logs/{model}/random_partial_filler_text"
            desc = f"Random Partial Filler (lorem): {model} / {ds} / part ${{SLURM_ARRAY_TASK_ID}}"
            cmd = (
                f"python main.py \\\n"
                f"  --model {model} \\\n"
                f"  --experiment random_partial_filler_text \\\n"
                f"  --dataset {ds} \\\n"
                f"  --filler-type lorem \\\n"
                f"  --part ${{SLURM_ARRAY_TASK_ID}} \\\n"
                f"  --total-parts 5 \\\n"
                f"  --num-chains 1 \\\n"
                f"  --verbose"
            )
            # Override: add array directive
            script = make_template(
                job_name=job_name, time_limit=time_limit, gpu=gpu,
                account=account, log_dir=log_dir, modules=MODULES["default"],
                env_activate=ENV[model], description=desc, command=cmd,
            )
            # Insert array directive after job-name line
            script = script.replace(
                f"#SBATCH --job-name={job_name}\n",
                f"#SBATCH --job-name={job_name}\n#SBATCH --array=1-5\n",
            )
            path = f"submission_scripts/random_partial_filler_text/{model}/run_rpf_{model}_{ds.replace('-','_')}.sh"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(script)
            os.chmod(path, 0o755)
            print(f"  ✓ {path}")


def gen_mistral_perturbations():
    """Mistral generation: mistakes (1h) and paraphrase (1:40h) per dataset per model."""
    account = "rrg-ravanelm"
    gpu = "h100:1"

    for model in MODELS:
        for ds in SAKURA:
            sm, sd = SHORT_MODEL[model], SHORT_DATASET[ds]
            for mode, time_limit, short_mode in [
                ("mistakes",   "01:00:00", "mis"),
                ("paraphrase", "01:40:00", "par"),
            ]:
                job_name = f"mp-{short_mode}-{sm}-{sd}"
                log_dir = f"logs/external_llm_perturbations/mistral/{model}/{ds.replace('-','_')}"
                desc = f"Mistral {mode}: {model} / {ds}"
                cmd = (
                    f"python scripts/generate_perturbations.py \\\n"
                    f"  --model {model} \\\n"
                    f"  --dataset {ds} \\\n"
                    f"  --mode {mode} \\\n"
                    f"  --num-chains 1"
                )
                script = make_template(
                    job_name=job_name, time_limit=time_limit, gpu=gpu,
                    account=account, log_dir=log_dir, modules=MODULES["mistral"],
                    env_activate=ENV["mistral"], description=desc, command=cmd,
                )
                path = f"submission_scripts/external_llm_perturbations/mistral/{model}/{ds.replace('-','_')}/run_generate_{mode}.sh"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(script)
                os.chmod(path, 0o755)
                print(f"  ✓ {path}")


if __name__ == "__main__":
    os.chdir(PROJECT)
    print("=== Generating Early Answering scripts ===")
    gen_early_answering()
    print("\n=== Generating Random Partial Filler scripts ===")
    gen_random_partial_filler()
    print("\n=== Generating Mistral Perturbation scripts ===")
    gen_mistral_perturbations()
    print("\nDone! All scripts generated.")
