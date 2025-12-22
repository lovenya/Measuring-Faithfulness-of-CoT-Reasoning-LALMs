# Model Environment Setup Guide

This document provides reproducible instructions for setting up Python virtual environments on the **Rorqual cluster** (Compute Canada) for audio language models.

> [!NOTE]
> These environments were created on 2025-12-19 on Rorqual cluster using `--no-index` to leverage pre-built wheels from the cluster wheelhouse.

---

## Prerequisites

Load the following modules before any environment operations:

```bash
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow python/3.11
```

---

## Qwen 2 Audio Environment

**Location**: `/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/qwen_env`

### Creation Steps

```bash
# 1. Load modules
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow python/3.11

# 2. Create virtual environment
python3.11 -m venv /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/qwen_env

# 3. Activate and upgrade pip
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/qwen_env/bin/activate
pip install --upgrade pip

# 4. Install core packages (from cluster wheelhouse)
pip install --no-index torch torchvision torchaudio
pip install --no-index accelerate transformers librosa soundfile peft
pip install --no-index bitsandbytes auto-gptq einops nltk
```

### Installed Versions (as of 2025-12-19)

| Package | Version |
|---------|---------|
| Python | 3.11 |
| torch | 2.9.1+computecanada |
| transformers | 4.57.3+computecanada |
| accelerate | 1.12.0+computecanada |
| peft | 0.18.0+computecanada |
| librosa | 0.11.0+computecanada |
| bitsandbytes | 0.49.0+computecanada |

### Activation (for scripts)

```bash
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow python/3.11
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/qwen_env/bin/activate
```

---

## SALMONN Environment

**Location**: `/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/salmonn_env`

> [!CAUTION]
> SALMONN requires **specific older versions** of `transformers` and `peft` due to API changes in newer versions. The latest versions will NOT work.

### Original Requirements (from SALMONN repo)

```
torch==2.0.1
torchaudio==2.0.2
peft==0.3.0
soundfile
librosa
transformers==4.28.0
sentencepiece==0.1.97
accelerate==0.20.3
bitsandbytes==0.35.0
gradio==3.23.0
```

### Creation Steps (Rorqual Cluster)

```bash
# 1. Load modules
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow python/3.11

# 2. Create virtual environment
python3.11 -m venv /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/salmonn_env

# 3. Activate and upgrade pip
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/salmonn_env/bin/activate
pip install --upgrade pip

# 4. Install core packages (from cluster wheelhouse)
pip install --no-index torch torchvision torchaudio
pip install --no-index soundfile librosa sentencepiece accelerate bitsandbytes gradio
pip install --no-index omegaconf nltk

# 5. CRITICAL: Install COMPATIBLE versions of transformers and peft
#    (newer versions break SALMONN's Qformer imports)
pip install transformers==4.31.0  # NOT latest!
pip install peft==0.5.0           # NOT latest!
```

### Installed Versions (as of 2025-12-19)

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11 | |
| torch | 2.9.1+computecanada | |
| **transformers** | **4.31.0** | ⚠️ Must be 4.31.0 or similar - newer breaks SALMONN |
| **peft** | **0.5.0** | ⚠️ Must be ~0.5.0 - newer requires newer transformers |
| accelerate | 1.12.0+computecanada | |
| librosa | 0.11.0+computecanada | |
| omegaconf | 2.3.0+computecanada | Required by SALMONN config |
| nltk | 3.9.2+computecanada | For sentence tokenization |

### Activation (for scripts)

```bash
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow python/3.11
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/salmonn_env/bin/activate

# Performance environment variables (speeds up loading)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
```

---

## Mistral Small 3 Environment (for Perturbation Generation)

**Location**: `/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/mistral_env`

> [!NOTE]
> This environment is used for `scripts/generate_perturbations.py` to pre-generate mistakes and paraphrases using Mistral Small 3 as an external perturbation model.

### Creation Steps (Rorqual Cluster)

```bash
# 1. Load modules (includes rust and opencv for Mistral)
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow rust opencv python/3.11

# 2. Create virtual environment
python3.11 -m venv /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/mistral_env

# 3. Activate and upgrade pip
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/mistral_env/bin/activate
pip install --upgrade pip

# 4. Install core packages (from cluster wheelhouse)
pip install --no-index torch torchvision torchaudio transformers accelerate nltk
```

### Installed Versions (as of 2025-12-19)

| Package | Version |
|---------|---------|
| Python | 3.11 |
| torch | 2.9.1+computecanada |
| transformers | 4.57.3+computecanada |
| accelerate | 1.12.0+computecanada |
| nltk | 3.9.2+computecanada |

### Activation (for scripts)

```bash
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow rust opencv python/3.11
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/mistral_env/bin/activate
```

---

## SLURM Submission Script Template

Use this template in your submission scripts:

```bash
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-csubakan-ab
#SBATCH --job-name=your_job_name
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "## Job Started: $(date) ##"

# Load modules
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow python/3.11

# Activate environment (choose one)
# For Qwen 2 Audio:
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/qwen_env/bin/activate
# For SALMONN:
# source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/salmonn_env/bin/activate

# Navigate to project
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs

# Run your script
python your_script.py

echo "## Job Finished: $(date) ##"
```

---

## Troubleshooting

### CUDA not available on login node
This is expected. CUDA will be available on GPU compute nodes when jobs run.

### Package not found with --no-index
Some packages may not be in the cluster wheelhouse. Remove `--no-index` for those specific packages:
```bash
pip install package_name  # will fetch from PyPI
```

### Version conflicts
The cluster wheelhouse contains pre-tested compatible versions. If you need specific versions, you may need to build from source or use a different Python version.

---

## Quick Reference

| Model | Environment Path | Activation |
|-------|-----------------|------------|
| Qwen 2 Audio | `qwen_env` | `source .../qwen_env/bin/activate` |
| SALMONN | `salmonn_env` | `source .../salmonn_env/bin/activate` + env vars |
| Mistral Small 3 | `mistral_env` | `source .../mistral_env/bin/activate` |
