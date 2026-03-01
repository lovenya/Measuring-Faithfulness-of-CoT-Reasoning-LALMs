---
description: Load modules and activate the appropriate virtual environment for a model
---

# Activate Environment

Load required modules and activate the correct Python virtual environment for the specified model.

## Step 1: Deactivate any current environment

```bash
deactivate 2>/dev/null || true
```

## Step 2: Load required modules

// turbo

```bash
module load StdEnv/2023 cuda rust gcc arrow
```

## Step 3: Choose which model environment to activate

### For Audio Flamingo 3 HF (`flamingo_hf`):

// turbo

```bash
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/af3_new_hf_env/bin/activate
```

### For Qwen Omni (`qwen_omni`):

// turbo

```bash
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/qwen_omni_env/bin/activate
```

### For Qwen2-Audio (`qwen`):

// turbo

```bash
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/qwen_new_env/bin/activate
```

### For SALMONN / SALMONN_7B:

// turbo

```bash
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/salmonn_env/bin/activate
```

### For Mistral (perturbation generation):

// turbo

```bash
module load opencv
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/mistral_env/bin/activate
```

### For Analysis (CPU-only tasks, plotting, data processing):

// turbo

```bash
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/analysis_env/bin/activate
```

**Note:** For analysis/CPU-only tasks, use an interactive node with `def-csubakan-ab_cpu` account:

```bash
salloc --time=01:00:00 --cpus-per-task=4 --mem=16G --account=def-csubakan-ab_cpu
```

---

## Verify activation

```bash
which python
python --version
```
