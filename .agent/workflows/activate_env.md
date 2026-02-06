---
description: Load modules and activate the appropriate conda environment for a model
---

# Activate Environment

Load required modules and activate the correct conda environment for the specified model.

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

**Which model will you be using?**

- `qwen` - Qwen2-Audio model
- `salmonn` or `salmonn_7b` - SALMONN models
- `flamingo` - Audio Flamingo 3 model

---

### For Qwen:

// turbo

```bash
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/qwen_env/bin/activate
```

### For SALMONN / SALMONN_7B:

// turbo

```bash
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/salmonn_env/bin/activate
```

### For Audio Flamingo 3:

// turbo

```bash
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/audio-flamingo-env/bin/activate
```

---

## Verify activation

```bash
which python
python --version
```
