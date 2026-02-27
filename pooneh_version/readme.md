# Audio Faithfulness Experiments — Model Runner

This repository provides a unified interface to run **audio faithfulness and intervention experiments** across multiple large audio-language models (e.g., Qwen-Instruct, Qwen2.5-Omni, AFthink / Audio-Flamingo 3, SALMONN).  
Each model is wrapped with a standardized runner so experiments can be reproduced consistently.

[Image of machine learning inference pipeline showing standard zero-shot generation versus reasoning-conditioned generation]

---

## Overview

The pipeline follows four main steps:

1. **Environment Setup:** Create an environment and install model-specific dependencies.
2. **Data Preparation:** Download the Sakura dataset repository.
3. **Baseline Inference:** Run the model wrapper to generate a reasoning chain and initial prediction.
4. **Conditioned Inference (Faithfulness Test):** Re-run the model, forcing it to use its own generated reasoning to see if it remains faithful to its logic.

---

## Step 1 — Environment Setup & Dependencies

We strongly recommend creating a **separate Conda environment per model** to avoid severe dependency conflicts (especially between different versions of `transformers`, `peft`, and `accelerate`).

Please install dependencies by following each model’s official page:

* **AF3 (Audio Flamingo 3):** [https://huggingface.co/nvidia/audio-flamingo-3-hf](https://huggingface.co/nvidia/audio-flamingo-3-hf)
* **Qwen2.5-Omni:** [https://huggingface.co/Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
* **SALMONN / Qwen-Instruct:** Follow their respective official repository guidelines.

*Example:*
```bash
conda create -n af3-env python=3.10 -y
conda activate af3-env
# Install specific requirements here...
```

## Step 2 — Get Sakura Repository & Prepare Data
Clone the Sakura repository and ensure the audio data is saved locally. You will need to pass the path to this data when running the inference scripts.

```bash
mkdir -p data/
cd data/
Sakura --> git clone [https://github.com/ckyang1124/SAKURA.git](https://github.com/ckyang1124/SAKURA.git)
MMAU --> HF clone [https://huggingface.co/datasets/gamma-lab-umd/MMAU-test](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test)
MMAR -->  HF clone [https://huggingface.co/datasets/BoJack/MMAR](https://huggingface.co/datasets/BoJack/MMAR)
```

## Step 3 — Run Baseline Inference

Use the provided wrapper scripts (e.g., AF3_wrapper.py) to execute the baseline model runs. These scripts prompt the model to generate a step-by-step reasoning chain followed by a final multiple-choice letter prediction.

Required Arguments:

--input: Path to the .jsonl manifest containing instructions and audio paths.

--output: Destination path for the generated predictions and reasoning.

--data_root: The path to the root directory where the SAKURA repository was cloned.

--num_runs: Numebr of trails per sample. (set to 1).

--use_reasoning: If set , it uses prompt that asks for reason ehn answer. otehrwise, it tries to directly provide answer(no-reasoning baseline)

Command Example:
```bash
python interface/AF3_wrapper.py \
  --input processed_input/sakura/{name}/{name}_manifest.jsonl \
  --output result/baselie/af3/{name}/baseline_{name}_REAS.jsonl \
  --data_root data/ \
  --num_runs 1 \
  --use_reasoning \

  ```

  ###  Step Post-Processing (Recovering Null Labels)
   To extract labels for any remaining null predictions, these scripts parse the model's text using exact string matching, word stemming, and semantic overlap.
   - For SAKURA: Run python pooneh_version/repair_and_save_final.py
   - For MMAR and MMAU: Run python pooneh_version/repair_and_save_final_mm.py
   Fix path inside the scripts.

## Step 4 — Run Conditioned Inference (Faithfulness)
To test if the model actually relies on its own logic, use the conditioned inference scripts. These scripts take the runs_reasonings generated in Step 3, inject them back into the prompt, and force the model to make a final prediction.

Command Example:
```bash 
  python interface/{model}_conditioned.py \
  --manifest  processed_input/sakura/{name}/{name}_manifest.jsonl \
  --results_in  result/baselie/af3/{name}/baseline_{name}_REAS_post_process.jsonl \
  --output   result/baselie/af3/{name}/baseline_{name}/conditioned_{name}.jsonl \
  --data_root data/
  --num_runs 1
  ```
# Result
## AF3
## 📊 Evaluation Summary
🔎 With Reasoning
| Dataset                     | Total Items | Remaining Nulls | New Mean Accuracy | Missed reasoning |
|------------------------------|------------|-----------------|-------------------|------------------|
| Animal                       | 1000       | 10              | 81.30%            |23                |
| Animal – 2-Step Sanity Check | 977        | changed_pred 110| 75.54%            |-                 |
| LANGUAGE                     | 1000       | 17              | 79.60%            |10                |
| EMOTION                      | 1000       | 34              | 45.70%            |18                |
| GENDER                       | 1000       | 6               | 57.20%            |4                 |
| MMAU                         | 1000       | 71              | 67.70%            |0                 |
| MMAR                         | 997        | 4               | 52.86%            |27                |

🚫 Without Reasoning
| Dataset                     | Total Items | Remaining Nulls | New Mean Accuracy |
|------------------------------|------------|-----------------|-------------------|
| Animal                       | 1000       | 0              | 76.70%             |
| LANGUAGE                     | 1000       | 0              | 76.30%             |
| EMOTION                      | 1000       | 0              | 76.70%             |
| GENDER                       | 1000       | 0              | 49.40%             |
| MMAU                         | 1000       | 7              | 74.60%             |
| MMAR                         | 997        | 0              | 54.26%             |

## QWEN2.5
## 📊 Evaluation Summary
🔎 With Reasoning
| Dataset                     | Total Items | Remaining Nulls | New Mean Accuracy | Missed reasoning |
|------------------------------|------------|-----------------|-------------------|------------------|
| Animal                       | 1000       | __              | __            |-                |
| Animal – 2-Step Sanity Check | 977        | __              | __           |-                 |
| LANGUAGE                     | 1000       | __              | __            |-                |
| EMOTION                      | 1000       | __              | __            |-                |
| GENDER                       | 1000       | __               | __            |-                 |
| MMAU                         | 1000       | __              |__            |0                 |
| MMAR                         | 997        | __               | __            |-                |

🚫 Without Reasoning
| Dataset                     | Total Items | Remaining Nulls | New Mean Accuracy |
|------------------------------|------------|-----------------|-------------------|
| Animal                       | 1000       | 0              | 81.40%             |
| LANGUAGE                     | 1000       | 0              | -                  |
| EMOTION                      | 1000       | 0              | -                  |
| GENDER                       | 1000       | 0              | -                  |
| MMAU                         | 1000       | 0              | 66.60%             |
| MMAR                         | 997        | 0              | -                  |