#!/bin/bash
# run_mistral_catchup.sh

for ds in sakura-animal sakura-emotion sakura-gender sakura-language mmar; do
    echo "Running mistakes for $ds..."
    python scripts/generate_perturbations.py --model flamingo_hf --dataset $ds --mode mistakes
    
    echo "Running paraphrased for $ds..."
    python scripts/generate_perturbations.py --model flamingo_hf --dataset $ds --mode paraphrase
done
