#!/bin/bash
set -e

# Delete only the scattered directories
echo "Deleting old scattered directories..."
rm -rf data/sakura/animal_masked/silence_scattered data/sakura/animal_masked/noise_scattered
rm -rf data/sakura/emotion_masked/silence_scattered data/sakura/emotion_masked/noise_scattered
rm -rf data/sakura/gender_masked/silence_scattered data/sakura/gender_masked/noise_scattered
rm -rf data/sakura/language_masked/silence_scattered data/sakura/language_masked/noise_scattered
echo "Deletion complete."

# Activate environment
source analysis_env/bin/activate

# Regenerate
echo "Regenerating..."

for DOMAIN in animal emotion gender language; do
    echo "--- Processing $DOMAIN ---"
    
    python data_processing/mask_audio_dataset.py \
        --source data/sakura/$DOMAIN \
        --output data/sakura/${DOMAIN}_masked \
        --mask-type silence \
        --mode scattered \
        --workers 16
        
    python data_processing/mask_audio_dataset.py \
        --source data/sakura/$DOMAIN \
        --output data/sakura/${DOMAIN}_masked \
        --mask-type noise \
        --mode scattered \
        --snr 0 \
        --workers 16
done

echo "All scattered tracking regenerated successfully!"
