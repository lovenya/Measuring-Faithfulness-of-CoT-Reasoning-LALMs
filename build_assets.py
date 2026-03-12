import os
import shutil
import subprocess
from pathlib import Path

# Paths
SRC_DIR = Path("LALM_Faithfulness/figures")
DOCS_ASSETS = Path("docs/assets")

# Create docs/assets if it doesn't exist
DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
(DOCS_ASSETS / "audio_intervention").mkdir(exist_ok=True)
(DOCS_ASSETS / "cot_intervention").mkdir(exist_ok=True)

# Helper function to convert or copy
def process_file(src_path, dest_dir):
    if not src_path.exists():
        print(f"File not found: {src_path}")
        return
        
    dest_path = dest_dir / src_path.name
    
    if src_path.suffix.lower() == '.png':
        print(f"Copying {src_path} -> {dest_path}")
        shutil.copy2(src_path, dest_path)
    elif src_path.suffix.lower() == '.pdf':
        png_path = dest_path.with_suffix('.png')
        print(f"Converting {src_path} -> {png_path}")
        # Try convert (ImageMagick)
        try:
            # -density 300 for high quality
            subprocess.run(["convert", "-density", "300", "-trim", str(src_path), "-quality", "100", str(png_path)], check=True)
        except Exception as e:
            print(f"Failed to convert {src_path} with ImageMagick: {e}")
            # Try pdftocairo
            try:
                subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", "300", str(src_path), str(png_path.with_suffix(''))], check=True)
            except Exception as e2:
                print(f"Failed to convert {src_path} with pdftocairo: {e2}")
                print("Please convert manually or install ImageMagick / poppler-utils.")


# Audio Interventions
audio_src = SRC_DIR / "audio_intervention"
audio_dest = DOCS_ASSETS / "audio_intervention"

audio_files = [
    "Combined_mask_Accuracy.pdf",
    "Combined_mask_Consistency.pdf",
    "Combined_noise_Accuracy.pdf",
    "Combined_noise_Consistency.pdf",
    "af3_adv_accuracy.pdf",
    "af3_adv_consistency.pdf",
    "qwen2.5_adv_accuracy.pdf",
    "qwen2.5_adv_consistency.pdf",
    "legend_adversarial.pdf",
    "legend_datasets_only.pdf",
    "legend_models_only.pdf"
]

for f in audio_files:
    process_file(audio_src / f, audio_dest)

# CoT Interventions
cot_src = SRC_DIR / "cot_intervention"
cot_dest = DOCS_ASSETS / "cot_intervention"

# The structure is nested, let's copy the specific files according to the paper
# Flamingo HF
fhf = cot_src / "flamingo_hf"
process_file(fhf / "adding_mistakes" / "cross_dataset_adding_mistakes_vs_rpf_flamingo_hf_restricted-mistral.png", cot_dest)
process_file(fhf / "random_partial_filler_text" / "cross_dataset_random_partial_filler_text_0pct_flamingo_hf_lorem_restricted.png", cot_dest)
process_file(fhf / "early_answering" / "cross_dataset_early_answering_100pct_flamingo_hf_restricted.png", cot_dest)  # guess
process_file(fhf / "paraphrasing" / "cross_dataset_paraphrasing_0pct_words_flamingo_hf_restricted-mistral.png", cot_dest) # guess

# Qwen Omni
qwen = cot_src / "qwen_omni"
process_file(qwen / "adding_mistakes" / "cross_dataset_adding_mistakes_vs_rpf_qwen_omni_restricted-mistral.png", cot_dest)
process_file(qwen / "random_partial_filler_text" / "cross_dataset_random_partial_filler_text_0pct_qwen_omni_lorem_restricted.png", cot_dest)
process_file(qwen / "early_answering" / "cross_dataset_early_answering_100pct_qwen_omni_restricted.png", cot_dest) # guess
process_file(qwen / "paraphrasing" / "cross_dataset_paraphrasing_0pct_words_qwen_omni_restricted-mistral.png", cot_dest) # guess

print("Asset build complete! Please verify generated PNGs in docs/assets.")
