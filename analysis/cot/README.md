# CoT analysis

Each experiment now has its own script:

- `plot_paraphrasing.py`
- `plot_early_answering.py`
- `plot_partial_filler_text.py`
- `plot_flipped_partial_filler_text.py`
- `plot_random_partial_filler_text.py`
- `plot_adding_mistakes.py`

Shared utilities live in `common.py`, while `plot_cot_experiment.py` is the shared backend.

## Canonical defaults

- `paraphrasing`: `--reference-mode 0pct`
- `early_answering`: `--reference-mode 100pct`
- `partial_filler_text` family: `--reference-mode 0pct`
- `adding_mistakes`: `--reference-mode baseline`
  - optional `--include-100-anchor`

## Example

```bash
python analysis/cot/plot_paraphrasing.py \
  --model qwen_audio_2 \
  --perturbation-source mistral \
  --save-pdf
```
