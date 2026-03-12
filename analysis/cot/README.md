# CoT analysis

Each experiment now has its own script:

- `plot_paraphrasing.py`
- `plot_early_answering.py`
- `plot_partial_filler.py` (single script, select `--mode start|end|random`)
- `plot_adding_mistakes.py`

Shared utilities live in `common.py`, while `plot_cot_experiment.py` is the shared backend.

## Canonical defaults

- `paraphrasing`: `--reference-mode 0pct`
- `early_answering`: `--reference-mode 100pct`
- `partial_filler_text` family: `--reference-mode 0pct`
- `adding_mistakes`: `--reference-mode baseline`
  - optional `--include-100-anchor`

## Output paths

Plots follow the same model/experiment structure as results:
- `plots/{model}/paraphrasing/`
- `plots/{model}/early_answering/`
- `plots/{model}/partial_filler_text/{mode}/`
- `plots/{model}/adding_mistakes/`

## Example

```bash
python analysis/cot/plot_partial_filler.py \
  --model qwen_audio_2 \
  --mode random \
  --filler-type lorem \
  --save-pdf
```
