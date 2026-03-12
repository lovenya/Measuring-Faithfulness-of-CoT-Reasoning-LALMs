# Analysis layout

- `analysis/cot/`
  - Active CoT analysis scripts (one script per experiment).
  - Shared logic is centralized but each experiment has a separate CLI.
- `analysis/audio_interventions/`
  - Active audio intervention analysis scripts.
  - `jasco/` currently contains:
    - `jasco_evaluation_llm_as_a_judge.py`
    - `evaluate_jasco_results.py`
- `analysis/old_scripts/`
  - Archived legacy analysis scripts kept for reference only.

## CoT defaults

- `paraphrasing`: reference = `0%`
- `early_answering`: reference = `100%`
- `partial_filler_text` family: reference = `0%`
- `adding_mistakes`: reference = `baseline`
  - optional `--include-100-anchor`
