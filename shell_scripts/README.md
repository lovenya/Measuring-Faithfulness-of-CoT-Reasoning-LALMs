# Shell Scripts for Reproducibility

This directory contains reproducibility scripts grouped by purpose.

## Structure

- `envs/`: Python environment creation scripts.
- `data/`: Dataset download/preparation scripts.
- `models/`: Model download/setup scripts.

## Notes

- Scripts in this folder are designed to be portable.
- Cluster-specific steps (for example `module load ...`) are intentionally not hardcoded here.
- For HPC-specific pip behavior, use optional flags via environment variables where supported.

## Current scripts

- `envs/create_env_aflamingo_3.sh`: creates `env/aflamingo_3`, installs runtime dependencies, and bootstraps local NLTK data required by `main.py`.
