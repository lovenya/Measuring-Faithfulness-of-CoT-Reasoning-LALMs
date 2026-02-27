#!/usr/bin/env bash
set -euo pipefail

# This script is intended to be sourced:
#   source scripts/activate_env.sh qwen
#
# It loads cluster modules and activates the model-specific Python virtualenv.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced to persist environment changes." >&2
  echo "Use: source scripts/activate_env.sh <qwen|salmonn|salmonn_7b|flamingo|flamingo_hf|mistral|analysis>" >&2
  exit 1
fi

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  echo "Missing model argument." >&2
  echo "Use: source scripts/activate_env.sh <qwen|salmonn|salmonn_7b|flamingo|flamingo_hf|mistral|analysis>" >&2
  return 1
fi

REPO_ROOT="/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs"
ENV_PATH=""

case "$MODEL" in
  qwen)
    ENV_PATH="$REPO_ROOT/qwen_new_env/bin/activate"
    ;;
  salmonn|salmonn_7b)
    ENV_PATH="$REPO_ROOT/salmonn_env/bin/activate"
    ;;
  flamingo)
    ENV_PATH="$REPO_ROOT/audio-flamingo-env/bin/activate"
    ;;
  flamingo_hf)
    ENV_PATH="$REPO_ROOT/af3_new_hf_env/bin/activate"
    ;;
  mistral)
    ENV_PATH="$REPO_ROOT/mistral_env/bin/activate"
    ;;
  analysis)
    ENV_PATH="$REPO_ROOT/analysis_env/bin/activate"
    ;;
  *)
    echo "Unsupported model/env key: $MODEL" >&2
    echo "Supported: qwen, salmonn, salmonn_7b, flamingo, flamingo_hf, mistral, analysis" >&2
    return 1
    ;;
esac

deactivate 2>/dev/null || true

module load StdEnv/2023 cuda rust gcc arrow

if [[ ! -f "$ENV_PATH" ]]; then
  echo "Activation file not found: $ENV_PATH" >&2
  return 1
fi

source "$ENV_PATH"
echo "Activated env for '$MODEL': $ENV_PATH"
echo "python -> $(which python)"
