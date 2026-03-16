#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_ROOT="${ENV_ROOT:-env}"
ENV_NAME="${ENV_NAME:-aflamingo_3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_INSTALL_ARGS="${PIP_INSTALL_ARGS:-}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python binary '${PYTHON_BIN}' not found in PATH." >&2
  exit 1
fi

ENV_PATH="${REPO_ROOT}/${ENV_ROOT}/${ENV_NAME}"
mkdir -p "${REPO_ROOT}/${ENV_ROOT}"

if [[ ! -d "${ENV_PATH}" ]]; then
  echo "[INFO] Creating virtual environment at ${ENV_PATH}"
  "${PYTHON_BIN}" -m venv "${ENV_PATH}"
else
  echo "[INFO] Reusing existing virtual environment at ${ENV_PATH}"
fi

# shellcheck disable=SC1090
source "${ENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

PACKAGES=(
  torch
  torchvision
  torchaudio
  transformers
  accelerate
  sentencepiece
  librosa
  soundfile
  numpy
  pandas
  nltk
  peft
)

if [[ -n "${PIP_INSTALL_ARGS}" ]]; then
  # Intentional split for user-provided pip args, e.g. "--no-index".
  # shellcheck disable=SC2206
  EXTRA_PIP_ARGS=(${PIP_INSTALL_ARGS})
else
  EXTRA_PIP_ARGS=()
fi

python -m pip install "${EXTRA_PIP_ARGS[@]}" "${PACKAGES[@]}"

export NLTK_DATA="${REPO_ROOT}/nltk_data"
mkdir -p "${NLTK_DATA}"

python - << 'PY'
import os
import nltk

nltk_data = os.environ["NLTK_DATA"]
nltk.data.path.append(nltk_data)

for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
        print(f"[nltk] {pkg} already available in {nltk_data}")
    except LookupError:
        print(f"[nltk] downloading {pkg} into {nltk_data}")
        ok = nltk.download(pkg, download_dir=nltk_data, quiet=False)
        if not ok:
            raise RuntimeError(f"Failed to download NLTK package: {pkg}")
PY

if [[ ! -d "${REPO_ROOT}/nltk_data/tokenizers/punkt" ]]; then
  echo "[ERROR] Missing required NLTK data at '${REPO_ROOT}/nltk_data/tokenizers/punkt'." >&2
  exit 1
fi

cat << DONE
[OK] Environment setup complete.

Environment path:
  ${ENV_PATH}

To activate:
  source ${ENV_PATH}/bin/activate

Optional HPC install mode example:
  PIP_INSTALL_ARGS="--no-index" bash shell_scripts/envs/create_env_aflamingo_3.sh
DONE
