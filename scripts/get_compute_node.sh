#!/usr/bin/env bash
set -euo pipefail

# Request an interactive Slurm compute node.
# Defaults match the repository workflow docs but can be overridden with flags.

usage() {
  cat <<'EOF'
Usage:
  scripts/get_compute_node.sh [options]

Options:
  --time HH:MM:SS        Allocation time (default: 02:00:00)
  --gpus SPEC            GPU request spec (default: nvidia_h100_80gb_hbm3_3g.40gb:1)
  --cpus N               CPUs per task (default: 4)
  --mem SIZE             Memory request (default: 64G)
  --account NAME         Slurm account (default: rrg-ravanelm)
  --help                 Show this help

Example:
  scripts/get_compute_node.sh --time 06:00:00 --cpus 8 --mem 96G
EOF
}

TIME_REQ="02:00:00"
GPU_REQ="nvidia_h100_80gb_hbm3_3g.40gb:1"
CPUS_REQ="4"
MEM_REQ="64G"
ACCOUNT_REQ="rrg-ravanelm"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --time)
      TIME_REQ="${2:-}"
      shift 2
      ;;
    --gpus)
      GPU_REQ="${2:-}"
      shift 2
      ;;
    --cpus)
      CPUS_REQ="${2:-}"
      shift 2
      ;;
    --mem)
      MEM_REQ="${2:-}"
      shift 2
      ;;
    --account)
      ACCOUNT_REQ="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

echo "Requesting compute node:"
echo "  time=$TIME_REQ gpus=$GPU_REQ cpus=$CPUS_REQ mem=$MEM_REQ account=$ACCOUNT_REQ"

exec salloc \
  --time="$TIME_REQ" \
  --gpus="$GPU_REQ" \
  --cpus-per-task="$CPUS_REQ" \
  --mem="$MEM_REQ" \
  --account="$ACCOUNT_REQ"
