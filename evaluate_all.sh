#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "No python executable found. Set PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
fi

CONFIG="evaluation/configs/evaluation.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG="$1"
  shift
fi

EXTRA_ARGS=()
if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
  EXTRA_ARGS+=(--output-dir "$EVAL_OUTPUT_DIR")
fi
if [[ -n "${EVAL_WORKSPACE_DIR:-}" ]]; then
  EXTRA_ARGS+=(--workspace-dir "$EVAL_WORKSPACE_DIR")
fi
if [[ "${EVAL_NO_SKIP_COMPLETED:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-skip-completed)
fi

"$PYTHON_BIN" evaluation/run_evaluation.py --config "$CONFIG" --python-bin "$PYTHON_BIN" "${EXTRA_ARGS[@]}" "$@"
