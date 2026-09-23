#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TESTDATA_DIR="${TESTDATA_DIR:-evaluation/testdata}"
PROMPT_DIR_NAME="${PROMPT_DIR_NAME:-full_video_prompt}"
FRAME_DIR_NAME="${FRAME_DIR_NAME:-frames}"
AUDIO_DIR_NAME="${AUDIO_DIR_NAME:-audio}"
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

"$PYTHON_BIN" evaluation/build_testdata_prompt_csv.py \
  --testdata-dir "$TESTDATA_DIR" \
  --mode text_only \
  --prompt-dir-name "$PROMPT_DIR_NAME" \
  --frame-dir-name "$FRAME_DIR_NAME" \
  --audio-dir-name "$AUDIO_DIR_NAME" \
  --output "$TESTDATA_DIR/generated_prompts/text_only_full_video_prompts.csv"

"$PYTHON_BIN" new_infer.py --config-file configs/inference/ablation_text_only_ovi.yaml
