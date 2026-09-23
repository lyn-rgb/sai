#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TESTDATA_DIR="${TESTDATA_DIR:-evaluation/testdata}"
PROMPT_DIR_NAME="${PROMPT_DIR_NAME:-full_video_prompt}"
FRAME_DIR_NAME="${FRAME_DIR_NAME:-frames}"
AUDIO_DIR_NAME="${AUDIO_DIR_NAME:-audio}"
DREAMID_DIR="$ROOT_DIR/comparsion/DreamID-Omni"
DREAMID_DATA_DIR="$DREAMID_DIR/generated_testdata/evaluation_testdata"
DREAMID_OUTPUT_DIR="$ROOT_DIR/outputs/comparison/dreamid_omni_testdata"
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

"$PYTHON_BIN" evaluation/build_dreamid_testdata.py \
  --testdata-dir "$TESTDATA_DIR" \
  --prompt-dir-name "$PROMPT_DIR_NAME" \
  --frame-dir-name "$FRAME_DIR_NAME" \
  --audio-dir-name "$AUDIO_DIR_NAME" \
  --output-dir "$DREAMID_DATA_DIR"

if [[ "${DREAMID_OVERWRITE:-0}" == "1" ]]; then
  rm -rf "$DREAMID_OUTPUT_DIR"
fi

cd "$DREAMID_DIR"
"$PYTHON_BIN" inference_r2av_testdata.py --config-file dreamid_omni/configs/inference/inference_r2av_testdata.yaml
