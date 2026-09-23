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
CONFIG_FILE="configs/inference/ablation_no_extractors.yaml"
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

RUN_CONFIG="$CONFIG_FILE"
if [[ -n "${LORA_PATH:-}" ]]; then
  if [[ ! -f "$LORA_PATH" ]]; then
    echo "LORA_PATH does not exist: $LORA_PATH" >&2
    exit 1
  fi
  RUN_CONFIG="$(mktemp "${TMPDIR:-/tmp}/ablation_no_extractors.XXXXXX.yaml")"
  "$PYTHON_BIN" - "$CONFIG_FILE" "$RUN_CONFIG" "$LORA_PATH" <<'PY'
from pathlib import Path
import sys

src, dst, lora_path = map(Path, sys.argv[1:])
lines = src.read_text(encoding="utf-8").splitlines()
updated = []
for line in lines:
    if line.startswith("lora_path:"):
        updated.append(f"lora_path: {lora_path}")
    else:
        updated.append(line)
dst.write_text("\n".join(updated) + "\n", encoding="utf-8")
PY
else
  CONFIG_LORA_PATH="$("$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
from pathlib import Path
import sys

for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if line.startswith("lora_path:"):
        print(line.split(":", 1)[1].strip())
        break
PY
)"
  if [[ ! -f "$CONFIG_LORA_PATH" ]]; then
    echo "LoRA checkpoint does not exist: $CONFIG_LORA_PATH" >&2
    echo "Set a valid path with: LORA_PATH=/path/to/step.safetensors bash inference_ablation_no_extractors.sh" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" evaluation/build_testdata_prompt_csv.py \
  --testdata-dir "$TESTDATA_DIR" \
  --mode id2v \
  --prompt-dir-name "$PROMPT_DIR_NAME" \
  --frame-dir-name "$FRAME_DIR_NAME" \
  --audio-dir-name "$AUDIO_DIR_NAME" \
  --output "$TESTDATA_DIR/generated_prompts/id2v_full_video_prompts.csv"

"$PYTHON_BIN" new_infer.py --config-file "$RUN_CONFIG"
