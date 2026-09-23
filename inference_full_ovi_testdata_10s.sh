#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TESTDATA_DIR="${TESTDATA_DIR:-evaluation/testdata}"
PROMPT_DIR_NAME="${PROMPT_DIR_NAME:-full_video_prompt}"
FRAME_DIR_NAME="${FRAME_DIR_NAME:-frames}"
AUDIO_DIR_NAME="${AUDIO_DIR_NAME:-audio}"
CONFIG_FILE="${CONFIG_FILE:-configs/inference/full_ovi_testdata_10s.yaml}"
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
  --mode id2v \
  --prompt-dir-name "$PROMPT_DIR_NAME" \
  --frame-dir-name "$FRAME_DIR_NAME" \
  --audio-dir-name "$AUDIO_DIR_NAME" \
  --output "$TESTDATA_DIR/generated_prompts/id2v_full_video_prompts.csv"

RUN_CONFIG="$CONFIG_FILE"
if [[ -n "${LORA_PATH:-}" || -n "${DURATION_SECONDS:-}" || -n "${FPS:-}" || -n "${OUTPUT_DIR:-}" || -n "${SKIP_EXISTING_OUTPUTS:-}" ]]; then
  RUN_CONFIG="$(mktemp "${TMPDIR:-/tmp}/full_ovi_testdata_10s.XXXXXX.yaml")"
  "$PYTHON_BIN" - "$CONFIG_FILE" "$RUN_CONFIG" "${LORA_PATH:-}" "${DURATION_SECONDS:-}" "${FPS:-}" "${OUTPUT_DIR:-}" "${SKIP_EXISTING_OUTPUTS:-}" <<'PY'
from pathlib import Path
import sys

src, dst = map(Path, sys.argv[1:3])
lora_path, duration_seconds, fps, output_dir, skip_existing_outputs = sys.argv[3:8]
updates = {}
if lora_path:
    updates["lora_path"] = lora_path
if duration_seconds:
    updates["duration_seconds"] = duration_seconds
if fps:
    updates["fps"] = fps
if output_dir:
    updates["output_dir"] = output_dir
if skip_existing_outputs:
    updates["skip_existing_outputs"] = skip_existing_outputs

updated = []
seen = set()
for line in src.read_text(encoding="utf-8").splitlines():
    key = line.split(":", 1)[0].strip() if ":" in line else ""
    if key in updates:
        value = updates[key]
        if key in {"duration_seconds", "fps"}:
            updated.append(f"{key}: {value}")
        else:
            updated.append(f"{key}: {value}")
        seen.add(key)
    else:
        updated.append(line)
for key, value in updates.items():
    if key not in seen:
        updated.append(f"{key}: {value}")
dst.write_text("\n".join(updated) + "\n", encoding="utf-8")
PY
fi

CONFIG_LORA_PATH="$("$PYTHON_BIN" - "$RUN_CONFIG" <<'PY'
from pathlib import Path
import sys

for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if line.startswith("lora_path:"):
        print(line.split(":", 1)[1].strip())
        break
PY
)"

if [[ -n "$CONFIG_LORA_PATH" && ! -f "$CONFIG_LORA_PATH" ]]; then
  echo "LoRA checkpoint does not exist: $CONFIG_LORA_PATH" >&2
  echo "Set a valid path with: LORA_PATH=/path/to/step.safetensors bash inference_full_ovi_testdata_10s.sh" >&2
  exit 1
fi

"$PYTHON_BIN" new_infer.py --config-file "$RUN_CONFIG"
