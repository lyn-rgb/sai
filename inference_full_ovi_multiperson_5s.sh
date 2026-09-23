#!/usr/bin/env bash
# Multi-person inference over a test set (N reference persons per sample).
# Derived from `inference_full_ovi_testdata_5s.sh`; expects
#   $TESTDATA_DIR/full_video_prompt/{sample_id}_full_caption.txt
#   $TESTDATA_DIR/frames/{sample_id}_frame_p{i}.{jpg,png,...}
#   $TESTDATA_DIR/audio/{sample_id}_audio_p{i}.wav
# where `i` is the reference slot order used at training time (0 = first person).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TESTDATA_DIR="${TESTDATA_DIR:-evaluation/testdata_multiperson}"
PROMPT_DIR_NAME="${PROMPT_DIR_NAME:-full_video_prompt}"
FRAME_DIR_NAME="${FRAME_DIR_NAME:-frames}"
AUDIO_DIR_NAME="${AUDIO_DIR_NAME:-audio}"
CONFIG_FILE="${CONFIG_FILE:-configs/inference/full_ovi_multiperson_5s.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" evaluation/build_testdata_prompt_csv.py \
  --testdata-dir "$TESTDATA_DIR" \
  --mode multiperson \
  --prompt-dir-name "$PROMPT_DIR_NAME" \
  --frame-dir-name "$FRAME_DIR_NAME" \
  --audio-dir-name "$AUDIO_DIR_NAME" \
  --output "$TESTDATA_DIR/generated_prompts/multiperson_prompts.csv"

RUN_CONFIG="$CONFIG_FILE"
if [[ -n "${LORA_PATH:-}" || -n "${OUTPUT_DIR:-}" || -n "${N_REFS:-}" ]]; then
  RUN_CONFIG="$(mktemp "${TMPDIR:-/tmp}/full_ovi_multiperson_5s.XXXXXX.yaml")"
  "$PYTHON_BIN" - "$CONFIG_FILE" "$RUN_CONFIG" "${LORA_PATH:-}" "${OUTPUT_DIR:-}" "${N_REFS:-}" <<'PY'
from pathlib import Path
import sys

src, dst, lora_path, output_dir, n_refs = sys.argv[1:6]
overrides = {"lora_path": lora_path, "output_dir": output_dir, "n_refs": n_refs}
lines = []
for line in Path(src).read_text(encoding="utf-8").splitlines():
    key = line.split(":", 1)[0].strip() if ":" in line else None
    if key in overrides and overrides[key]:
        lines.append(f"{key}: {overrides[key]}")
    else:
        lines.append(line)
Path(dst).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote run config to {dst}")
PY
fi

"$PYTHON_BIN" new_infer.py --config-file "$RUN_CONFIG"
