#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TESTDATA_DIR="${TESTDATA_DIR:-evaluation/testdata}"
PROMPT_DIR_NAME="${PROMPT_DIR_NAME:-full_video_prompt}"
FRAME_DIR_NAME="${FRAME_DIR_NAME:-frames}"
AUDIO_DIR_NAME="${AUDIO_DIR_NAME:-audio}"
ASR_DIR_NAME="${ASR_DIR_NAME:-ASR}"
SAMPLE_IDS="${SAMPLE_IDS:-}"
LIMIT="${LIMIT:-}"

SORA2MINI_DIR="$ROOT_DIR/comparsion/Sora2-mini"
SORA2MINI_CSV="$SORA2MINI_DIR/generated_testdata/evaluation_testdata/ravg.csv"
SORA2MINI_OUTPUT_DIR="${SORA2MINI_OUTPUT_DIR:-$ROOT_DIR/outputs/comparison/sora2_mini_testdata/ip_image_True_ip_audio_True}"
SORA2MINI_CONFIG="$SORA2MINI_DIR/generated_testdata/evaluation_testdata/inference_ravg.yaml"
SORA2MINI_MODEL_PATH="${SORA2MINI_MODEL_PATH:-/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/Lab/custom/Sora2-mini/UniAVGen}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
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

BUILD_ARGS=(
  --testdata-dir "$TESTDATA_DIR"
  --prompt-dir-name "$PROMPT_DIR_NAME"
  --frame-dir-name "$FRAME_DIR_NAME"
  --audio-dir-name "$AUDIO_DIR_NAME"
  --asr-dir-name "$ASR_DIR_NAME"
  --output-csv "$SORA2MINI_CSV"
)

if [[ -n "$SAMPLE_IDS" ]]; then
  BUILD_ARGS+=(--sample-ids "$SAMPLE_IDS")
fi

if [[ -n "$LIMIT" ]]; then
  BUILD_ARGS+=(--limit "$LIMIT")
fi

"$PYTHON_BIN" evaluation/build_sora2mini_testdata.py "${BUILD_ARGS[@]}"

mkdir -p "$(dirname "$SORA2MINI_CONFIG")" "$SORA2MINI_OUTPUT_DIR"

"$PYTHON_BIN" - \
  "$SORA2MINI_DIR/configs/inference.yaml" \
  "$SORA2MINI_CONFIG" \
  "$SORA2MINI_MODEL_PATH" \
  "$SORA2MINI_OUTPUT_DIR" \
  "$SORA2MINI_CSV" \
  "${SEED:-}" \
  "${NUM_STEPS:-}" \
  "${SHIFT:-}" \
  "${AUDIO_GUIDANCE_SCALE:-}" \
  "${VIDEO_GUIDANCE_SCALE:-}" \
  "${SLG_LAYER:-}" \
  "${MACFG_PROP:-}" \
  "${SORA2MINI_SKIP_EXISTING_OUTPUTS:-0}" <<'PY'
from pathlib import Path
import sys

(
    src_config,
    dst_config,
    model_path,
    output_dir,
    test_csv,
    seed,
    num_steps,
    shift,
    audio_guidance_scale,
    video_guidance_scale,
    slg_layer,
    macfg_prop,
    skip_existing_outputs,
) = sys.argv[1:14]

updates = {
    "model_path": model_path,
    "output_dir": output_dir,
    "test_csv": test_csv,
    "compatible_output_names": "true",
    "skip_existing_outputs": skip_existing_outputs,
}
if seed:
    updates["seed"] = seed
if num_steps:
    updates["num_steps"] = num_steps
if shift:
    updates["shift"] = shift
if audio_guidance_scale:
    updates["audio_guidance_scale"] = audio_guidance_scale
if video_guidance_scale:
    updates["video_guidance_scale"] = video_guidance_scale
if slg_layer:
    updates["slg_layer"] = slg_layer
if macfg_prop:
    updates["macfg_prop"] = macfg_prop

updated = []
seen = set()
for line in Path(src_config).read_text(encoding="utf-8").splitlines():
    key = line.split(":", 1)[0].strip() if ":" in line else ""
    if key in updates:
        updated.append(f"{key}: {updates[key]}")
        seen.add(key)
    else:
        updated.append(line)
for key, value in updates.items():
    if key not in seen:
        updated.append(f"{key}: {value}")
Path(dst_config).write_text("\n".join(updated) + "\n", encoding="utf-8")
print(f"Sora2-mini config: {dst_config}")
print(f"Sora2-mini output: {output_dir}")
PY

if [[ "${SORA2MINI_OVERWRITE:-0}" == "1" ]]; then
  find "$SORA2MINI_OUTPUT_DIR" -maxdepth 1 -type f -name '*.mp4' -delete
fi

cd "$SORA2MINI_DIR"
"$TORCHRUN_BIN" --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" inference.py --task 1 --config_file "$SORA2MINI_CONFIG"
