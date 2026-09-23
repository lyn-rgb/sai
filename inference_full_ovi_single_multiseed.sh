#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAMPLE_ID="${SAMPLE_ID:-${1:-00001}}"
TESTDATA_DIR="${TESTDATA_DIR:-evaluation/testdata}"
PROMPT_DIR_NAME="${PROMPT_DIR_NAME:-full_video_prompt}"
FRAME_DIR_NAME="${FRAME_DIR_NAME:-frames}"
AUDIO_DIR_NAME="${AUDIO_DIR_NAME:-audio}"
CONFIG_FILE="${CONFIG_FILE:-configs/inference/full_ovi_testdata_5s.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/full_ovi_single_${SAMPLE_ID}_multiseed}"

NUM_SEEDS="${NUM_SEEDS:-4}"
RANDOM_SEEDS="${RANDOM_SEEDS:-true}"
BASE_SEED="${BASE_SEED:-102}"
SEED_LIST="${SEED_LIST:-}"
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

RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/full_ovi_single_${SAMPLE_ID}.XXXXXX")"
PROMPT_CSV="$RUN_DIR/${SAMPLE_ID}_prompt.csv"
RUN_CONFIG="$RUN_DIR/config.yaml"

"$PYTHON_BIN" - \
  "$CONFIG_FILE" \
  "$RUN_CONFIG" \
  "$PROMPT_CSV" \
  "$TESTDATA_DIR" \
  "$SAMPLE_ID" \
  "$PROMPT_DIR_NAME" \
  "$FRAME_DIR_NAME" \
  "$AUDIO_DIR_NAME" \
  "$OUTPUT_DIR" \
  "$NUM_SEEDS" \
  "$RANDOM_SEEDS" \
  "$BASE_SEED" \
  "$SEED_LIST" \
  "${LORA_PATH:-}" \
  "${DURATION_SECONDS:-}" \
  "${FPS:-}" \
  "${USE_REFERENCE_TOKENS:-}" \
  "${USE_IP_EMBEDDINGS:-}" \
  "${CROP_FACE:-}" \
  "${SKIP_EXISTING_OUTPUTS:-}" <<'PY'
from pathlib import Path
import csv
import sys

(
    config_path,
    run_config_path,
    prompt_csv_path,
    testdata_dir,
    sample_id,
    prompt_dir_name,
    frame_dir_name,
    audio_dir_name,
    output_dir,
    num_seeds,
    random_seeds,
    base_seed,
    seed_list,
    lora_path,
    duration_seconds,
    fps,
    use_reference_tokens,
    use_ip_embeddings,
    crop_face,
    skip_existing_outputs,
) = sys.argv[1:21]

config_path = Path(config_path)
run_config_path = Path(run_config_path)
prompt_csv_path = Path(prompt_csv_path)
testdata_dir = Path(testdata_dir).resolve()

prompt_path = testdata_dir / prompt_dir_name / f"{sample_id}_full_caption.txt"
if not prompt_path.is_file():
    raise FileNotFoundError(f"Prompt not found: {prompt_path}")

frame_dir = testdata_dir / frame_dir_name
frame_path = None
for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
    candidate = frame_dir / f"{sample_id}_frame{ext}"
    if candidate.is_file():
        frame_path = candidate.resolve()
        break
if frame_path is None:
    raise FileNotFoundError(f"Frame image not found for sample {sample_id} in {frame_dir}")

audio_path = (testdata_dir / audio_dir_name / f"{sample_id}_audio.wav").resolve()
if not audio_path.is_file():
    raise FileNotFoundError(f"Audio not found: {audio_path}")

prompt_csv_path.parent.mkdir(parents=True, exist_ok=True)
with prompt_csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text_prompt", "ip_image_path", "ip_audio_path"])
    writer.writeheader()
    writer.writerow({
        "text_prompt": prompt_path.read_text(encoding="utf-8").strip(),
        "ip_image_path": str(frame_path),
        "ip_audio_path": str(audio_path),
    })

updates = {
    "mode": "id2v",
    "text_prompt": str(prompt_csv_path),
    "output_dir": output_dir,
    "seed": base_seed,
    "num_seeds": num_seeds,
    "random_seeds": random_seeds,
    "each_example_n_times": "1",
}
if seed_list:
    updates["seed_list"] = seed_list
if lora_path:
    updates["lora_path"] = lora_path
if duration_seconds:
    updates["duration_seconds"] = duration_seconds
if fps:
    updates["fps"] = fps
if use_reference_tokens:
    updates["use_reference_tokens"] = use_reference_tokens
if use_ip_embeddings:
    updates["use_ip_embeddings"] = use_ip_embeddings
if crop_face:
    updates["crop_face"] = crop_face
if skip_existing_outputs:
    updates["skip_existing_outputs"] = skip_existing_outputs

updated = []
seen = set()
for line in config_path.read_text(encoding="utf-8").splitlines():
    key = line.split(":", 1)[0].strip() if ":" in line else ""
    if key in updates:
        updated.append(f"{key}: {updates[key]}")
        seen.add(key)
    else:
        updated.append(line)
for key, value in updates.items():
    if key not in seen:
        updated.append(f"{key}: {value}")
run_config_path.write_text("\n".join(updated) + "\n", encoding="utf-8")

print(f"Sample: {sample_id}")
print(f"Prompt CSV: {prompt_csv_path}")
print(f"Run config: {run_config_path}")
print(f"Output dir: {output_dir}")
PY

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
  echo "Set a valid path with: LORA_PATH=/path/to/step.safetensors bash inference_full_ovi_single_multiseed.sh" >&2
  exit 1
fi

"$PYTHON_BIN" infer_multi_seed.py --config-file "$RUN_CONFIG"
