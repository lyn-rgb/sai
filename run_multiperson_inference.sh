#!/usr/bin/env bash
# 多人参考微调后的推理：训练 CSV 里的样本 → testdata → prompt CSV → new_infer.py
#
#   bash run_multiperson_inference.sh                      # 用最新 ckpt + 训练 CSV 的前 10 条样本
#   LIMIT=20 bash run_multiperson_inference.sh              # 测多少条
#   EACH_EXAMPLE_N_TIMES=2 bash run_multiperson_inference.sh  # 每条多个 seed（看稳定性）
#   SAMPLE_IDS="<id1> <id2>" bash run_multiperson_inference.sh  # 指定样本
#   SAMPLE_IDS="<hash1> <hash2>" bash run_multiperson_inference.sh
#   LORA_PATH=logs/xxx/ckpt/step-5000.safetensors bash run_multiperson_inference.sh
#   SOURCE=testdata TESTDATA_DIR=/abs/my_testdata bash run_multiperson_inference.sh   # 用自己准备的素材
#   PREPARE_ONLY=1 bash run_multiperson_inference.sh       # 只准备数据、打印将执行的命令
#
# 关键点：参考音频必须与训练时喂进去的那一段一致 —— 训练用的是「目标窗口之外」的语音切片，
# 不是整段 TSE 文件，所以 `SOURCE=csv` 会用同一个 helper（dataset/ref_audio.outside_pieces）
# 重新切一遍；参考脸直接复制训练用的 2.2 倍留白裁剪图。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ============================== 配置区 ==============================
# 路径都可以用环境变量覆盖（和训练脚本保持一致，ckpt_dir / embedder 都从这里取）
CKPT_DIR=${CKPT_DIR:-/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/ckpts}
META_DIR=${META_DIR:-/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/train_multiperson/multiperson_n2}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train/model_multiperson.yaml}
INFER_CONFIG=${INFER_CONFIG:-configs/inference/full_ovi_multiperson_5s.yaml}

# 微调产物：默认取 logs 下最新的 step-*.safetensors（可用 LORA_PATH 指定）
LORA_PATH=${LORA_PATH:-}
# 推理素材来源：csv = 从训练 CSV 的样本生成（默认）；testdata = 直接用下面这个目录里已有的素材
SOURCE=${SOURCE:-csv}
TESTDATA_DIR=${TESTDATA_DIR:-$ROOT_DIR/evaluation/testdata_multiperson}
SAMPLE_IDS=${SAMPLE_IDS:-}          # 空 = 取 CSV 前 LIMIT 条（可写多个 id，用空格分隔）
LIMIT=${LIMIT:-10}                  # 默认多测几条；每条 5s 视频按 SAMPLE_STEPS 计一次采样时间

OUTPUT_DIR=${OUTPUT_DIR:-./outputs/full_ovi_multiperson_5s}
SAMPLE_STEPS=${SAMPLE_STEPS:-50}    # 采样步数（50 是 5s 的常用值；越大越慢）
SEED=${SEED:-102}
EACH_EXAMPLE_N_TIMES=${EACH_EXAMPLE_N_TIMES:-1}
N_REFS=${N_REFS:-2}                 # 必须与训练/ckpt 一致
SKIP_EXISTING=${SKIP_EXISTING:-0}   # 1 = 跳过已生成的输出
# 参考图是否再检一次脸：默认 0（我们的参考图就是人脸裁剪，与训练条件一致，也不需要裁剪权重）。
# 置 1 需要 crop_insightface_root（含 models/buffalo_l）+ crop_landmark_ckpt_path，否则会直接报错。
CROP_FACE=${CROP_FACE:-0}
# 1（默认）= 推理结束后把参考脸/参考音频/caption/超参快照归档到结果目录（<output_dir>/inputs/）
DUMP_INPUTS=${DUMP_INPUTS:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-}
# ====================================================================

CONDA_ENV=${CONDA_ENV:-sai}
PREPARE_ONLY=${PREPARE_ONLY:-0}
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON_BIN=python; else PYTHON_BIN=python3; fi
fi

fail() { echo "❌ $*" >&2; exit 1; }

[[ -f "$TRAIN_CONFIG" ]]  || fail "训练配置不存在：${TRAIN_CONFIG}（要用它取 ckpt/embedder 路径，保证训练与推理一致）"
[[ -f "$INFER_CONFIG" ]]  || fail "推理配置不存在：$INFER_CONFIG"

if [[ -n "$CONDA_ENV" ]]; then
  CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
  if [[ -n "$CONDA_BIN" ]]; then
    # shellcheck disable=SC1090
    source "$(dirname "$CONDA_BIN")/../etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
fi

# ------------------------------ 1. 选 ckpt ------------------------------
if [[ -z "$LORA_PATH" ]]; then
  LORA_PATH=$(ls -t "$ROOT_DIR"/logs/*multiperson*/ckpt/step-*.safetensors 2>/dev/null | head -1 || true)
fi
[[ -n "$LORA_PATH" && -f "$LORA_PATH" ]] || fail "没找到微调 ckpt：请用 LORA_PATH=<ckpt> 指定，或确认 logs/*multiperson*/ckpt/ 下有 step-*.safetensors"
LORA_PATH=$(cd "$(dirname "$LORA_PATH")" && pwd)/$(basename "$LORA_PATH")
echo "微调 ckpt   : $LORA_PATH"
echo "             （$(du -h "$LORA_PATH" | cut -f1)；太小说明导出为空，检查训练日志里的 trainable 参数数）"
[[ -f "$CKPT_DIR/Ovi/model.safetensors" ]] || fail "底模不存在：$CKPT_DIR/Ovi/model.safetensors（改脚本顶部的 CKPT_DIR）"

# ------------------------------ 2. 准备 testdata ------------------------------
if [[ "$SOURCE" == "csv" ]]; then
  META_CSV="$META_DIR/multiperson_meta.csv"
  [[ -f "$META_CSV" ]] || fail "训练 CSV 不存在：${META_CSV}（改 META_DIR，或 SOURCE=testdata 用自己的素材）"
  # 窗口帧数与参考长度决定「参考音频怎么切」，必须与生成 CSV 时一致：
  # CSV 的参数指纹是权威记录（num_frames 是训练的 CLI 参数，不在 yaml 里），配置只做交叉校验
  read -r NUM_FRAMES REF_AUDIO_FRAMES FRAME_NOTE < <(
    "$PYTHON_BIN" - "$META_CSV.params.json" "$TRAIN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

params_path, train_config = sys.argv[1], sys.argv[2]
num_frames = ref_frames = None
source = []
if Path(params_path).is_file():
    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    num_frames = params.get("num_frames")
    if params.get("ref_audio_seconds") is not None:
        ref_frames = int(round(float(params["ref_audio_seconds"]) * 24))
    source.append("CSV 参数指纹")

config = {}
for line in Path(train_config).read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped and not stripped.startswith("#") and ":" in stripped:
        key, value = stripped.split(":", 1)
        config[key.strip()] = value.split("#")[0].strip()

warning = ""
if config.get("ref_audio_frames"):
    if ref_frames is None:
        ref_frames = int(config["ref_audio_frames"])
        source.append("训练配置")
    elif int(config["ref_audio_frames"]) != int(ref_frames):
        warning = f"⚠️CSV={ref_frames}配置={config['ref_audio_frames']}"
if num_frames is None:
    num_frames = 121
    source.append("默认 121")
print(num_frames, ref_frames, "+".join(source) + warning)
PY
  )
  echo "推理参数    : num_frames=$NUM_FRAMES ref_audio_frames=${REF_AUDIO_FRAMES}（来源：${FRAME_NOTE}）"
  if [[ "$FRAME_NOTE" == *"⚠️"* && "${ALLOW_PARAM_MISMATCH:-0}" != "1" ]]; then
    fail "CSV 参数指纹与训练配置的 ref_audio_frames 不一致：$FRAME_NOTE → 参考音频会按错的长度切。\
用与训练一致的那份 CSV/配置，或 ALLOW_PARAM_MISMATCH=1 强制执行"
  fi
  echo "==> [1/3] 从训练 CSV 生成推理素材（参考音频按「窗口外」切片，与训练同一条规则）"
  "$PYTHON_BIN" evaluation/build_multiperson_testdata.py \
    --meta-csv "$META_CSV" --testdata-dir "$TESTDATA_DIR" \
    --num-frames "$NUM_FRAMES" --ref-audio-frames "$REF_AUDIO_FRAMES" \
    ${SAMPLE_IDS:+--ids $SAMPLE_IDS} --limit "$LIMIT"
else
  echo "==> [1/3] 使用已有 testdata：$TESTDATA_DIR"
  [[ -d "$TESTDATA_DIR" ]] || fail "testdata 目录不存在：$TESTDATA_DIR"
fi

# ------------------------------ 3. prompt CSV + 推理 ------------------------------
PROMPT_CSV="$TESTDATA_DIR/generated_prompts/multiperson_prompts.csv"
echo "==> [2/3] 生成 prompt CSV"
"$PYTHON_BIN" evaluation/build_testdata_prompt_csv.py \
  --testdata-dir "$TESTDATA_DIR" --mode multiperson --output "$PROMPT_CSV"

# 运行配置：训练配置提供路径（ckpt_dir / face / audio embedder）与多人相关开关，
# 推理配置提供采样参数；两者合并写一份临时 yaml，避免两份配置里的路径各自漂移。
# 便携写法：BSD 的 mktemp 只替换结尾的 XXXXXX，后缀要自己加
RUN_CONFIG="$(mktemp "${TMPDIR:-/tmp}/multiperson_infer.XXXXXX").yaml"
"$PYTHON_BIN" - "$TRAIN_CONFIG" "$INFER_CONFIG" "$RUN_CONFIG" \
  "$LORA_PATH" "$CKPT_DIR" "$PROMPT_CSV" "$OUTPUT_DIR" "$N_REFS" "$SAMPLE_STEPS" "$SEED" \
  "$EACH_EXAMPLE_N_TIMES" "$SKIP_EXISTING" "$TESTDATA_DIR" "$CROP_FACE" <<'PY'
from pathlib import Path
import sys

(train_config, infer_config, out_path, lora_path, ckpt_dir, prompt_csv, output_dir,
 n_refs, sample_steps, seed, each_n, skip_existing, testdata_dir, crop_face) = sys.argv[1:15]


def read_yaml_lines(path):
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
            and not line.lstrip().startswith("#")]


def key_of(line):
    return line.split(":", 1)[0].strip() if ":" in line else None


train_lines = read_yaml_lines(train_config)
merged = []
for line in read_yaml_lines(infer_config):
    key = key_of(line)
    if key in ("text_prompt",):                      # 换成这次生成的 prompt CSV
        continue
    if key in ("ckpt_dir", "face_embedder_ckpt_dir", "audio_embedder_ckpt_dir",
               "n_refs", "use_ref_av_fusion", "fusion_lora_rank", "ref_audio_frames"):
        continue                                     # 一律以训练配置为准（见下）
    merged.append(line)

# 训练侧取值：路径与「多人开关」都必须和训练一致，否则权重加载不上或参考 token 形状不对
overrides = {"ckpt_dir": ckpt_dir, "lora_path": lora_path, "text_prompt": prompt_csv,
             "output_dir": output_dir, "n_refs": n_refs, "sample_steps": sample_steps,
             "seed": seed, "each_example_n_times": each_n,
             "skip_existing_outputs": "true" if str(skip_existing) == "1" else "false",
             "crop_face": "true" if str(crop_face) == "1" else "false"}
from_train = {}
for line in train_lines:
    key = key_of(line)
    if key in ("ckpt_dir", "face_embedder_ckpt_dir", "audio_embedder_ckpt_dir",
               "n_refs", "use_ref_av_fusion", "fusion_lora_rank", "ref_audio_frames"):
        from_train[key] = line.split(":", 1)[1].split("#")[0].strip()

ref_frames = int(from_train.get("ref_audio_frames", 24))
# 参考音频采样数必须 = 帧数 × 16000 / 24（见 dataset/MULTIPERSON_DATA.md）
overrides["ref_audio_samples"] = str(int(round(ref_frames * 16000 / 24)))

final = []
used = set()
for line in merged:
    key = key_of(line)
    if key in overrides:
        final.append(f"{key}: {overrides[key]}")
        used.add(key)
    else:
        final.append(line)
        if key:
            used.add(key)

# 上面刻意丢掉了「以训练配置为准」的键（每行都要重新取），这里统一补回；
# 漏掉 use_ref_av_fusion / fusion_lora_rank 的后果很隐蔽：参考对的融合层不会被创建，
# ckpt 里那两个 LoRA 会变成「多余 key」被静默跳过，推理时机制等于没加载。
for key, value in {**from_train, **overrides}.items():
    if key not in used:
        final.append(f"{key}: {value}")
        used.add(key)

required = ["ckpt_dir", "ckpt_name", "lora_path", "face_embedder_ckpt_dir", "audio_embedder_ckpt_dir",
            "n_refs", "use_ref_av_fusion", "fusion_lora_rank", "ref_audio_samples", "text_prompt",
            "output_dir", "mode", "video_frame_height_width"]
missing = [key for key in required if key not in used]
if missing:
    raise SystemExit(f"运行配置缺少必要键：{missing}（训练/推理配置里有没有这些项？）")
Path(out_path).write_text("\n".join(final) + "\n", encoding="utf-8")
print(f"[run config] lora_path={lora_path}")
print(f"[run config] n_refs={n_refs} ref_audio_samples={overrides['ref_audio_samples']} "
      f"use_ref_av_fusion={from_train.get('use_ref_av_fusion')} fusion_lora_rank={from_train.get('fusion_lora_rank')}")
PY

echo "==> [3/3] 开始推理（输出到 ${OUTPUT_DIR}）"
if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "PREPARE_ONLY=1：数据已准备好，未执行推理。命令为："
  echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON_BIN new_infer.py --config-file $RUN_CONFIG"
  echo "（临时配置保留在 ${RUN_CONFIG}，可直接编辑后重跑）"
  exit 0
fi
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" new_infer.py --config-file "$RUN_CONFIG"

# 归档输入：结果文件夹里同时留一份参考人脸/参考音频/caption，脚本不用再回头找 testdata
if [[ "$DUMP_INPUTS" == "1" ]]; then
  echo "==> 归档参考素材到结果目录"
  "$PYTHON_BIN" evaluation/dump_inference_inputs.py \
    --prompt-csv "$PROMPT_CSV" --output-dir "$OUTPUT_DIR" --run-config "$RUN_CONFIG"
fi
echo "完成。输出在 ${OUTPUT_DIR}/ip_image_True_ip_audio_True_N${N_REFS}/ 下（<序号>_crop-True_<prompt>_<HxW>_<seed>_0.mp4）"
