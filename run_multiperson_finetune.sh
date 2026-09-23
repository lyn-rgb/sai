#!/usr/bin/env bash
# 一键多人参考微调：参考人脸特征 → meta CSV → 分布式训练
#
#   bash run_multiperson_finetune.sh                 # 三步全跑
#   STEPS=meta  bash run_multiperson_finetune.sh     # 只做数据准备（特征 + CSV）
#   STEPS=train bash run_multiperson_finetune.sh     # 只训练（复用已有 CSV）
#   FORCE_META=1 bash run_multiperson_finetune.sh    # 强制重建 CSV（特征已有则复用）
#
# 单机/多机自动适配：平台给了 WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT 就走多机配置，
# 否则按单机跑（进程数 = PET_NPROC_PER_NODE 或 GPU 数）。
#
# ⚠️ 只需要改下面「配置区」。三个根目录互相独立，必须写绝对路径。
set -euo pipefail

# ============================== 配置区 ==============================
# --- 1. 数据（三个根目录互相独立）---
ANN_ROOT=/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/avannotate_out/work
VIDEO_ROOT=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/lizhen1/datasets/redrem/OpenHumanVid/clips
CLIP_LIST=/inspire/hdd/global_user/zhangkaipeng-24043/lizhen1/code/scripts/fids_over5s_two_face_en_2spk.txt
FEAT_DIR=/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/train_multiperson/ref_feats
META_DIR=/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/train_multiperson/multiperson_n2

# --- 2. 训练目标：这三项决定 yield，转换脚本会打印每条样本的丢弃原因 ---
N_REFS=2                 # 每条样本的参考人数，必须与数据里「既说话又有音频」的人数一致
NUM_FRAMES=121           # 目标窗口帧数 @24fps（121 = 5.0s）
REF_AUDIO_SECONDS=1.0    # 每人参考音频长度；窗口越长/参考越长，可用样本越少

# --- 3. 训练 ---
CKPT_DIR=/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/ckpts   # 内含 Ovi/model.safetensors 与 InsightFace/
FACE_EMBEDDER_CKPT=$CKPT_DIR/InsightFace
START_CKPT=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/logs/baseline/2026-01-11_sai_1M_cropped_ip-embs_self-lora_bs-16/ckpt/step-98000.safetensors
CONFIG=configs/train/model_multiperson.yaml
OUTPUT_DIR=./logs
LEARNING_RATE=2.5e-5
NUM_EPOCHS=3
SAVE_STEPS=2000
CONDA_ENV=${CONDA_ENV:-sai}    # 留空则不动 conda 环境
STEPS=${STEPS:-all}            # all | feats | meta | train（可用环境变量覆盖）
# ====================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON_BIN=python; else PYTHON_BIN=python3; fi
fi

REF_AUDIO_FRAMES=$("$PYTHON_BIN" -c "print(round($REF_AUDIO_SECONDS * 24))")
META_CSV="$META_DIR/multiperson_meta.csv"

case "$STEPS" in
  feats) RUN_FEATS=1; RUN_META=0; RUN_TRAIN=0 ;;
  meta)  RUN_FEATS=1; RUN_META=1; RUN_TRAIN=0 ;;
  train) RUN_FEATS=0; RUN_META=0; RUN_TRAIN=1 ;;
  all)   RUN_FEATS=1; RUN_META=1; RUN_TRAIN=1 ;;
  *) echo "STEPS 必须是 all|feats|meta|train，当前为 '$STEPS'" >&2; exit 1 ;;
esac

# ------------------------------ 预检 ------------------------------
fail() { echo "❌ $*" >&2; exit 1; }

[[ "$ANN_ROOT" == /* ]]   || fail "ANN_ROOT 必须是绝对路径：$ANN_ROOT"
[[ "$VIDEO_ROOT" == /* ]] || fail "VIDEO_ROOT 必须是绝对路径：$VIDEO_ROOT"
[[ "$FEAT_DIR" == /* ]]   || fail "FEAT_DIR 必须是绝对路径：$FEAT_DIR"
[[ "$META_DIR" == /* ]]   || fail "META_DIR 必须是绝对路径：$META_DIR"
[[ -d "$ANN_ROOT" ]]      || fail "标注根目录不存在：$ANN_ROOT"
[[ -d "$VIDEO_ROOT" ]]    || fail "视频根目录不存在：$VIDEO_ROOT"
[[ -f "$CONFIG" ]]        || fail "训练配置不存在：$CONFIG"
[[ -f "$CKPT_DIR/Ovi/model.safetensors" ]] || fail "底模不存在：$CKPT_DIR/Ovi/model.safetensors"
[[ -d "$FACE_EMBEDDER_CKPT" ]]            || fail "InsightFace 权重目录不存在：$FACE_EMBEDDER_CKPT"
if [[ $RUN_TRAIN -eq 1 ]]; then
  [[ -f "$START_CKPT" ]] || fail "起始 checkpoint 不存在：${START_CKPT}（注意这是单人 ckpt，不是多人轮的产物）"
fi

# 多机 / 单机
if [[ -n "${WORLD_SIZE:-}" && -n "${RANK:-}" ]]; then
  NUM_MACHINES=$WORLD_SIZE
  MACHINE_RANK=$RANK
  NPROC_PER_NODE=${PET_NPROC_PER_NODE:-$("$PYTHON_BIN" -c "import torch;print(torch.cuda.device_count())")}
  ACCEL_CFG="$ROOT_DIR/configs/train/accelerate_config_zero0_multinodes.yaml"
  LAUNCH_EXTRA=(--main_process_ip "${MASTER_ADDR:?多机模式需要 MASTER_ADDR}" --main_process_port "${MASTER_PORT:?多机模式需要 MASTER_PORT}")
else
  NUM_MACHINES=1
  MACHINE_RANK=0
  NPROC_PER_NODE=${PET_NPROC_PER_NODE:-$("$PYTHON_BIN" -c "import torch;print(torch.cuda.device_count())")}
  ACCEL_CFG="$ROOT_DIR/configs/train/accelerate_config_zero0_1node.yaml"
  LAUNCH_EXTRA=()
fi
[[ "$NPROC_PER_NODE" -ge 1 ]] || fail "没有检测到 GPU（NPROC_PER_NODE=${NPROC_PER_NODE}）"
NUM_PROCESSES=$((NUM_MACHINES * NPROC_PER_NODE))
# 有效 batch ≈ 128 条样本（B=1 × 进程数 × 累积步数），与 train_*.sh 的 (32/N)*4 等价
GRAD_ACC_STEPS=$((128 / NUM_PROCESSES)); [[ $GRAD_ACC_STEPS -ge 1 ]] || GRAD_ACC_STEPS=1

echo "============================== 预检 =============================="
echo "标注根      : $ANN_ROOT"
echo "视频根      : $VIDEO_ROOT"
echo "数据列表    : ${CLIP_LIST:-(无，使用 $VIDEO_ROOT/<id>.mp4)}"
echo "参考特征目录: $FEAT_DIR"
echo "meta CSV    : $META_CSV"
echo "参考人数    : $N_REFS | 目标窗口: $NUM_FRAMES 帧 | 参考音频: ${REF_AUDIO_SECONDS}s (${REF_AUDIO_FRAMES} 帧)"
echo "机器/进程   : ${NUM_MACHINES} 台 × ${NPROC_PER_NODE} = ${NUM_PROCESSES} | 梯度累积: ${GRAD_ACC_STEPS}"
echo "加速配置    : $ACCEL_CFG"
echo "=================================================================="

if [[ -n "$CONDA_ENV" ]]; then
  CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
  if [[ -n "$CONDA_BIN" ]]; then
    # shellcheck disable=SC1090
    source "$(dirname "$CONDA_BIN")/../etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "conda 环境   : $(python -c 'import sys;print(sys.executable)')"
  else
    echo "⚠️  没找到 conda，跳过环境激活（如需请设置 CONDA_ENV=\"\" 或 CONDA_BIN）"
  fi
fi

# ------------------------------ 步骤 1：参考人脸特征 ------------------------------
mkdir -p "$FEAT_DIR"
FEAT_COUNT=$(find "$FEAT_DIR" -maxdepth 1 -name '*.pt' | wc -l | tr -d ' ')
if [[ $RUN_FEATS -eq 1 ]]; then
  # 这一步是增量的（已存在的 .pt 会跳过），所以每次都跑一遍；
  # 不能按「.pt 数量 > 0 就跳过」——部分完成的目录会被误判成已完成，后面整批样本因缺特征被丢。
  # FORCE_FEATS=1 → --overwrite，重算已有文件。
  OVERWRITE_ARGS=()
  [[ "${FORCE_FEATS:-0}" == "1" ]] && OVERWRITE_ARGS=(--overwrite)
  echo "==> [1/3] 提取参考人脸特征（antelopev2，增量；已有 $FEAT_COUNT 个 .pt，FORCE_FEATS=1 可重算）"
  "$PYTHON_BIN" dataset/extract_ref_face_feats.py \
    --annotation-root "$ANN_ROOT" \
    --output-dir      "$FEAT_DIR" \
    --face-embedder-ckpt "$FACE_EMBEDDER_CKPT" \
    ${OVERWRITE_ARGS[@]+"${OVERWRITE_ARGS[@]}"}
  FEAT_COUNT=$(find "$FEAT_DIR" -maxdepth 1 -name '*.pt' | wc -l | tr -d ' ')
  [[ "$FEAT_COUNT" -gt 0 ]] || fail "没有生成任何参考人脸特征，检查 $ANN_ROOT/*/s3-cluster/faces/"
fi

# ------------------------------ 步骤 2：meta CSV ------------------------------
if [[ $RUN_META -eq 1 ]]; then
  if [[ -f "$META_CSV" && "${FORCE_META:-0}" != "1" ]]; then
    echo "==> [2/3] 跳过 meta CSV（已存在 ${META_CSV}，FORCE_META=1 可强制重建）"
  else
    mkdir -p "$META_DIR"
    echo "==> [2/3] 生成 meta CSV"
    LIST_ARGS=()
    if [[ -n "$CLIP_LIST" ]]; then
      # 列表里的相对路径默认相对于「视频根目录」（例如 `part_001/ab/cd/<hash>`）；
      # 转换脚本在基准明显不对时会自动探测并打印提示
      LIST_ARGS=(--list "$CLIP_LIST" --list-base "${LIST_BASE:-$VIDEO_ROOT}")
    fi
    "$PYTHON_BIN" dataset/build_meta_from_avannotate.py \
      --annotation-root "$ANN_ROOT" \
      --video-root      "$VIDEO_ROOT" \
      --feat-dir        "$FEAT_DIR" \
      --output          "$META_CSV" \
      --n-refs          "$N_REFS" \
      --num-frames      "$NUM_FRAMES" \
      --ref-audio-seconds "$REF_AUDIO_SECONDS" \
      ${LIST_ARGS[@]+"${LIST_ARGS[@]}"}
  fi
  [[ -f "$META_CSV" ]] || fail "meta CSV 未生成：$META_CSV"
  ROWS=$("$PYTHON_BIN" -c "import csv;print(sum(1 for _ in csv.DictReader(open('$META_CSV'))))")
  echo "    meta CSV 共 $ROWS 行"
  [[ "$ROWS" -gt 0 ]] || fail "meta CSV 是空的：当前窗口/参考长度下没有可用样本，见上面的丢弃原因（可调小 NUM_FRAMES 或 REF_AUDIO_SECONDS）"
  # dataset 会 glob `--meta_dir` 下所有 *.csv，别的 csv 会被当成训练数据
  EXTRA_CSV=$(find "$META_DIR" -maxdepth 1 -name '*.csv' ! -name "$(basename "$META_CSV")" | wc -l | tr -d ' ')
  if [[ "$EXTRA_CSV" -gt 0 ]]; then
    echo "⚠️  $META_DIR 下还有 $EXTRA_CSV 个其它 csv，会被一起当作训练数据；建议单独放一个目录"
  fi
fi

# ------------------------------ 步骤 3：训练 ------------------------------
if [[ $RUN_TRAIN -eq 1 ]]; then
  [[ -f "$META_CSV" ]] || fail "缺少 meta CSV：${META_CSV}（先跑 STEPS=meta）"

  # 运行配置：把本次的 n_refs / ref_audio_frames / fix_prompt_with_asr 写进去，
  # 保证「转换用的人数与参考长度」和「训练吃的值」永远一致
  OUTPUT_PATH="$OUTPUT_DIR/$(date '+%Y-%m-%d_%H-%M-%S')_multiperson_n${N_REFS}_bs-$(printf '%02d' "$NUM_PROCESSES")"
  mkdir -p "$OUTPUT_PATH"
  RUN_CONFIG="$OUTPUT_PATH/run_config.yaml"
  "$PYTHON_BIN" - "$CONFIG" "$RUN_CONFIG" "$N_REFS" "$REF_AUDIO_FRAMES" "$CKPT_DIR" <<'PY'
from pathlib import Path
import sys

src, dst, n_refs, ref_frames, ckpt_dir = sys.argv[1:6]
overrides = {"n_refs": n_refs, "ref_audio_frames": ref_frames, "ckpt_dir": ckpt_dir}
lines = []
for line in Path(src).read_text(encoding="utf-8").splitlines():
    key = line.split(":", 1)[0].strip() if ":" in line and not line.lstrip().startswith("#") else None
    lines.append(f"{key}: {overrides[key]}" if key in overrides else line)
Path(dst).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"运行配置已写入 {dst}（n_refs={n_refs}, ref_audio_frames={ref_frames}, ckpt_dir={ckpt_dir}）")
PY

  echo "==> [3/3] 开始训练，输出目录：$OUTPUT_PATH"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
  export TORCH_NCCL_ENABLE_MONITORING=0
  export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export TOKENIZERS_PARALLELISM=false
  if [[ $NUM_MACHINES -gt 1 ]]; then
    export NCCL_CROSS_NIC=1 NCCL_IB_GID_INDEX=3 NCCL_IB_TIMEOUT=22 NCCL_NET_PLUGIN=none
  fi

  accelerate launch \
    --config_file "$ACCEL_CFG" \
    --num_machines "$NUM_MACHINES" \
    --num_processes "$NUM_PROCESSES" \
    --machine_rank "$MACHINE_RANK" \
    ${LAUNCH_EXTRA[@]+"${LAUNCH_EXTRA[@]}"} \
    train.py \
    --data_root "$ANN_ROOT" \
    --meta_dir "$META_DIR" \
    --model_config_path "$RUN_CONFIG" \
    --height 480 \
    --width 864 \
    --num_frames "$NUM_FRAMES" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --num_epochs "$NUM_EPOCHS" \
    --save_steps "$SAVE_STEPS" \
    --remove_prefix_in_ckpt "pipe.model." \
    --resume_from_ckpt "$START_CKPT" \
    --output_path "$OUTPUT_PATH"
else
  echo "==> 数据准备完成（STEPS=${STEPS}），未启动训练"
fi
