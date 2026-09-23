# 多人参考数据格式（meta CSV）

`TextAudioVideoFaceDataset` 通过 `--meta_dir` 下所有 `*.csv` 建索引（`dataset/text_video_audio_dataset.py`）。
单人旧 CSV 与多人新 CSV 可以放在同一个目录里混用：旧的走 N=1 退化路径（每个 GPU step 仍然是原来那套输入）。

## 一条样本 = 一个固定目标窗口

多人行用 `target_start_frame` **钉住**目标窗口，原因见下面的「参考音频不泄漏」。窗口的参考时刻是
**target_fps（默认 24）下的帧号**，不是源视频帧号。

## 列

| 列 | 必需 | 说明 |
|---|---|---|
| `video_path` | ✅ | 目标视频，**建议绝对路径**（`osp.join(data_root, abs)` 会直接用绝对路径） |
| `audio_path` | ✅ | 目标音频（16k 单声道混音），同样建议绝对路径 |
| `caption` | ✅ | 训练 prompt，`<S>…<E>` 包裹台词 |
| `num_frames` | ✅ | 源视频帧数，仅用于 dataset 的最低长度过滤 |
| `face_paths` / `feat_paths` | ✅ | **分号分隔**，每人一项，顺序 = 参考槽位顺序（slot 0 = 第一个人） |
| `spk_audio_paths` | ⭕ | 两级分割：人用 `;`，**同一个人的多段音频用 `,`**。缺失时退化为旧行为 |
| `spk_segments` | ⭕ | JSON：`[[[start,end], ...], ...]`，与 `spk_audio_paths` 逐段一一对应（秒，源时间轴） |
| `target_start_frame` | ⭕ | 固定目标窗口的起始帧（target_fps 帧号）；缺失则随机取窗口（旧行为） |
| `bbox` | ⭕ | 只对旧单人的 `ip_image` 生效；多人路径不使用（缺失也没问题） |

- `face_paths` 每一项可以是**单张图**（推荐）或**裁剪人脸视频**（随机取一帧）。
- `feat_paths` 每一项是预计算的 ArcFace 特征 `.pt`（`{"face_emb": [512]}` 或 `{"face_embs": [T,512], "valids": [T]}`），
  **必须用 antelopev2**（训练/推理用的就是它，标注流水线自己的 buffalo_l 向量不能用）。
- 参考音频会被裁剪/补齐到 `ref_audio_frames * 16000 / 24` 个采样（配置里的 `ref_audio_frames`）。

## 参考音频不泄漏（多人行的关键约束）

单人 recipe 刻意把参考音频取在目标窗口**之前**；多人行同样不能让参考里包含目标内容，否则模型直接“抄”参考即可。
所以多人行：

1. 用 `target_start_frame` 钉住窗口；
2. 数据集只从 **不与窗口重叠** 的 `spk_segments` 段里取参考音频（不足 `ref_audio_frames` 就报错，样本会被丢弃）。

代价是数据可用量取决于「每人有多少语音落在窗口外」。样例数据的实测（窗口内至少 1s 语音 + 每人窗口外至少 ref 秒）：

| 视频 | 时长 | 每人语音 | 121帧(5.0s)窗口 | 81帧(3.4s)窗口 |
|---|---|---|---|---|
| b0e0… | 11.95s | 2.24s / 2.60s | 需 ref ≤ 0.5s | 需 ref ≤ 1.5s |
| b41c… | 8.72s | 2.33s / 2.30s | 需 ref ≤ 1.0s | 需 ref ≤ 1.0s |

即：**窗口越长、参考越长，能用的样本越少**。样例里 `num_frames=121 + ref=2.0s` 的组合一条都用不了。

**当前工作点：`num_frames=121`（5.0s 目标）+ `ref_audio_frames=24`（1.0s 参考）**，样例实测 1/2 可用
（b41c 通过；b0e0 需要 ref ≤ 0.5s，c479 被 QA 拦下）。若真实语料的 yield 偏低，把
`--num-frames` 降到 81（3.4s 目标，与早期 `train.sh` 一致）可以把阈值放宽到 ref ≤ 1.5s —— 转换脚本会
逐条打印丢弃原因，跑一遍全量语料就能看出该往哪边调。

## 顺序约定（重点）

参考人脸与参考音频**靠列内位置配对**：第 i 个人的脸 ↔ 第 i 个人的音频，模型通过参考对的 AV 交叉注意力把二者绑定。
画面里「哪个人在左边」没有 token 级锚点，只能靠 caption 文本（建议显式写「左边的人 / 右边的人」）与数据统计学习。
**务必固定并长期遵守同一个约定**（例如「slot 0 = 第 0 帧最左边的脸」），否则学到的绑定会互相冲突。

## 从 avannotate 标注生成（`example_data` 那套数据）

> 全流程（特征 → meta CSV → 训练）可以直接用一键脚本：`bash run_multiperson_finetune.sh`，
> 只做数据准备用 `STEPS=meta bash run_multiperson_finetune.sh`。
> 它会把本次的 `n_refs` / `ref_audio_frames` 写进一份临时运行配置，保证转换参数与训练参数不会脱节。
> 下面是等价的、手工分步的命令。

三个根目录互相独立，全部用绝对路径：

```bash
# 1) 参考人脸特征（antelopev2，必须）
python dataset/extract_ref_face_feats.py \
    --annotation-root /abs/.../annotation_examples \
    --output-dir      /abs/.../ref_feats \
    --face-embedder-ckpt ./ckpts/InsightFace

# 2) 生成 meta CSV
python dataset/build_meta_from_avannotate.py \
    --annotation-root /abs/.../annotation_examples \
    --video-root      /abs/.../examples \
    --feat-dir        /abs/.../ref_feats \
    --output          /abs/.../meta/multiperson.csv \
    --list /abs/.../examples.txt --list-base . \
    --n-refs 2 --num-frames 121 --ref-audio-seconds 1.0
```

转换脚本做的事：读 `s11-compose/annotation.json`（utterances/word 时间戳/镜头描述）、`s7-tse/segments.json`
（每人分离后的干净语音）、`s6-associate` + `s10-caption` + `qa/report.json`，然后

- **槽位顺序 = face id 顺序**（F001 → slot 0）；只把「既说话、又有 TSE 音频」的身份当作槽位，数量必须等于 `n_refs`；
- 过滤：`qa.passed`、无画外人声（`offscreen_speakers == 0`）、无歧义关联；
- 选目标窗口：让每个人在窗口外都留够 `ref_audio_seconds`，且窗口内至少有 `min_target_speech_seconds` 语音；
- **caption = 整段场景描述 + 窗口内的台词**：global caption 与各 shot caption 保持完整（视觉描述覆盖全片，
  与预训练数据一致）；台词只保留落在目标窗口内的部分（逐词裁剪，含说话人与情绪标签），形如
  `… <F002> sad: <S>many of these trophies<E>` —— 这样 `<S>` 与模型要生成的音频逐字对齐；
  加 `--keep-all-dialogue` 可改为保留全片所有台词；
- 所有路径写成绝对路径，`target_start_frame`/`spk_segments` 一并落盘。

对应的训练配置（`configs/train/model_multiperson.yaml`）里：

```yaml
n_refs: 2
use_ref_av_fusion: true
ref_audio_frames: 24          # 必须与 --ref-audio-seconds * 24 一致
fix_prompt_with_asr: false    # 见下
```

> **为什么必须关掉 ASR 重写**：`modules/glm_asr/glmasr.py` 的 `re.sub(r"<S>\s*(.*)<E>", …, flags=re.S)`
> 是贪婪匹配，会把「第一个 `<S>` 到最后一个 `<E>`」之间的全部内容替换成一段混音转写 —— 多人 caption 里
> 夹在中间的 `<F001>` 标签会一起被吃掉。关掉之后 caption 与窗口的对齐由转换脚本保证。

推理侧要同步：`configs/inference/full_ovi_multiperson_5s.yaml` 的
`ref_audio_samples` 必须等于 `ref_audio_frames * 16000 / 24`（24 → 16000，48 → 32000，96 → 64000）。
