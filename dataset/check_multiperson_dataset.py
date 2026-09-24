"""Pre-training sanity check for a multi-person meta CSV.

    python dataset/check_multiperson_dataset.py \
        --meta-dir  /abs/.../train_multiperson/multiperson_n2 \
        --feat-dir  /abs/.../train_multiperson/ref_feats \
        --n-refs 2 --num-frames 121 --ref-audio-frames 24 \
        --samples 3

Checks, in order:

A. 结构（全量）：必备列齐、每条样本的参考人数一致、引用的文件都在、caption 里有 `<S>`、
   `target_start_frame` 落在 [0, 总帧数-窗口) 内；
B. 参考脸 / 特征（全量）：参考图是 `--ref-size` 见方且不是纯色，`.pt` 是 512 维非零向量；
C. 参考音频约束（全量）：按 `spk_segments` + 目标窗口复算一遍「每人在窗口外还剩多少参考音频」，
   必须 ≥ 参考音频长度——这是训练时数据集会硬性断言的条件（转换脚本已经保证过一次，这里独立复算，
   避免两边算法漂移）。**两套口径都算**：纯区间算术，以及按 wav 文件实际长度裁剪（= 数据集真实的
   切片行为）；两者不等说明 `segments.json` 记的区间和音频文件长度对不上；
D. 真实加载（抽检 `--samples` 条）：直接实例化 `TextAudioVideoFaceDataset` 取样本，
   打印各张量的形状与取值范围，并确认 N 个参考不是同一个（防止退化样本）。

退出码 0 = 全部通过；1 = 有 error。warning 不会让脚本失败（例如 C 里贴着阈值下限的样本）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.ref_audio import obtainable_seconds, outside_seconds

REQUIRED_COLUMNS = {
    "video_path", "audio_path", "caption", "num_frames", "face_paths", "feat_paths",
    "spk_audio_paths", "spk_segments", "target_start_frame",
}


def parse_spk_segments(raw: str) -> list[list[list[float]]]:
    """`[[[start, end], ...], ...]` —— 每人一个区间列表。"""
    parsed = json.loads(raw)
    return [[[float(a), float(b)] for a, b in person] for person in parsed]


def check_structure(rows: list[dict], n_refs: int, target_fps: float, num_frames: int, errors: list, warnings: list):
    for index, row in enumerate(rows):
        tag = f"第 {index + 1} 行"
        face_paths = [p for p in row["face_paths"].split(";") if p]
        feat_paths = [p for p in row["feat_paths"].split(";") if p]
        spk_groups = [g for g in row["spk_audio_paths"].split(";") if g]
        if len({len(face_paths), len(feat_paths), len(spk_groups)}) != 1:
            errors.append(f"{tag}: face/feat/spk 数量不一致 ({len(face_paths)}/{len(feat_paths)}/{len(spk_groups)})")
            continue
        if len(face_paths) != n_refs:
            errors.append(f"{tag}: 参考人数 {len(face_paths)} != n_refs {n_refs}")
        for path in [*face_paths, *feat_paths, row["video_path"], row["audio_path"]]:
            if not Path(path).is_file():
                errors.append(f"{tag}: 文件不存在 {path}")
        for group in spk_groups:
            for path in group.split(","):
                if path and not Path(path).is_file():
                    errors.append(f"{tag}: 参考音频不存在 {path}")
        if "<S>" not in row["caption"] or "<E>" not in row["caption"]:
            errors.append(f"{tag}: caption 里没有 <S>…<E> 台词")
        start = int(row["target_start_frame"])
        if start < 0:
            errors.append(f"{tag}: target_start_frame 为负 ({start})")
        try:
            spans = parse_spk_segments(row["spk_segments"])
        except Exception as e:                                   # noqa: BLE001
            errors.append(f"{tag}: spk_segments 解析失败 ({e})")
            continue
        if len(spans) != len(face_paths):
            errors.append(f"{tag}: spk_segments 人数 {len(spans)} != 参考人数 {len(face_paths)}")
            continue
        # 窗口起点不该晚于最后一句台词（说明窗口选到了没内容的地方）
        latest = max((b for person in spans for _, b in person), default=0.0)
        if latest > 0 and start / target_fps > latest:
            warnings.append(f"{tag}: 窗口起点 {start / target_fps:.2f}s 晚于最后一句台词 {latest:.2f}s")


def check_outside_audio(rows: list[dict], target_fps: float, num_frames: int, ref_audio_seconds: float,
                        errors: list, warnings: list) -> None:
    """复算「每人在目标窗口外还剩多少参考音频」——训练时数据集会硬性断言这一条。

    两套口径都算，因为它们是链路两端各自的实现：
      * `outside_seconds`   —— 纯区间算术，用 CSV 里的 `spk_segments`（与转换脚本同源）；
      * `obtainable_seconds` —— 按 wav 文件实际长度裁剪（与数据集 `load_ref_audios` 同源）。
    只有后者会暴露「区间算术说够、文件里却没有这段音频」——也就是训练断言失败的真正原因。
    """
    for index, row in enumerate(rows):
        tag = f"第 {index + 1} 行"
        try:
            spans_per_person = parse_spk_segments(row["spk_segments"])
        except Exception:                                        # noqa: BLE001 - 结构检查里已报过
            continue
        groups = [group.split(",") for group in row["spk_audio_paths"].split(";") if group]
        if len(groups) != len(spans_per_person):
            continue                                             # 结构检查里已报过
        start_s = int(row["target_start_frame"]) / target_fps
        window = (start_s, start_s + num_frames / target_fps)
        for person_index, (paths, spans) in enumerate(zip(groups, spans_per_person)):
            outside = outside_seconds(spans, *window)
            obtainable, notes = obtainable_seconds(paths, spans, *window)
            # 数据集允许 2% 的取整误差（其余部分会补零），阈值与它对齐
            if obtainable < ref_audio_seconds * 0.98:
                errors.append(f"{tag}: 第 {person_index} 人窗口外实际只能取到 {obtainable:.2f}s 参考音频 "
                              f"(< {ref_audio_seconds:.2f}s 的 98%)，训练时会断言失败")
                for note in [n for n in notes if "⚠️" in n or "读不到" in n][:3]:
                    errors.append(f"{tag}:    {note}")
                continue
            if outside < ref_audio_seconds * 0.98:
                errors.append(f"{tag}: 第 {person_index} 人窗口外只有 {outside:.2f}s 参考音频 "
                              f"(区间算术 < {ref_audio_seconds:.2f}s 的 98%)；若实际可得 {obtainable:.2f}s 则说明 "
                              f"CSV 是用另一组 --num-frames/--ref-audio-seconds 生成的")
                continue
            if outside < ref_audio_seconds * 1.05 or obtainable < ref_audio_seconds * 1.05:
                warnings.append(f"{tag}: 第 {person_index} 人窗口外 {outside:.2f}s / 实际可得 "
                                f"{obtainable:.2f}s 参考音频，贴着阈值")
            if abs(outside - obtainable) > 0.15:
                warnings.append(f"{tag}: 第 {person_index} 人区间算术 {outside:.2f}s ≠ 实际可得 "
                                f"{obtainable:.2f}s（音频文件长度与 segments.json 的区间对不上）")


def check_references(rows: list[dict], ref_size: int, errors: list, warnings: list) -> None:
    """参考脸尺寸 / 特征维度与是否全零（需要 PIL 与 torch，训练环境里一定有）。"""
    try:
        import torch
        from PIL import Image
    except ImportError as e:                                     # 在没装 torch 的机器上只跑结构检查
        warnings.append(f"跳过参考脸/特征检查：{e}（请在训练环境里重跑本脚本）")
        return

    for index, row in enumerate(rows):
        tag = f"第 {index + 1} 行"
        for path in row["face_paths"].split(";"):
            if not path:
                continue
            with Image.open(path) as image:
                size = image.size
            if size != (ref_size, ref_size):
                errors.append(f"{tag}: 参考脸尺寸 {size[0]}x{size[1]} != {ref_size}x{ref_size}"
                              f"（可能是紧裁剪兜底产物）")
                break
        for path in row["feat_paths"].split(";"):
            if not path:
                continue
            feature = torch.load(path, map_location="cpu")
            embedding = feature.get("face_emb", feature.get("face_embs"))
            if embedding is None or embedding.numel() != 512:
                errors.append(f"{tag}: 参考特征维度异常 ({None if embedding is None else tuple(embedding.shape)})")
                break
            if float(embedding.abs().sum()) == 0:
                errors.append(f"{tag}: 参考特征全零 {path}")
                break


def check_reference_source(rows: list[dict], feat_dir: Path | None, errors: list, warnings: list) -> None:
    """参考脸是否都来自提取器产出的留白裁剪（`<feat_dir>/faces/`），否则尺度会差 2.2 倍。"""
    if feat_dir is None:
        return
    marker = f"{str(feat_dir).rstrip('/')}/faces/"
    fallback = sum(1 for row in rows if marker not in row["face_paths"])
    if fallback:
        warnings.append(f"{fallback}/{len(rows)} 行的参考脸不是留白裁剪（不在 {marker} 下）："
                        f"这些样本的参考尺度与预训练差 2.2 倍，建议 FORCE_FEATS=1 重算")


def check_dataloader(meta_dir: Path, data_root: str, n_refs: int, num_frames: int, ref_audio_frames: int,
                     height: int, width: int, samples: int, errors: list):
    """真的把数据集跑起来取几个样本——这一步会在开训前暴露路径/解码层面的问题。"""
    import torch
    from dataset.text_video_audio_dataset import TextAudioVideoFaceDataset

    min_video_len = math.ceil((ref_audio_frames + 24 / 2 + num_frames) / 24)
    dataset = TextAudioVideoFaceDataset(
        data_root=data_root or ".",        # csv 里是绝对路径时这个只作为兜底
        meta_dir=meta_dir,
        audio_sr=16000,
        ref_audio_frames=ref_audio_frames,
        normalize_audio=True,
        height=height,
        width=width,
        height_div=32,
        width_div=32,
        num_frames=num_frames,
        min_video_len=min_video_len,
        n_refs=n_refs,
    )
    print(f"\n[D] 真实加载抽检（数据集共 {len(dataset)} 条）")
    for index in range(min(samples, len(dataset))):
        try:
            sample = dataset[index]
        except Exception as e:                                                   # noqa: BLE001
            errors.append(f"加载第 {index} 条失败：{type(e).__name__}: {e}")
            continue
        ip_image = sample["ip_image"]
        ip_audio = sample["ip_audio"]
        if ip_image is None or sample["video"] is None:
            # 数据集内部重试 100 次仍失败时会返回全 None，说明这条样本读不出来
            errors.append(f"样本 {index}: 加载失败（数据集重试 100 次后仍返回空，检查视频/音频是否可解码）")
            continue
        shapes = {k: tuple(v.shape) for k, v in sample.items() if isinstance(v, torch.Tensor)}
        print(f"  样本 {index}: video{shapes['video']} audio{shapes['audio']} "
              f"ip_image{shapes['ip_image']} ip_audio{shapes['ip_audio']} "
              f"(embs {tuple(sample['ip_image_embs'].shape)}, ref_valid {sample['ref_valid'].tolist()})")
        if shapes["ip_image"][1] != n_refs:
            errors.append(f"样本 {index}: ip_image 参考数 {shapes['ip_image'][1]} != n_refs {n_refs}")
        if shapes["video"][1] != num_frames:
            errors.append(f"样本 {index}: 视频帧数 {shapes['video'][1]} != num_frames {num_frames}")
        # N 个参考不能是同一个（脚本对缺槽位会复制 slot 0，那种样本训练价值低）
        if n_refs > 1 and torch.allclose(ip_image[:, 0], ip_image[:, 1]):
            errors.append(f"样本 {index}: 参考 0 与参考 1 完全相同（槽位是复制的）")
        if n_refs > 1 and torch.allclose(ip_audio[0], ip_audio[1]):
            errors.append(f"样本 {index}: 参考音频 0 与 1 完全相同")
        if not (-1.01 <= float(ip_image.min()) and float(ip_image.max()) <= 1.01):
            errors.append(f"样本 {index}: 参考图取值超出 [-1, 1] "
                          f"({float(ip_image.min()):.3f}, {float(ip_image.max()):.3f})")


def main() -> int:
    parser = argparse.ArgumentParser(description="开训前自检多人 meta CSV 与参考素材")
    parser.add_argument("--meta-dir", type=Path, required=True,
                        help="包含 multiperson_meta.csv 的目录（与训练 --meta_dir 一致）")
    parser.add_argument("--meta-csv", type=Path, default=None, help="直接指定 csv（默认取目录下第一个）")
    parser.add_argument("--feat-dir", type=Path, default=None, help="参考素材目录（只用于提示，路径以 csv 为准）")
    parser.add_argument("--data-root", default=None, help="相对路径行的兜底根目录（csv 里是绝对路径时可省略）")
    parser.add_argument("--n-refs", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--ref-audio-frames", type=int, default=24, help="必须与训练配置一致（24 = 1.0s）")
    parser.add_argument("--ref-size", type=int, default=512)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--samples", type=int, default=3, help="真实加载抽检条数，0 = 跳过")
    parser.add_argument("--limit", type=int, default=None, help="只检查前 N 行（默认全量）")
    args = parser.parse_args()

    target_fps = 24.0
    ref_audio_seconds = args.ref_audio_frames / target_fps

    csv_path = args.meta_csv
    if csv_path is None:
        candidates = sorted(args.meta_dir.glob("*.csv"))
        if not candidates:
            print(f"❌ {args.meta_dir} 下没有 csv")
            return 1
        if len(candidates) > 1:
            print(f"⚠️  {args.meta_dir} 下有 {len(candidates)} 个 csv，训练时会被一起加载；取 {candidates[0].name} 检查")
        csv_path = candidates[0]

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        rows = list(reader)
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"检查 {csv_path}（{len(rows)} 行）")
    print(f"参数：n_refs={args.n_refs} num_frames={args.num_frames} "
          f"ref_audio={args.ref_audio_frames} 帧 ({ref_audio_seconds:.2f}s) ref_size={args.ref_size}")

    errors: list[str] = []
    warnings: list[str] = []
    missing = REQUIRED_COLUMNS - columns
    if missing:
        print(f"❌ 缺少列：{sorted(missing)}")
        return 1

    # 参数指纹：CSV 生成时的 num_frames / ref_audio_seconds / n_refs 与本次不一致时，
    # 窗口外的参考音频总量必然对不上——先报出来，免得把问题误判成数据集 bug
    params_path = csv_path.with_name(csv_path.name + ".params.json")
    if params_path.is_file():
        params = json.loads(params_path.read_text(encoding="utf-8"))
        mismatched = [
            f"{key}={params.get(key)}（CSV）!= {value}（本次）"
            for key, value in (("n_refs", args.n_refs), ("num_frames", args.num_frames),
                               ("ref_audio_seconds", round(args.ref_audio_frames / target_fps, 4)))
            if params.get(key) != value
        ]
        if mismatched:
            errors.append("meta CSV 的参数与本次检查/训练不一致：" + "；".join(mismatched)
                          + " → 用 STEPS=meta FORCE_META=1 按同一组参数重建")
        else:
            print(f"参数指纹一致（CSV 生成于 n_refs={params['n_refs']}, num_frames={params['num_frames']}, "
                  f"ref_audio_seconds={params['ref_audio_seconds']}，{params.get('kept_rows', '?')} 行）")
    else:
        warnings.append(f"没有参数指纹 {params_path.name}（旧版 CSV）：无法确认它与本次训练参数是否一致")

    def run_stage(title, function, *stage_args):
        """跑一个检查阶段，只报告本阶段新增的 error/warning 数（累计计数看不出是哪一步的）。"""
        before = (len(errors), len(warnings))
        print(f"\n{title}")
        function(*stage_args)
        print(f"    完成（error {len(errors) - before[0]}, warning {len(warnings) - before[1]}）")

    run_stage("[A] 结构与路径", check_structure,
              rows, args.n_refs, target_fps, args.num_frames, errors, warnings)
    run_stage("[B] 参考脸尺寸与参考特征", check_references, rows, args.ref_size, errors, warnings)
    run_stage("[B2] 参考脸来源（留白裁剪 vs 紧裁剪兜底）",
              check_reference_source, rows, args.feat_dir, errors, warnings)
    run_stage("[C] 窗口外参考音频复算", check_outside_audio,
              rows, target_fps, args.num_frames, ref_audio_seconds, errors, warnings)

    if args.samples > 0:
        check_dataloader(args.meta_dir, args.data_root, args.n_refs, args.num_frames,
                         args.ref_audio_frames, args.height, args.width, args.samples, errors)

    print("\n============================ 结果 ============================")
    if warnings:
        print(f"⚠️  {len(warnings)} 条 warning：")
        for message in warnings[:10]:
            print(f"    {message}")
        if len(warnings) > 10:
            print(f"    … 还有 {len(warnings) - 10} 条")
    if errors:
        print(f"❌ {len(errors)} 条 error：")
        for message in errors[:20]:
            print(f"    {message}")
        if len(errors) > 20:
            print(f"    … 还有 {len(errors) - 20} 条")
        print("结论：先修这些再开训（STEPS=meta FORCE_META=1 重建 CSV 或 FORCE_FEATS=1 重算参考脸）")
        return 1
    print("✅ 全部通过，可以开训：STEPS=train bash run_multiperson_finetune.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
