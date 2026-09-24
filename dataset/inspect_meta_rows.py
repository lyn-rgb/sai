"""按视频 id 查 meta CSV 的具体行，并用训练时的同一套算术复算「每人窗口外还有多少秒参考音频」。

    # 查指定的几个样本（id 可以是裸 hash、带 .mp4、或整条路径）
    python dataset/inspect_meta_rows.py --meta-dir /abs/multiperson_n2 \
        --num-frames 121 --ref-audio-frames 24 \
        --ids 5f0ad92cbcf8b797ec8b91a3ce5c8030 156d4679c9b35c3a10ee917397e29997

    # 看整体分布（有多少行贴阈值 / 不满足）
    python dataset/inspect_meta_rows.py --meta-dir /abs/multiperson_n2 \
        --num-frames 121 --ref-audio-frames 24 --all

用途：训练报 `person N ... has only 0.xx s of reference audio outside the target window` 时，
先用它判断问题在哪一侧——
  * 该 id **不在 CSV 里** → 训练读的不是这份 CSV（--meta_dir 指错，或期间的 CSV 被换过）；
  * 在 CSV 里且复算值 **< 阈值** → CSV 本身不满足约束（多半是用另一组
    --num-frames/--ref-audio-seconds 生成的，重建即可）；
  * 在 CSV 里且复算值 **≥ 阈值** → 加载层的问题，把这些行的 spk_segments / spk_audio_paths 原文贴出来。

算术与 `dataset/check_multiperson_dataset.py`、`dataset/text_video_audio_dataset.py` 共用同一份实现，
不存在三处各写一套的风险。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.build_meta_from_avannotate import video_id_of
from dataset.check_multiperson_dataset import parse_spk_segments, outside_seconds


def load_rows(csv_path: Path) -> dict[str, dict]:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index = {}
    for row in rows:
        index[video_id_of(Path(row["video_path"]))] = row
    return index


def describe(row: dict, num_frames: int, target_fps: float, ref_audio_seconds: float) -> str:
    spans_per_person = parse_spk_segments(row["spk_segments"])
    start_frame = int(row["target_start_frame"])
    window = (start_frame / target_fps, start_frame / target_fps + num_frames / target_fps)
    outside = [outside_seconds(spans, *window) for spans in spans_per_person]

    worst = min(outside) if outside else 0.0
    if worst < ref_audio_seconds * 0.98:
        verdict = "❌ 不足"
    elif worst < ref_audio_seconds * 1.05:
        verdict = "⚠️ 贴阈值"
    else:
        verdict = "✅ 充足"

    detail = ", ".join(f"P{i}={value:.2f}s" for i, value in enumerate(outside))
    return (f"  start_frame={start_frame:>4} 窗口={window[0]:.2f}-{window[1]:.2f}s "
            f"每人窗口外=[{detail}] 阈值={ref_audio_seconds:.2f}s  {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description="按 id 查 meta CSV 行并复算窗口外参考音频")
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None, help="直接指定 csv（默认取目录下第一个）")
    parser.add_argument("--ids", nargs="*", default=[], help="视频 id（裸 hash / <hash>.mp4 / 整条路径均可）")
    parser.add_argument("--ids-file", type=Path, default=None, help="从文件读 id（每行一个）")
    parser.add_argument("--all", action="store_true", help="输出所有行的分布统计")
    parser.add_argument("--limit", type=int, default=20, help="--all 时最多列出多少条最紧的行")
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--ref-audio-frames", type=int, default=24)
    args = parser.parse_args()

    ref_audio_seconds = args.ref_audio_frames / args.target_fps
    csv_path = args.csv
    if csv_path is None:
        candidates = sorted(args.meta_dir.glob("*.csv"))
        if not candidates:
            print(f"❌ {args.meta_dir} 下没有 csv")
            return 1
        if len(candidates) > 1:
            print(f"⚠️  {args.meta_dir} 下有 {len(candidates)} 个 csv，训练会全部加载："
                  f"{[c.name for c in candidates]}；本次只查 {candidates[0].name}")
        csv_path = candidates[0]
    params_path = csv_path.with_name(csv_path.name + ".params.json")

    rows = load_rows(csv_path)
    print(f"CSV: {csv_path}\n行数: {len(rows)}")
    if params_path.is_file():
        params = json.loads(params_path.read_text(encoding="utf-8"))
        print(f"生成参数: n_refs={params.get('n_refs')} num_frames={params.get('num_frames')} "
              f"ref_audio_seconds={params.get('ref_audio_seconds')}（保留 {params.get('kept_rows')} 行）")
        if (params.get("num_frames"), params.get("ref_audio_seconds")) != (args.num_frames, round(ref_audio_seconds, 4)):
            print("⚠️  与本次查询/训练参数不一致 → 这些行的窗口外参考音频很可能不够，需重建 CSV")
    else:
        print(f"⚠️  没有参数指纹 {params_path.name}（旧版 CSV），无法确认生成参数与本次是否一致")

    ids = list(args.ids)
    if args.ids_file is not None:
        ids.extend(line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines() if line.strip())

    exit_code = 0
    if ids:
        print(f"\n查询 {len(ids)} 个 id：")
        for raw in ids:
            video_id = video_id_of(Path(raw))
            row = rows.get(video_id)
            if row is None:
                print(f"  {video_id}: ❌ 不在这份 CSV 里（训练读的可能不是这个 --meta-dir，或 CSV 被换过）")
                exit_code = 1
                continue
            print(f"  {video_id}:")
            line = describe(row, args.num_frames, args.target_fps, ref_audio_seconds)
            print(line)
            if "❌" in line:
                exit_code = 1
            print(f"      spk_segments    : {row['spk_segments']}")
            print(f"      spk_audio_paths : {row['spk_audio_paths']}")

    if args.all:
        scored = []
        for video_id, row in rows.items():
            spans_per_person = parse_spk_segments(row["spk_segments"])
            start_frame = int(row["target_start_frame"])
            window = (start_frame / args.target_fps,
                      start_frame / args.target_fps + args.num_frames / args.target_fps)
            outside = [outside_seconds(spans, *window) for spans in spans_per_person]
            scored.append((min(outside) if outside else 0.0, video_id))

        below = [item for item in scored if item[0] < ref_audio_seconds * 0.98]
        tight = [item for item in scored if ref_audio_seconds * 0.98 <= item[0] < ref_audio_seconds * 1.05]
        print(f"\n整体分布（阈值 {ref_audio_seconds:.2f}s）：")
        print(f"  ❌ 不足 : {len(below)} 行")
        print(f"  ⚠️ 贴阈值: {len(tight)} 行")
        print(f"  ✅ 充足 : {len(scored) - len(below) - len(tight)} 行")
        for value, video_id in sorted(scored)[: args.limit]:
            print(f"    {video_id}: {value:.2f}s")
        if below:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
