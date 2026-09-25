"""Turn rows of a multi-person training meta CSV into an inference test set.

    python evaluation/build_multiperson_testdata.py \
        --meta-csv /abs/.../multiperson_n2/multiperson_meta.csv \
        --testdata-dir /abs/.../testdata_multiperson \
        --ids <hash1> <hash2>            # 省略 = 取前 --limit 条（默认 10）

Produces exactly the layout `evaluation/build_testdata_prompt_csv.py --mode multiperson` expects:

    <testdata_dir>/full_video_prompt/<sample_id>_full_caption.txt   训练用的 caption（含 <F00X>: <S>…<E>）
    <testdata_dir>/frames/<sample_id>_frame_p{i}.jpg                第 i 个人的参考脸
    <testdata_dir>/audio/<sample_id>_audio_p{i}.wav                 第 i 个人的参考音频

Why not just point the engine at the training files? Because the **reference audio must be the same
slice the model was trained on**: the training reference is built from the person's speech that lies
*outside* the target window (`dataset/ref_audio.outside_pieces`), not from a whole segment file. This
script redoes that slicing with the very same helper, so the inference condition matches training.

Reference faces are copied as they are (the 2.2x margin crops from
`dataset/extract_ref_face_feats.py`) - that is the crop scale the recipe was trained with.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.build_meta_from_avannotate import video_id_of      # noqa: E402
from dataset.ref_audio import outside_pieces                     # noqa: E402


def read_wav(path: Path):
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM wav is supported (got {handle.getsampwidth() * 8}-bit)")
        frames = handle.readframes(handle.getnframes())
        return handle.getnchannels(), handle.getframerate(), frames


def write_wav(path: Path, channels: int, sample_rate: int, frames: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


def build_reference_audio(paths: list, spans: list, window: tuple, sample_rate: int,
                          needed_samples: int, target_path: Path) -> tuple:
    """The person's speech outside the target window, up to `needed_samples` (same rule as training).

    Returns `(written_samples, source_files, padded)`.
    """
    needed_bytes = needed_samples * 2
    pieces, sources, total, channels = [], [], 0, 1
    for index, raw_path in enumerate(paths):
        path = Path(raw_path)
        channels, rate, frames = read_wav(path)
        if rate != sample_rate:
            raise ValueError(f"{path}: {rate} Hz != {sample_rate} Hz (the training reference is 16 kHz)")
        span = spans[index] if index < len(spans) else None
        # 该文件里落在目标窗口之外的部分；没有区间信息时用整段
        byte_slices = [(0, len(frames))]
        if span is not None:
            byte_slices = [(max(0, int(round((start - float(span[0])) * sample_rate)) * 2 * channels),
                            min(len(frames), int(round((end - float(span[0])) * sample_rate)) * 2 * channels))
                           for start, end in outside_pieces(float(span[0]), float(span[1]), *window)]
        used = False
        for begin, end in byte_slices:
            if end > begin:
                pieces.append(frames[begin:end])
                total += end - begin
                used = True
        if used:
            sources.append(path.name)
        if total >= needed_bytes:                 # 与数据集一致：够了就不再取后面的文件
            break

    joined = b"".join(pieces)[:needed_bytes]
    padded = len(joined) < needed_bytes            # 与数据集一致：不足则补零（训练侧允许 2% 误差）
    joined = joined + b"\x00" * (needed_bytes - len(joined))
    write_wav(target_path, channels, sample_rate, joined)
    return len(joined) // (2 * channels), sources, padded


def build_sample(row: dict, sample_id: str, testdata_dir: Path, ref_audio_samples: int,
                 num_frames: int, target_fps: float = 24.0, sample_rate: int = 16000) -> dict:
    face_paths = [p for p in row["face_paths"].split(";") if p]
    spk_groups = [g.split(",") for g in row["spk_audio_paths"].split(";") if g]
    if not face_paths or not spk_groups or len(face_paths) != len(spk_groups):
        raise ValueError(f"{sample_id}: face_paths / spk_audio_paths 数量不一致或为空")
    spans_per_person = json.loads(row["spk_segments"]) if row.get("spk_segments") else \
        [[] for _ in spk_groups]

    start_s = int(row["target_start_frame"]) / target_fps
    window = (start_s, start_s + num_frames / target_fps)

    notes = []
    for person_index, (face_path, paths) in enumerate(zip(face_paths, spk_groups)):
        frame_target = testdata_dir / "frames" / f"{sample_id}_frame_p{person_index}.jpg"
        frame_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(face_path, frame_target)
        audio_target = testdata_dir / "audio" / f"{sample_id}_audio_p{person_index}.wav"
        spans = spans_per_person[person_index] if person_index < len(spans_per_person) else []
        if not spans:                                 # 没有区间信息：直接用第一段整段
            channels, rate, frames = read_wav(Path(paths[0]))
            write_wav(audio_target, channels, rate, frames)
            written, sources, padded = len(frames) // (2 * channels), [Path(paths[0]).name], False
        else:
            written, sources, padded = build_reference_audio(
                paths, spans, window, sample_rate, ref_audio_samples, audio_target)
        notes.append(f"p{person_index}: {written} 采样 ({written / sample_rate:.2f}s)"
                     + ("，尾部补零（窗口外语音不足）" if padded else "")
                     + f" ← {', '.join(sources) if sources else '无'}")
    prompt_target = testdata_dir / "full_video_prompt" / f"{sample_id}_full_caption.txt"
    prompt_target.parent.mkdir(parents=True, exist_ok=True)
    prompt_target.write_text(row["caption"].strip() + "\n", encoding="utf-8")
    return {"sample_id": sample_id, "caption": row["caption"], "notes": notes}


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 CSV → 多人推理用的 testdata 目录")
    parser.add_argument("--meta-csv", type=Path, required=True, help="训练用的 multiperson_meta.csv")
    parser.add_argument("--testdata-dir", type=Path, required=True)
    parser.add_argument("--ids", nargs="*", default=[], help="视频 id（裸 hash / <hash>.mp4 / 整条路径）")
    parser.add_argument("--limit", type=int, default=10, help="没给 --ids 时取前几条")
    parser.add_argument("--num-frames", type=int, default=121,
                        help="必须与训练一致：参考音频按「目标窗口之外」切片需要它")
    parser.add_argument("--ref-audio-frames", type=int, default=24)
    parser.add_argument("--target-fps", type=float, default=24.0)
    args = parser.parse_args()

    ref_audio_samples = int(round(args.ref_audio_frames / args.target_fps * 16000))
    with open(args.meta_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index = {video_id_of(Path(row["video_path"])): row for row in rows}
    sample_ids = [video_id_of(Path(raw)) for raw in args.ids] or \
        [video_id_of(Path(row["video_path"])) for row in rows[: args.limit]]

    print(f"CSV: {args.meta_csv}（{len(rows)} 行）→ testdata: {args.testdata_dir}")
    print(f"参考音频 {ref_audio_samples} 采样（{args.ref_audio_frames} 帧 @ {args.target_fps}fps），"
          f"窗口 {args.num_frames} 帧")
    kept, seen = 0, set()
    for sample_id in sample_ids:
        if sample_id in seen:      # CSV 里同一个视频出现两次：testdata 只能存一份，跳过并说明
            print(f"  ⚠️  {sample_id}: 重复出现，跳过（同名样本会互相覆盖）")
            continue
        seen.add(sample_id)
        row = index.get(sample_id)
        if row is None:
            print(f"  ❌ {sample_id}: 不在这份 CSV 里")
            continue
        try:
            info = build_sample(row, sample_id, args.testdata_dir, ref_audio_samples,
                                args.num_frames, args.target_fps)
        except Exception as e:                                           # noqa: BLE001
            print(f"  ❌ {sample_id}: {type(e).__name__}: {e}")
            continue
        kept += 1
        print(f"  ✅ {sample_id}: {len(info['notes'])} 人")
        for note in info["notes"]:
            print(f"        {note}")
        print(f"        caption: {info['caption'][:110]}{'…' if len(info['caption']) > 110 else ''}")
    if not kept:
        print("❌ 一条都没生成", file=sys.stderr)
        return 1
    print(f"\n共 {kept} 条样本 → 接下来生成 prompt CSV：\n"
          f"  python evaluation/build_testdata_prompt_csv.py --mode multiperson \\\n"
          f"      --testdata-dir {args.testdata_dir} --output {args.testdata_dir}/generated_prompts/multiperson_prompts.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
