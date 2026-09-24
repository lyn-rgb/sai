"""Regression test for the reference-audio window rule (stdlib only, no torch needed).

    python tests/test_ref_audio_clamp.py

Why this exists: `segments.json` records `[start, end]` per extracted speech segment, but the wav
file next to it is not guaranteed to be that long (real corpora do contain short/overwritten files).
The converter used to trust the interval, choose the target window from it and write it into the CSV,
while the dataset slices the audio by **file length** - so a row could look fine at conversion time
and then fail at training with

    person 0 of ... has only 0.00s of reference audio outside the target window ...

The fix is that the converter now clamps every segment end to the real wav duration before choosing
the window. This test pins it with a synthetic annotation tree:

1. files match their recorded intervals      -> row is written, both arithmetics agree;
2. a file is shorter than its interval       -> the written spans are clamped, so the dataset's own
   rule (`obtainable >= 98% of the reference length`) still holds - and the *unclamped* arithmetic
   would have broken it (that assertion is what makes this test meaningful);
3. a file is far too short to be usable      -> the identity is dropped with a reason instead of the
   row being written with an unreachable guarantee.
"""
from __future__ import annotations

import json
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset.build_meta_from_avannotate as converter
from dataset.ref_audio import obtainable_seconds, outside_seconds, wav_seconds

FPS = 24.0
NUM_FRAMES = 121
REF_SECONDS = 1.0              # = ref_audio_frames 24 @ 24fps, must match the training config
DURATION = 12.0                # seconds of the synthetic clip

# (person, [(span_start, span_end, wav_seconds), ...]) - the wav is written separately so a test can
# make it shorter than the span it belongs to
SPEECH = {
    "F001": [((0.0, 2.0), 2.0), ((9.0, 11.0), 2.0)],
    "F002": [((0.5, 2.5), 2.0), ((9.5, 11.5), 2.0)],
}

FAILURES: list = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"  {'✅' if condition else '❌'} {label}" + (f"  {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def write_wav(path: Path, seconds: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(round(seconds * sample_rate)))


def build_fixture(root: Path, shorten: dict | None = None) -> tuple:
    """A minimal avannotate tree. `shorten` maps a segment name to the wav length to write."""
    shorten = shorten or {}
    annotation_root = root / "annotations"
    video_root = root / "videos"
    feat_dir = root / "feats"
    video_id = "fixture0000"
    annotation_dir = annotation_root / video_id
    (video_root).mkdir(parents=True, exist_ok=True)
    (video_root / f"{video_id}.mp4").write_bytes(b"")
    (feat_dir).mkdir(parents=True, exist_ok=True)

    frame_count = int(round(DURATION * FPS))
    (annotation_dir / "s0-preprocess").mkdir(parents=True, exist_ok=True)
    (annotation_dir / "s0-preprocess" / "probe.json").write_text(json.dumps({
        "timeline": {"duration": DURATION, "fps": FPS, "frame_count": frame_count, "sample_rate": 16000},
    }), encoding="utf-8")
    write_wav(annotation_dir / "s0-preprocess" / "mix.wav", DURATION)

    segments, utterances = [], []
    for face_id, entries in SPEECH.items():
        for index, (span, file_seconds) in enumerate(entries):
            name = f"{face_id}_{index:04d}"
            write_wav(annotation_dir / "s7-tse" / "audio" / face_id / f"{name}.wav",
                      shorten.get(name, file_seconds))
            segments.append({
                "identity": face_id, "index": index, "name": name,
                "start": span[0], "end": span[1], "duration": span[1] - span[0],
                "flags": [], "audio": f"s7-tse/audio/{face_id}/{name}.wav",
                "sample_rate": 16000, "rms": 0.1, "silent": False,
            })
            utterances.append({
                "face_id": face_id, "start": span[0], "end": span[1],
                "text": f"line {index} of {face_id}", "tag": None,
                "audio_path": f"s7-tse/audio/{face_id}/{name}.wav",
                "words": [{"text": f"line {index} of {face_id}", "start": span[0], "end": span[1]}],
            })
    (annotation_dir / "s7-tse").mkdir(parents=True, exist_ok=True)
    (annotation_dir / "s7-tse" / "segments.json").write_text(json.dumps({"segments": segments}),
                                                           encoding="utf-8")

    (annotation_dir / "s11-compose").mkdir(parents=True, exist_ok=True)
    (annotation_dir / "s11-compose" / "annotation.json").write_text(json.dumps({
        "video": {"video_id": video_id, "duration": DURATION, "fps": FPS},
        "shots": [{"index": 1, "start": 0.0, "end": DURATION, "caption": "Two people talking."}],
        "utterances": utterances,
        "face_tracks": [{"face_id": face_id, "speaks": True} for face_id in SPEECH],
        "global_caption": "Two people are talking in a bright room.",
    }), encoding="utf-8")

    # the converter only checks that the reference material exists; the margin crops are optional
    (annotation_dir / "s3-cluster" / "faces").mkdir(parents=True, exist_ok=True)
    for face_id in SPEECH:
        (annotation_dir / "s3-cluster" / "faces" / f"{face_id}.jpg").write_bytes(b"")
        (feat_dir / f"{video_id}_{face_id}.pt").write_bytes(b"")

    return annotation_dir, video_root / f"{video_id}.mp4", feat_dir


def convert(annotation_dir: Path, video_path: Path, feat_dir: Path, min_per_person_seconds: float = 0.0):
    adjustments: list = []
    row, reason = converter.build_row(
        annotation_dir, video_path, feat_dir, n_refs=len(SPEECH), num_frames=NUM_FRAMES,
        target_fps=FPS, ref_seconds=REF_SECONDS, require_qa_pass=False,
        allow_offscreen_speech=True, adjustments=adjustments,
        min_per_person_seconds=min_per_person_seconds)
    return row, reason, adjustments


def window_of(row: dict) -> tuple:
    start = int(row["target_start_frame"]) / FPS
    return start, start + NUM_FRAMES / FPS


def person_seconds(row: dict) -> tuple:
    """`(per-person obtainable seconds from the files, per-person seconds from the written spans)`."""
    spans_per_person = json.loads(row["spk_segments"])
    groups = [group.split(",") for group in row["spk_audio_paths"].split(";")]
    obtainable, spans = [], []
    for paths, person_spans in zip(groups, spans_per_person):
        seconds, _ = obtainable_seconds(paths, person_spans, *window_of(row))
        obtainable.append(seconds)
        spans.append(outside_seconds(person_spans, *window_of(row)))
    return obtainable, spans


def scenario_matching_files(root: Path) -> None:
    print("\n[1] wav 文件与记录的区间一致 → 行应当被写出，两套算术一致")
    annotation_dir, video_path, feat_dir = build_fixture(root)
    row, reason, adjustments = convert(annotation_dir, video_path, feat_dir)
    check(row is not None, "写出了一行", f"drop 原因：{reason}")
    if row is None:
        return
    check(not adjustments, "没有区间被收紧", str(adjustments))
    obtainable, spans = person_seconds(row)
    check(all(seconds >= REF_SECONDS for seconds in obtainable),
          "每人窗口外实际可得 ≥ 参考长度", f"{obtainable}")
    check(all(abs(a - b) < 1e-6 for a, b in zip(obtainable, spans)),
          "两套算术完全一致", f"可得 {obtainable} / 区间 {spans}")
    # 窗口必须留下真正的台词，而不只是「重叠秒数」：这里的 fixture 被有意造成
    # 「窗口外分数最高的窗口恰好只擦到语音边缘」，旧口径（按重叠算）会选中它并写出没有 <S> 的 caption
    annotation = json.loads((annotation_dir / "s11-compose" / "annotation.json").read_text(encoding="utf-8"))
    turns = converter.caption_utterances(annotation, *window_of(row), set(SPEECH), True)
    check(bool(turns), "窗口内有整词台词（不是只擦到语音边缘）", str(turns))
    check("<S>" in row["caption"] and "<E>" in row["caption"], "caption 里有 <S>…<E>")
    in_window = sum(w["end"] - w["start"] for u in annotation["utterances"] for w in u["words"]
                    if window_of(row)[0] <= (w["start"] + w["end"]) / 2 < window_of(row)[1])
    check(in_window >= 1.0, "窗口内整词语音 ≥ min_target_seconds 1.0s", f"{in_window:.2f}s")
    print(f"      窗口 {window_of(row)[0]:.2f}-{window_of(row)[1]:.2f}s，台词：{turns}")


def scenario_short_file(root: Path) -> None:
    print("\n[2] wav 比记录的区间短（2.0s 的段只有 1.0s 音频）→ 写出的区间必须按文件收紧")
    annotation_dir, video_path, feat_dir = build_fixture(root, shorten={"F001_0000": 1.0})
    # 未收紧时会写成什么样：区间算术（旧转换脚本 + 旧 checker 的口径）
    raw_spans = json.loads((annotation_dir / "s7-tse" / "segments.json").read_text(encoding="utf-8"))
    raw = [(s["identity"], s["start"], s["end"]) for s in raw_spans["segments"]]
    check(any(face_id == "F001" and abs(end - start - 2.0) < 1e-9 for face_id, start, end in raw),
          "fixture 里记录的 F001 首段仍是 2.0s（问题前提成立）")

    row, reason, adjustments = convert(annotation_dir, video_path, feat_dir)
    check(bool(adjustments), "转换器报告了被收紧的区间", str(adjustments))
    check(row is not None, "写出了一行", f"drop 原因：{reason}")
    if row is None:
        return

    obtainable, spans = person_seconds(row)
    check(all(abs(a - b) < 1e-6 for a, b in zip(obtainable, spans)),
          "写出的区间与文件长度一致（两套算术不再漂移）", f"可得 {obtainable} / 区间 {spans}")
    check(all(seconds >= REF_SECONDS * 0.98 for seconds in obtainable),
          "满足数据集断言（obtainable ≥ 98% 参考长度）", f"{obtainable}")
    # 下面两行是这个场景的意义所在：不收紧的话，旧口径会给出文件里取不到的秒数
    written = json.loads(row["spk_segments"])
    check(written[0][0][1] == 1.0, "F001 首段的区间被写成文件长度 1.0s", str(written[0]))
    unclamped = outside_seconds([(0.0, 2.0), (9.0, 11.0)], *window_of(row))   # segments.json 原文
    print(f"      F001 窗口外：区间算术 {unclamped:.2f}s（旧口径，实际取不到）"
          f" → 实际可得 {obtainable[0]:.2f}s（新口径）")


def scenario_unusable_file(root: Path) -> None:
    print("\n[3] wav 短到不可用 → 这个身份应当被丢弃（而不是写出一个取不到的保证）")
    annotation_dir, video_path, feat_dir = build_fixture(
        root, shorten={"F002_0000": 0.02, "F002_0001": 0.02})
    row, reason, adjustments = convert(annotation_dir, video_path, feat_dir)
    check(row is None, "没有写出这一行", "两段音频都不可用却仍然写出了行")
    check(reason is not None and "identity_count" in reason,
          "drop 原因说明是可用身份数不足", str(reason))
    check(bool(adjustments), "转换器报告了被丢掉的语音段", str(adjustments))


def scenario_require_every_speaker(root: Path) -> None:
    print("\n[4] 要求「窗口内两个人都说话」→ 必须放弃窗口外分数最高、但只有一个人说话的窗口")
    annotation_dir, video_path, feat_dir = build_fixture(root)
    annotation = json.loads((annotation_dir / "s11-compose" / "annotation.json").read_text(encoding="utf-8"))

    def per_person_words(row: dict) -> list:
        """每人窗口内的整词秒数（caption 取词用的是同一个规则）。"""
        window = window_of(row)
        totals = [0.0 for _ in SPEECH]
        for index, face_id in enumerate(sorted(SPEECH)):
            for utterance in annotation["utterances"]:
                if utterance["face_id"] != face_id:
                    continue
                for word in utterance["words"]:
                    if window[0] <= (word["start"] + word["end"]) / 2 < window[1]:
                        totals[index] += word["end"] - word["start"]
        return totals

    loose, _, _ = convert(annotation_dir, video_path, feat_dir)                 # 默认 0：只看总量
    strict, _, _ = convert(annotation_dir, video_path, feat_dir, 0.3)
    check(loose is not None and strict is not None, "两种设置都能写出样本")
    if loose is None or strict is None:
        return
    loose_words, strict_words = per_person_words(loose), per_person_words(strict)
    print(f"      不加门：窗口 {window_of(loose)[0]:.2f}s 起，每人窗口内整词 {[round(v, 2) for v in loose_words]}")
    print(f"      加门  ：窗口 {window_of(strict)[0]:.2f}s 起，每人窗口内整词 {[round(v, 2) for v in strict_words]}")
    check(min(loose_words) < 0.3, "不加门时确实会选到「有人没说话」的窗口（问题前提成立）")
    check(min(strict_words) >= 0.3, "加门后每人窗口内整词都 ≥ 0.3s（caption 里两人都有台词）",
          f"{strict_words}")
    turns = converter.caption_utterances(annotation, *window_of(strict), set(SPEECH), True)
    check(len(turns) >= 2, "加门后的 caption 里有两句台词", str(turns))
    # 加门的代价：窗口外参考音频变少，但仍必须满足约束
    obtainable, _ = person_seconds(strict)
    check(all(seconds >= REF_SECONDS for seconds in obtainable),
          "加门后仍满足「每人窗口外 ≥ 参考长度」", f"{obtainable}")


def main() -> int:
    print("参考音频窗口口径回归测试")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scenario_matching_files(root / "a")
        scenario_short_file(root / "b")
        scenario_unusable_file(root / "c")
        scenario_require_every_speaker(root / "d")
    if FAILURES:
        print(f"\n❌ {len(FAILURES)} 项失败：")
        for label in FAILURES:
            print(f"    {label}")
        return 1
    print("\n✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
