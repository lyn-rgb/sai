"""How many people actually talk inside the target window of each training row.

    python dataset/audit_window_speakers.py \
        --meta-csv /abs/.../multiperson_n2/multiperson_meta.csv \
        --num-frames 121 --ref-audio-frames 24

Why: the audio the model is asked to generate is the **target window**. The caption only carries the
`<S>…<E>` lines of the people who speak inside that window (see `caption_utterances`), so a window
where only one person speaks teaches "one speaker, one `<S>` line" - and the model then generates
single-speaker audio even when the prompt asks for two. Window selection maximises the reference
audio *outside* the window, which biases the window towards the quietest spot, so this is easy to hit
by accident.

The counts here come from `spk_segments` (interval arithmetic), which is the same data the caption is
built from, so `窗口内≥X 秒` is a good proxy for "this person has a `<S>` line in the caption".

Exit code 1 when most rows are single-speaker (a hint that `--min-per-person-speech-seconds` should be
used and the CSV rebuilt).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.check_multiperson_dataset import parse_spk_segments      # noqa: E402

MIN_SPEAKING_SECONDS = 0.2      # 少于这个秒数基本不会留下整词，caption 里就不会有这个人的台词


def in_window_seconds(spans: list, window_start: float, window_end: float) -> float:
    total = 0.0
    for start, end in spans:
        total += max(0.0, min(end, window_end) - max(start, window_start))
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="统计每条样本目标窗口内有几个人在说话")
    parser.add_argument("--meta-csv", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--min-speaking-seconds", type=float, default=MIN_SPEAKING_SECONDS,
                        help="窗口内说话超过这么多秒才算「这个人在说」（默认 0.2）")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--examples", type=int, default=3, help="单说话人样本里打印几条例子")
    args = parser.parse_args()

    with open(args.meta_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print(f"❌ {args.meta_csv} 里没有行")
        return 1

    window_seconds = args.num_frames / args.target_fps
    speaking_hist: dict[int, int] = {}
    multi, single = [], []
    per_person_seconds: list[float] = []
    for row in rows:
        try:
            spans_per_person = parse_spk_segments(row["spk_segments"]) if row.get("spk_segments") else []
        except Exception:                                            # noqa: BLE001
            spans_per_person = []
        start = int(row["target_start_frame"]) / args.target_fps
        window = (start, start + window_seconds)
        seconds = [in_window_seconds(spans, *window) for spans in spans_per_person]
        if not seconds:
            continue
        speakers = sum(1 for value in seconds if value >= args.min_speaking_seconds)
        speaking_hist[speakers] = speaking_hist.get(speakers, 0) + 1
        per_person_seconds.extend(seconds)
        (multi if speakers >= 2 else single).append((row["video_path"].split("/")[-1], seconds))

    total = sum(speaking_hist.values())
    print(f"CSV            : {args.meta_csv}")
    print(f"目标窗口       : {args.num_frames} 帧 = {window_seconds:.2f}s（判定阈值 {args.min_speaking_seconds}s）")
    print(f"统计样本数     : {total} / {len(rows)}")
    print("\n窗口内说话人数分布：")
    for speakers in sorted(speaking_hist):
        count = speaking_hist[speakers]
        bar = "█" * max(1, round(count / total * 40))
        print(f"  {speakers} 人: {count:>5} 行 ({count / total * 100:5.1f}%) {bar}")
    if per_person_seconds:
        values = sorted(per_person_seconds)
        print(f"\n每人窗口内秒数：中位 {values[len(values) // 2]:.2f}s，"
              f"10% 分位 {values[len(values) // 10]:.2f}s，最大 {values[-1]:.2f}s")
    if single and args.examples:
        print(f"\n单说话人样本示例（{len(single)} 行）：")
        for name, seconds in single[: args.examples]:
            print(f"  {name}: 每人窗口内 = {[round(v, 2) for v in seconds]}")
    print("\n⚠️  窗口内只有一个人说话的样本，caption 里就只有一句 <S> —— 模型学到的是"
          "「生成一个人说话」。要真正做双人对话，重建 CSV 时加 --min-per-person-speech-seconds 0.3~0.5"
          "（一键脚本：MIN_PER_PERSON_SPEECH_SECONDS=0.3 ...），并注意 yield 会下降。")
    return 0 if not single or len(single) / max(total, 1) < 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
