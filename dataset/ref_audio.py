"""参考音频的时间轴工具：窗口外片段与"窗口外还剩多少秒"。

被三处共用，保证算术只有一份、不会各自漂移：
  * `text_video_audio_dataset`  —— 训练时按它切片（`outside_pieces`）
  * `build_meta_from_avannotate` —— 转换时按它判定窗口可行性（`outside_seconds`）
  * `check_multiperson_dataset` / `inspect_meta_rows` —— 自检与排查（两者）

只依赖标准库，所以自检脚本在没有 torch 的机器上也能跑。
"""
from __future__ import annotations

import wave


def outside_pieces(start: float, end: float, window_start: float, window_end: float) -> list:
    """`(start, end)` 落在目标窗口之外的部分（窗口落在段中间时切成两段）。"""
    pieces = []
    if start < window_start:
        pieces.append((start, min(end, window_start)))
    if end > window_end:
        pieces.append((max(start, window_end), end))
    return [(a, b) for a, b in pieces if b > a]


def outside_seconds(spans, window_start: float, window_end: float) -> float:
    """若干 `[start, end]` 区间里，落在目标窗口之外的语音总秒数。"""
    total = 0.0
    for start, end in spans:
        overlap = max(0.0, min(end, window_end) - max(start, window_start))
        total += (end - start) - overlap
    return total


def wav_seconds(path) -> float:
    """WAV 文件的实际时长（只读头部，无第三方依赖）。"""
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def obtainable_seconds(paths, spans, window_start: float, window_end: float,
                       sample_rate: int = 16000):
    """按 `text_video_audio_dataset.load_ref_audios` 的**真实切片逻辑**算能取到多少秒参考音频。

    与 `outside_seconds`（纯区间算术）的区别：这里按文件实际长度裁剪，
    因此能暴露「区间算术说够、文件里却没有这段音频」的情况。
    返回 `(秒数, 每个文件的一致性说明列表)`。
    """
    total, notes = 0.0, []
    for index, path in enumerate(paths):
        span = spans[index] if index < len(spans) else None
        try:
            file_seconds = wav_seconds(path)
        except Exception as e:                                   # noqa: BLE001
            notes.append(f"{path}: 读不到 ({e})")
            continue
        if span is None:
            total += file_seconds
            notes.append(f"{path}: 无区间，整段 {file_seconds:.2f}s")
            continue
        span_start, span_end = float(span[0]), float(span[1])
        span_seconds = span_end - span_start
        notes.append(f"{path}: " + ("✅" if abs(file_seconds - span_seconds) <= 0.15
                                    else f"⚠️ 文件 {file_seconds:.2f}s ≠ 区间 {span_seconds:.2f}s"))
        for piece_start, piece_end in outside_pieces(span_start, span_end, window_start, window_end):
            begin = max(0, int(round((piece_start - span_start) * sample_rate)))
            finish = min(int(round(file_seconds * sample_rate)),
                         int(round((piece_end - span_start) * sample_rate)))
            if finish > begin:
                total += (finish - begin) / sample_rate
    return total, notes
