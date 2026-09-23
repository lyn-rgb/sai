"""Build a multi-person training meta CSV from `avannotate` annotation trees.

    python dataset/build_meta_from_avannotate.py \
        --annotation-root /abs/example_data/annotation_examples \
        --video-root      /abs/example_data/examples \
        --feat-dir        /abs/example_data/ref_feats \
        --output          /abs/example_data/meta/multiperson_meta.csv \
        --n-refs 2 --num-frames 121 --ref-audio-seconds 2.0

Input layout (one directory per video, as produced by the annotation pipeline):

    <annotation_root>/<video_id>/s0-preprocess/probe.json     timeline: fps, frame_count, duration
    <annotation_root>/<video_id>/s0-preprocess/mix.wav        mixed 16 kHz audio = the target audio
    <annotation_root>/<video_id>/s3-cluster/faces/F00X.jpg    per-person reference face
    <annotation_root>/<video_id>/s6-associate/assignments.json  speaker <-> face association
    <annotation_root>/<video_id>/s7-tse/segments.json         per-person extracted (clean) speech
    <annotation_root>/<video_id>/s7-tse/audio/F00X/*.wav      the extracted speech files
    <annotation_root>/<video_id>/s10-caption/captions.json    shot captions
    <annotation_root>/<video_id>/s11-compose/annotation.json  utterances + global/shot captions
    <annotation_root>/<video_id>/s11-compose/qa/report.json   quality gates and metrics

The video is taken from `<video_root>/<video_id>` (or `<video_id>.<ext>`, or from `--list`) - the
clips of the OpenHumanVid style corpora carry **no file extension**, so both forms are accepted and
`--video-exts` controls which suffixes are tried. The annotation's own `video.path` is the
producer's cluster path and is not usable locally. **Every path written into the CSV is absolute**,
so the data root and the annotation root can live in different places and `--data_root` at training
time is only a fallback for relative rows.

Conventions (see `dataset/MULTIPERSON_DATA.md`):
- Slot order = face id order (F001 -> slot 0, F002 -> slot 1, ...). Only identities that both speak
  and have extracted audio become slots; a clip must have exactly `n_refs` of them.
- The target window is pinned per row (`target_start_frame`, in **target-fps** frames) and chosen so
  that every person has enough reference audio **outside** it - the reference must not contain the
  target - while the window itself still holds enough speech to train on.
- The caption concatenates the whole scene description (global caption + shot captions) with the
  dialogue of the target window only, as `<F00X> tag: <S>text<E>` - so the `<S>` spans match the
  audio the model has to generate. `--keep-all-dialogue` keeps every line of the clip instead.
- Reference features must be computed with antelopev2 first (`dataset/extract_ref_face_feats.py`).
- `fix_prompt_with_asr` must be false for these rows: the ASR rewrite regex is greedy and would
  replace everything between the first `<S>` and the last `<E>`, deleting the speaker tags.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COLUMNS = [
    "video_path", "audio_path", "caption", "num_frames", "bbox",
    "face_paths", "feat_paths", "spk_audio_paths", "spk_segments", "target_start_frame",
]

# 语料里的视频常常不带扩展名（例如 `.../clips/<hash>`），这里按「原样 → 补扩展名 → 去扩展名」依次尝试
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v")


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def video_id_of(path: Path, exts=VIDEO_EXTS) -> str:
    """`<id>.mp4` 和无扩展名的 `<id>` 都要映射到 `<id>`。"""
    return path.stem if path.suffix.lower() in exts else path.name


def resolve_video_path(path: Path, exts=VIDEO_EXTS):
    """定位视频文件，兼容「列表写了扩展名但文件没有」与「列表没写扩展名」两种情况。"""
    if path.is_file():
        return path
    if path.suffix.lower() in exts:
        bare = path.with_suffix("")
        if bare.is_file():
            return bare
    for ext in exts:
        candidate = path.with_name(path.name + ext)
        if candidate.is_file():
            return candidate
    return None


def resampled_frame_count(probe: dict, target_fps: float) -> int:
    timeline = probe["timeline"]
    return int(timeline["frame_count"] // (timeline["fps"] / target_fps))


def speaking_identities(segments: dict) -> dict[str, list[dict]]:
    """Per person (face id) the extracted speech segments, in face id order."""
    by_identity: dict[str, list[dict]] = {}
    for segment in segments.get("segments", []):
        if segment.get("silent"):
            continue
        by_identity.setdefault(segment["identity"], []).append(segment)
    for spans in by_identity.values():
        spans.sort(key=lambda s: s["start"])
    return {face_id: by_identity[face_id] for face_id in sorted(by_identity)}


def outside_seconds(segments: list[dict], start_s: float, end_s: float) -> float:
    """How much of a person's speech lies outside the target window."""
    total = 0.0
    for segment in segments:
        length = segment["end"] - segment["start"]
        overlap = max(0.0, min(segment["end"], end_s) - max(segment["start"], start_s))
        total += length - overlap
    return total


def inside_seconds(segments_by_person: dict[str, list[dict]], start_s: float, end_s: float) -> float:
    """How much speech (any person) lies inside the target window."""
    total = 0.0
    for spans in segments_by_person.values():
        for segment in spans:
            total += max(0.0, min(segment["end"], end_s) - max(segment["start"], start_s))
    return total


def choose_window(segments_by_person: dict[str, list[dict]], total_resampled: int, num_frames: int,
                  target_fps: float, ref_seconds: float, min_target_seconds: float = 1.0,
                  step_seconds: float = 0.25):
    """Target window that maximises the smallest per-person amount of outside speech.

    A window is only usable when every person keeps `ref_seconds` of speech outside it (the
    reference audio has to come from somewhere else) and the window itself contains at least
    `min_target_seconds` of speech (otherwise the sample would teach the model to generate silence).
    """
    window_seconds = num_frames / target_fps
    latest_start_frame = total_resampled - num_frames
    if latest_start_frame < 0:
        return None
    step_frames = max(1, int(round(step_seconds * target_fps)))
    starts = list(range(0, latest_start_frame + 1, step_frames))
    if starts[-1] != latest_start_frame:
        starts.append(latest_start_frame)

    best = None
    for start_frame in starts:
        start_s = start_frame / target_fps
        end_s = start_s + window_seconds
        if inside_seconds(segments_by_person, start_s, end_s) < min_target_seconds:
            continue
        score = min(outside_seconds(spans, start_s, end_s) for spans in segments_by_person.values())
        if best is None or score > best[0] + 1e-9:
            best = (score, start_frame)
    if best is None or best[0] < ref_seconds:
        return None
    return best[1]


def caption_utterances(annotation: dict, window_start_s: float, window_end_s: float,
                       valid_face_ids: set[str] | None = None,
                       dialogue_in_window: bool = True) -> list[tuple[str, str]]:
    """`(label, text)` of the dialogue lines, keeping the speaker id (and emotion) of every turn.

    With `dialogue_in_window` (the default) only the words whose midpoint falls inside the target
    window survive, so the `<S>` spans match the audio the model is asked to generate
    (`fix_prompt_with_asr` is disabled for multi-person rows, so nothing else realigns them).
    Without it every line of the clip is kept in full.
    """
    turns = []
    for utterance in sorted(annotation.get("utterances", []), key=lambda u: u["start"]):
        if valid_face_ids is not None and utterance["face_id"] not in valid_face_ids:
            continue
        if dialogue_in_window:
            words = utterance.get("words") or []
            if words:
                kept = [w["text"] for w in words
                        if window_start_s <= (w["start"] + w["end"]) / 2 < window_end_s]
                if not kept:
                    continue
                text = " ".join(kept)
            else:
                if not (utterance["start"] < window_end_s and utterance["end"] > window_start_s):
                    continue
                text = (utterance.get("text") or "").strip()
        else:
            text = (utterance.get("text") or "").strip()
        text = " ".join(text.split())
        if not text:
            continue
        tag = (utterance.get("tag") or "").strip()
        label = f"<{utterance['face_id']}> {tag}:" if tag else f"<{utterance['face_id']}>:"
        turns.append((label, text))
    return turns


def build_caption(annotation: dict, window_start_s: float, window_end_s: float,
                  valid_face_ids: set[str] | None = None, dialogue_in_window: bool = True) -> str:
    """Global caption, shot captions and the dialogue lines concatenated into one prompt.

    The scene description always covers the whole clip (like the pretrained captions, whose visual
    part describes more than the 5 s training window); only the `<S>` dialogue is limited to the
    window, so that it matches the audio the model has to generate.
    """
    parts = []
    global_caption = (annotation.get("global_caption") or "").strip()
    if global_caption:
        parts.append(global_caption)
    for shot in annotation.get("shots", []):
        caption = (shot.get("caption") or "").strip()
        if caption and caption != global_caption:
            parts.append(caption)

    for label, text in caption_utterances(annotation, window_start_s, window_end_s,
                                          valid_face_ids, dialogue_in_window):
        parts.append(f"{label} <S>{text}<E>")
    return " ".join(parts)


def _index_with_base(lines: list[str], base: Path, exts):
    index: dict[str, Path] = {}
    unresolved: list[str] = []
    for line in lines:
        path = Path(line)
        if not path.is_absolute():
            path = base / path
        resolved = resolve_video_path(path, exts)
        if resolved is None:
            unresolved.append(line)
            continue
        index[video_id_of(resolved, exts)] = resolved.resolve()
    return index, unresolved


def build_video_index(list_path: Path | None, list_base: Path, video_root: Path, exts=VIDEO_EXTS):
    """`video_id -> 绝对视频路径`（列表文件优先）、没找到的行，以及实际使用的基准目录。

    列表里的相对路径按 `--list-base` 解析；若一行都命中不了，会依次试「视频根目录」和
    「列表文件所在目录」，用命中最多的那个（并打印提示），避免基准设错导致整批样本被判 missing。
    视频可以带扩展名也可以不带（`<id>.mp4` 与 `<id>` 都接受）。
    """
    if list_path is None:
        return {}, [], list_base

    lines = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}, [], list_base

    candidates: list[Path] = []
    for base in (list_base, video_root, list_path.parent):
        if base not in candidates:
            candidates.append(base)

    best = None
    for base in candidates:
        index, unresolved = _index_with_base(lines, base, exts)
        if best is None or len(index) > len(best[1]):
            best = (base, index, unresolved)
        if len(unresolved) == 0:
            break
    base, index, unresolved = best
    if base != list_base and index:
        print(f"[提示] --list 的相对路径按 {list_base} 解析不到文件，已改按 {base} 解析"
              f"（命中 {len(index)}/{len(lines)} 行）；可用 --list-base 固定")
    return index, unresolved, base


def build_row(annotation_dir: Path, video_path: Path, feat_dir: Path, n_refs: int, num_frames: int,
              target_fps: float, ref_seconds: float, require_qa_pass: bool, allow_offscreen_speech: bool,
              min_target_seconds: float = 1.0, dialogue_in_window: bool = True):
    """Returns `(row, None)` or `(None, reason)` when the clip cannot be used."""
    probe_path = annotation_dir / "s0-preprocess" / "probe.json"
    audio_path = annotation_dir / "s0-preprocess" / "mix.wav"
    annotation_path = annotation_dir / "s11-compose" / "annotation.json"
    segments_path = annotation_dir / "s7-tse" / "segments.json"
    qa_path = annotation_dir / "s11-compose" / "qa" / "report.json"
    for path in (probe_path, audio_path, annotation_path, segments_path):
        if not path.is_file():
            return None, f"missing_file: {path.relative_to(annotation_dir)}"

    probe = load_json(probe_path)
    annotation = load_json(annotation_path)
    segments = load_json(segments_path)
    qa = load_json(qa_path) if qa_path.is_file() else {}

    if require_qa_pass and qa and not qa.get("passed", False):
        failed = [g["name"] for g in qa.get("gates", []) if not g.get("passed", True)]
        return None, f"qa_failed: {', '.join(failed) or 'unknown gate'}"
    if not allow_offscreen_speech and qa.get("notes", {}).get("offscreen_speakers", 0):
        return None, f"offscreen_speech: {qa['notes']['offscreen_speakers']} offscreen speaker(s)"

    segments_by_person = speaking_identities(segments)
    if len(segments_by_person) != n_refs:
        return None, f"identity_count: {len(segments_by_person)} speaking identit(ies) with audio, n_refs={n_refs}"

    total_resampled = resampled_frame_count(probe, target_fps)
    start_frame = choose_window(segments_by_person, total_resampled, num_frames, target_fps, ref_seconds,
                                min_target_seconds=min_target_seconds)
    if start_frame is None:
        return None, (f"no_window: no {num_frames}-frame window with >= {min_target_seconds:.1f}s of speech inside "
                      f"and {ref_seconds:.1f}s of reference audio outside for every person "
                      f"({total_resampled} frames available)")

    face_paths, feat_paths, spk_paths, spk_spans = [], [], [], []
    for face_id, spans in segments_by_person.items():
        face_path = annotation_dir / "s3-cluster" / "faces" / f"{face_id}.jpg"
        feat_path = feat_dir / f"{video_path.stem}_{face_id}.pt"
        if not face_path.is_file():
            return None, f"missing_file: no reference face {face_path.name}"
        if not feat_path.is_file():
            return None, (f"missing_file: no reference features {feat_path} - run "
                          f"`dataset/extract_ref_face_feats.py` first")
        audio_files = []
        for span in spans:
            segment_path = annotation_dir / span["audio"]
            if not segment_path.is_file():
                return None, f"missing_file: no extracted speech {span['audio']}"
            audio_files.append(str(segment_path.resolve()))
        face_paths.append(str(face_path.resolve()))
        feat_paths.append(str(feat_path.resolve()))
        spk_paths.append(",".join(audio_files))
        spk_spans.append([[float(s["start"]), float(s["end"])] for s in spans])

    window_start_s = start_frame / target_fps
    window_end_s = window_start_s + num_frames / target_fps
    valid_face_ids = set(segments_by_person)
    turns = caption_utterances(annotation, window_start_s, window_end_s, valid_face_ids, dialogue_in_window)
    if not turns:
        return None, "no_dialogue: no utterance of a referenced person inside the target window"
    caption = build_caption(annotation, window_start_s, window_end_s, valid_face_ids, dialogue_in_window)

    row = {
        "video_path": str(video_path.resolve()),
        "audio_path": str(audio_path.resolve()),
        "caption": caption,
        "num_frames": probe["timeline"]["frame_count"],
        "bbox": "0,0,1,1",                       # unused by the multi-person path
        "face_paths": ";".join(face_paths),
        "feat_paths": ";".join(feat_paths),
        "spk_audio_paths": ";".join(spk_paths),
        "spk_segments": json.dumps(spk_spans),
        "target_start_frame": start_frame,
    }
    return row, None


def main() -> None:
    parser = argparse.ArgumentParser(description="avannotate -> multi-person training meta CSV.")
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True,
                        help="directory holding the videos, `<video_id>.mp4` or extension-less `<video_id>`")
    parser.add_argument("--video-exts", default=",".join(VIDEO_EXTS),
                        help="extensions tried when a video path does not exist as written")
    parser.add_argument("--feat-dir", type=Path, required=True,
                        help="output of `dataset/extract_ref_face_feats.py`")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--list", type=Path, default=None,
                        help="optional data list (one video path per line) instead of `<video_root>/<id>.mp4`")
    parser.add_argument("--list-base", type=Path, default=Path.cwd(),
                        help="base directory for relative paths inside --list")
    parser.add_argument("--n-refs", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--ref-audio-seconds", type=float, default=2.0,
                        help="must match ref_audio_frames / target_fps at training time")
    parser.add_argument("--min-target-speech-seconds", type=float, default=1.0,
                        help="speech required inside the target window, otherwise the sample is dropped")
    parser.add_argument("--keep-all-dialogue", action="store_true",
                        help="keep every dialogue line of the clip; by default only the lines inside "
                             "the target window are written (the scene description is always the full one)")
    parser.add_argument("--no-require-qa-pass", action="store_true",
                        help="keep clips whose QA gates did not pass")
    parser.add_argument("--allow-offscreen-speech", action="store_true",
                        help="keep clips that contain speech without a visible face")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    annotation_dirs = sorted(p.parent.parent for p in args.annotation_root.glob("*/s11-compose/annotation.json"))
    if not annotation_dirs:
        raise RuntimeError(f"No */s11-compose/annotation.json under {args.annotation_root}")
    if args.limit is not None:
        annotation_dirs = annotation_dirs[: args.limit]

    video_exts = tuple(e.strip() for e in args.video_exts.split(",") if e.strip())
    video_index, unresolved, list_base = build_video_index(args.list, args.list_base, args.video_root, video_exts)

    print(f"标注目录 {len(annotation_dirs)} 个，来自 --list 的视频 {len(video_index)} 个"
          f"（--list 里有 {len(unresolved)} 行没找到文件，基准目录 {list_base}）")
    for line in unresolved[:3]:
        print(f"  [list 未解析] {line}")
    if len(unresolved) > 3:
        print(f"  … 还有 {len(unresolved) - 3} 行")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept, dropped = 0, []
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for annotation_dir in annotation_dirs:
            video_id = annotation_dir.name
            video_path = video_index.get(video_id)
            if video_path is None:
                # 没在 --list 里（或没给 --list）：在 video-root 下找 `<id>` / `<id>.<ext>`
                video_path = resolve_video_path(args.video_root / video_id, video_exts)
            if video_path is None:
                dropped.append((video_id, f"missing_video: {args.video_root / video_id}[.mp4]"))
                continue
            row, reason = build_row(
                annotation_dir=annotation_dir,
                video_path=video_path,
                feat_dir=args.feat_dir,
                n_refs=args.n_refs,
                num_frames=args.num_frames,
                target_fps=args.target_fps,
                ref_seconds=args.ref_audio_seconds,
                require_qa_pass=not args.no_require_qa_pass,
                allow_offscreen_speech=args.allow_offscreen_speech,
                min_target_seconds=args.min_target_speech_seconds,
                dialogue_in_window=not args.keep_all_dialogue,
            )
            if row is None:
                dropped.append((video_id, reason))
                continue
            writer.writerow(row)
            kept += 1

    print(f"Wrote {kept} row(s) to {args.output}, dropped {len(dropped)}")
    if dropped:
        # 同一类原因只打一行（几千条同质日志没有意义），每类给两个例子
        grouped: dict[str, list[tuple[str, str]]] = {}
        for video_id, reason in dropped:
            grouped.setdefault(reason.split(":", 1)[0], []).append((video_id, reason))
        for category, items in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
            print(f"  [dropped {len(items):>5}] {category}")
            for video_id, reason in items[:2]:
                print(f"            e.g. {video_id}: {reason}")


if __name__ == "__main__":
    main()
