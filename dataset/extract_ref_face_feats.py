"""Extract per-person reference faces (with margin) and their ArcFace features.

    python dataset/extract_ref_face_feats.py \
        --annotation-root /abs/annotation_examples \
        --video-root      /abs/clips \
        --list            /abs/fids.txt \
        --output-dir      /abs/ref_feats \
        --face-embedder-ckpt ./ckpts/InsightFace

Why the faces are re-cropped instead of using `s3-cluster/faces/F00X.jpg`:

* those crops hug the face box exactly (halo-free), while the pretrained model, the training
  pipeline (`dataset/preprocess/preprocess.py`) and the inference engine all feed references that
  were cropped with a **2.2x margin** (the face covers ~45% of the frame). Feeding a tight crop
  would be a ~2.2x scale mismatch against the model's training distribution;
* the detector (SCRFD) also needs that surrounding context - on a tight crop it frequently finds
  no face at all, which is what made the previous version fail with "No face detected".

So for every identity the best detection of its track (`s2-tracks/tracks.jsonl` +
`s3-cluster/identities.json`) is cropped from the video with `--margin-scale` and resized to
`--ref-size`, exactly like the repo's cropper does. Outputs:

    <output_dir>/faces/<video_id>_<face_id>.jpg     the margin-cropped reference face
    <output_dir>/<video_id>_<face_id>.pt            {"face_emb": tensor(512)} (antelopev2)

`dataset/build_meta_from_avannotate.py` picks these up automatically (and falls back to the
tight crop for identities that are missing here).

The step is incremental: existing outputs are skipped unless `--overwrite`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insightface.app import FaceAnalysis

from dataset.text_video_audio_dataset import open_video_reader
from dataset.build_meta_from_avannotate import build_video_index, resolve_video_path, VIDEO_EXTS

FACE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# 检测器在「满屏都是脸」的紧裁剪上会失败，留白不足时按这些倍数补边重试
PAD_SCALES = (1.0, 2.0, 3.0)


def build_embedder(ckpt_dir: str, device: int) -> FaceAnalysis:
    cuda_provider_options = {
        "device_id": device,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    embedder = FaceAnalysis(name="antelopev2", root=ckpt_dir,
                            providers=[("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"])
    embedder.prepare(ctx_id=device, det_size=(640, 640))
    embedder.get(np.zeros((512, 512, 3), dtype=np.uint8))       # warm up
    return embedder


def largest_face_embedding(embedder: FaceAnalysis, image: np.ndarray):
    """Largest face's ArcFace embedding, retrying on a border-extended canvas if none is found."""
    for scale in PAD_SCALES:
        canvas = image
        if scale != 1.0:
            pad_y = int(round((scale - 1) / 2 * image.shape[0]))
            pad_x = int(round((scale - 1) / 2 * image.shape[1]))
            canvas = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        faces = embedder.get(canvas)
        if len(faces) == 0:
            continue
        face = sorted(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))[-1]
        return torch.from_numpy(face["embedding"]).to(torch.float32).reshape(-1), scale
    return None, None


def load_identity_tracks(annotation_dir: Path) -> dict[str, list[dict]]:
    """face_id -> its track detections, merged from `s2-tracks` + `s3-cluster/identities.json`."""
    tracks: dict[int, list[dict]] = {}
    tracks_path = annotation_dir / "s2-tracks" / "tracks.jsonl"
    if not tracks_path.is_file():
        return {}
    with open(tracks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            track = json.loads(line)
            tracks[track["track_id"]] = track.get("detections", [])

    identities_path = annotation_dir / "s3-cluster" / "identities.json"
    if not identities_path.is_file():
        return {}
    identities = json.loads(identities_path.read_text(encoding="utf-8"))

    by_face: dict[str, list[dict]] = {}
    for identity in identities.get("identities", []):
        detections = []
        for track_id in identity.get("track_ids", []):
            detections.extend(tracks.get(track_id, []))
        by_face[identity["face_id"]] = detections
    return by_face


def pick_detection(detections: list[dict]):
    """The frame to use as reference: high score, close to the track's median face size."""
    if not detections:
        return None
    heights = sorted(d["h"] for d in detections)
    median_h = heights[len(heights) // 2] or 1.0
    return max(detections, key=lambda d: d["score"] - 0.5 * abs(d["h"] - median_h) / median_h)


def crop_with_margin(frame: np.ndarray, detection: dict, margin_scale: float, ref_size: int) -> np.ndarray:
    """Square crop centred on the face, `margin_scale` times its size, resized to `ref_size`.

    Mirrors the repo's cropper (`scale=2.2` in `modules/face_cropper/cropper.py`), so the reference
    face has the same face-to-frame ratio the model was trained with.
    """
    size = max(detection["w"], detection["h"]) * margin_scale
    cx = detection["x"] + detection["w"] / 2
    cy = detection["y"] + detection["h"] / 2
    half = size / 2
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))

    height, width = frame.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - width), max(0, y1 - height)
    if pad_l or pad_t or pad_r or pad_b:
        frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad_l, y0 + pad_t, x1 + pad_l, y1 + pad_t

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        raise RuntimeError("empty crop")
    return cv2.resize(crop, (ref_size, ref_size), interpolation=cv2.INTER_LINEAR)


def read_frame(reader, frame_index: int) -> np.ndarray:
    """Frame as BGR, since `s2-tracks` frame indices refer to the source video.

    `decord`'s bridge is set to torch by `text_video_audio_dataset`, so the batch comes back as a
    tensor there and as an ndarray in a plain decord setup - handle both.
    """
    frame = reader.get_batch([int(frame_index)])[0]
    frame = frame.asnumpy() if hasattr(frame, "asnumpy") else frame.numpy()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Margin-cropped reference faces + ArcFace (antelopev2) features.")
    parser.add_argument("--annotation-root", type=Path, required=True,
                        help="directory holding one annotation directory per video")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="where `faces/<video_id>_<face_id>.jpg` and `<video_id>_<face_id>.pt` are written")
    parser.add_argument("--video-root", type=Path, default=None,
                        help="directory holding the videos (with or without extension)")
    parser.add_argument("--list", type=Path, default=None, help="optional data list (one video path per line)")
    parser.add_argument("--list-base", type=Path, default=Path.cwd(),
                        help="base directory for relative paths inside --list")
    parser.add_argument("--video-exts", default=",".join(VIDEO_EXTS))
    parser.add_argument("--margin-scale", type=float, default=2.2,
                        help="crop this many times the face size (the repo's cropper uses 2.2)")
    parser.add_argument("--ref-size", type=int, default=512, help="reference face image size")
    parser.add_argument("--face-embedder-ckpt", default="./ckpts/InsightFace")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    annotation_dirs = sorted(p.parent.parent for p in args.annotation_root.glob("*/s11-compose/annotation.json"))
    if not annotation_dirs:
        raise RuntimeError(f"No */s11-compose/annotation.json under {args.annotation_root}")
    if args.limit is not None:
        annotation_dirs = annotation_dirs[: args.limit]

    video_exts = tuple(e.strip() for e in args.video_exts.split(",") if e.strip())
    video_index, _, list_base = (
        build_video_index(args.list, args.list_base, args.video_root or args.annotation_root, video_exts)
        if args.list is not None else ({}, [], args.list_base)
    )
    print(f"标注目录 {len(annotation_dirs)} 个"
          + (f"，来自 --list 的视频 {len(video_index)} 个（基准 {list_base}）" if args.list is not None else ""))

    faces_dir = args.output_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    embedder = build_embedder(args.face_embedder_ckpt, args.device)

    written, skipped, failures = 0, 0, []
    padded, fallback_crops = 0, 0
    for annotation_dir in tqdm(annotation_dirs, desc="reference faces"):
        video_id = annotation_dir.name
        try:
            by_face = load_identity_tracks(annotation_dir)
        except Exception as e:                                  # noqa: BLE001 - report and continue
            failures.append(f"{video_id}: cannot read tracks ({e})")
            continue
        if not by_face:
            failures.append(f"{video_id}: no face tracks")
            continue

        video_path = video_index.get(video_id)
        if video_path is None and args.video_root is not None:
            video_path = resolve_video_path(args.video_root / video_id, video_exts)
        reader = None
        if video_path is not None:
            try:
                reader = open_video_reader(video_path)
            except Exception as e:                              # noqa: BLE001 - fall back to the tight crop
                failures.append(f"{video_id}: cannot open video ({e})")

        for face_id, detections in by_face.items():
            face_out = faces_dir / f"{video_id}_{face_id}.jpg"
            feat_out = args.output_dir / f"{video_id}_{face_id}.pt"
            if face_out.exists() and feat_out.exists() and not args.overwrite:
                skipped += 1
                continue

            # 优先用视频里的 2.2 倍留白裁剪；拿不到视频时退回标注里的紧裁剪（会记录一次 fallback）
            detection = pick_detection(detections)
            image = None
            if reader is not None and detection is not None:
                try:
                    image = crop_with_margin(read_frame(reader, detection["frame"]),
                                             detection, args.margin_scale, args.ref_size)
                except Exception as e:                          # noqa: BLE001
                    failures.append(f"{video_id}/{face_id}: crop failed ({e})")
            if image is None:
                tight = next((p for p in sorted((annotation_dir / "s3-cluster" / "faces").glob(f"{face_id}.*"))
                              if p.suffix.lower() in FACE_IMAGE_EXTS), None)
                if tight is None:
                    failures.append(f"{video_id}/{face_id}: no video frame and no s3-cluster/faces/{face_id}.jpg")
                    continue
                image = cv2.imread(str(tight))
                if image is None:
                    failures.append(f"{video_id}/{face_id}: cannot read {tight}")
                    continue
                fallback_crops += 1

            embedding, pad_scale = largest_face_embedding(embedder, image)
            if embedding is None:
                failures.append(f"{video_id}/{face_id}: no face detected (pad scales {PAD_SCALES})")
                continue
            if pad_scale != 1.0:
                padded += 1

            cv2.imwrite(str(face_out), image)
            torch.save({"face_emb": embedding}, feat_out)
            written += 1

    print(f"Wrote {written} reference faces to {args.output_dir} "
          f"(skipped {skipped}, failed {len(failures)}; {padded} 需补边才检出, {fallback_crops} 用了紧裁剪兜底)")
    for message in failures[:20]:
        print(f"  [failed] {message}")
    if len(failures) > 20:
        print(f"  … 还有 {len(failures) - 20} 条失败")


if __name__ == "__main__":
    main()
