"""Precompute ArcFace features for the per-person reference faces of an avannotate tree.

    python dataset/extract_ref_face_feats.py \
        --annotation-root /abs/path/annotation_examples \
        --output-dir      /abs/path/ref_feats \
        --face-embedder-ckpt ./ckpts/InsightFace

The training pipeline embeds the reference face with InsightFace **antelopev2**
(`dataset/preprocess/preprocess.py`, `ovi_fusion_engine.py`), so the features must come from the
same model: the annotation pipeline's own embeddings (`s1-faces/embeddings.npy`, buffalo_l) live in
a different space and cannot be reused.

Writes `<video_id>_<face_id>.pt` with `{"face_emb": tensor(512)}`, the format the dataset reads
through its `feat_paths` column.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insightface.app import FaceAnalysis

FACE_GLOB = "*/s3-cluster/faces/*"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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


def largest_face_embedding(embedder: FaceAnalysis, image_path: Path) -> torch.Tensor:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read {image_path}")
    faces = embedder.get(image)
    if len(faces) == 0:
        raise RuntimeError(f"No face detected in {image_path}")
    # same rule as the rest of the repo: the largest face wins
    face = sorted(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))[-1]
    return torch.from_numpy(face["embedding"]).to(torch.float32).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ArcFace (antelopev2) features for reference faces.")
    parser.add_argument("--annotation-root", type=Path, required=True,
                        help="directory holding one annotation directory per video")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="where the `<video_id>_<face_id>.pt` files are written")
    parser.add_argument("--face-embedder-ckpt", default="./ckpts/InsightFace")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    face_paths = sorted(p for p in args.annotation_root.glob(FACE_GLOB) if p.suffix.lower() in IMAGE_EXTS)
    if args.limit is not None:
        face_paths = face_paths[: args.limit]
    if not face_paths:
        raise RuntimeError(f"No reference faces found under {args.annotation_root} (expected {FACE_GLOB})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedder = build_embedder(args.face_embedder_ckpt, args.device)

    written, skipped, failed = 0, 0, []
    for face_path in tqdm(face_paths, desc="reference faces"):
        # <annotation_root>/<video_id>/s3-cluster/faces/<face_id>.jpg
        video_id = face_path.parents[2].name
        out_path = args.output_dir / f"{video_id}_{face_path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            embedding = largest_face_embedding(embedder, face_path)
        except Exception as e:                                    # noqa: BLE001 - report and continue
            failed.append(f"{face_path}: {e}")
            continue
        torch.save({"face_emb": embedding}, out_path)
        written += 1

    print(f"Wrote {written} features to {args.output_dir} (skipped {skipped}, failed {len(failed)})")
    for message in failed:
        print(f"  [failed] {message}")


if __name__ == "__main__":
    main()
