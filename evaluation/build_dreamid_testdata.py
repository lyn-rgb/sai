from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_prompts(testdata_dir: Path, prompt_dir_name: str) -> dict[str, str]:
    prompt_dir = testdata_dir / prompt_dir_name
    if not prompt_dir.is_dir():
        raise FileNotFoundError(f"{prompt_dir} does not exist or is not a directory.")

    prompts = {}
    for prompt_path in sorted(prompt_dir.glob("*_full_caption.txt")):
        sample_id = prompt_path.name.removesuffix("_full_caption.txt")
        prompts[sample_id] = prompt_path.read_text(encoding="utf-8").strip()
    if not prompts:
        raise RuntimeError(f"No *_full_caption.txt files found in {prompt_dir}.")
    return prompts


def resolve_frame_path(testdata_dir: Path, frame_dir_name: str, sample_id: str) -> Path | None:
    frame_dir = testdata_dir / frame_dir_name
    for ext in FRAME_EXTENSIONS:
        frame_path = frame_dir / f"{sample_id}_frame{ext}"
        if frame_path.is_file():
            return frame_path
    return None


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def build_caption(sample_id: str, prompt: str) -> str:
    single_line_prompt = " ".join(prompt.split())
    return (
        f"{single_line_prompt} "
        f"The reference frame and reference audio for sample {sample_id} identify the main speaker as <sub1>."
    )


def build_dreamid_tree(
    testdata_dir: Path,
    output_dir: Path,
    prompt_dir_name: str,
    frame_dir_name: str,
    audio_dir_name: str,
    copy_files: bool,
) -> None:
    prompts = load_prompts(testdata_dir, prompt_dir_name)
    oneip_dir = output_dir / "oneip"
    img_root = oneip_dir / "imgs"
    audio_root = oneip_dir / "audios"
    caption_root = oneip_dir / "captions"
    caption_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for sample_id in sorted(prompts):
        frame_path = resolve_frame_path(testdata_dir, frame_dir_name, sample_id)
        audio_path = testdata_dir / audio_dir_name / f"{sample_id}_audio.wav"
        if frame_path is None or not audio_path.is_file():
            continue

        link_or_copy(frame_path, img_root / sample_id / f"{sample_id}.png", copy_files=copy_files)
        link_or_copy(audio_path, audio_root / sample_id / f"{sample_id}.wav", copy_files=copy_files)
        with (caption_root / f"{sample_id}.json").open("w", encoding="utf-8") as f:
            json.dump(build_caption(sample_id, prompts[sample_id]), f, ensure_ascii=False)
        count += 1

    print(f"Wrote {count} DreamID oneip samples to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DreamID-Omni test_case layout from evaluation/testdata.")
    parser.add_argument("--testdata-dir", type=Path, default=Path("evaluation/testdata"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-dir-name", default="full_video_prompt")
    parser.add_argument("--frame-dir-name", default="frames")
    parser.add_argument("--audio-dir-name", default="audio")
    parser.add_argument("--copy-files", action="store_true", help="Copy media instead of creating symlinks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dreamid_tree(
        testdata_dir=args.testdata_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        prompt_dir_name=args.prompt_dir_name,
        frame_dir_name=args.frame_dir_name,
        audio_dir_name=args.audio_dir_name,
        copy_files=args.copy_files,
    )


if __name__ == "__main__":
    main()
