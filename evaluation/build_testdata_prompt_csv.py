from __future__ import annotations

import argparse
import csv
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


def resolve_frame_path(testdata_dir: Path, frame_dir_name: str, sample_id: str, person_idx: int | None = None) -> Path | None:
    frame_dir = testdata_dir / frame_dir_name
    stem = f"{sample_id}_frame" if person_idx is None else f"{sample_id}_frame_p{person_idx}"
    for ext in FRAME_EXTENSIONS:
        frame_path = frame_dir / f"{stem}{ext}"
        if frame_path.is_file():
            return frame_path
    return None


def resolve_audio_path(testdata_dir: Path, audio_dir_name: str, sample_id: str, person_idx: int | None = None) -> Path | None:
    stem = f"{sample_id}_audio" if person_idx is None else f"{sample_id}_audio_p{person_idx}"
    audio_path = testdata_dir / audio_dir_name / f"{stem}.wav"
    return audio_path if audio_path.is_file() else None


def resolve_person_refs(
    testdata_dir: Path,
    sample_id: str,
    frame_dir_name: str,
    audio_dir_name: str,
    max_persons: int = 8,
) -> list[tuple[Path, Path]]:
    """One (reference frame, reference audio) pair per person, person 0 first.

    Person files are named `{sample_id}_frame_p{i}.{ext}` / `{sample_id}_audio_p{i}.wav`; the
    order of the persons is the reference slot order the model was trained with.
    """
    refs = []
    for person_idx in range(max_persons):
        frame_path = resolve_frame_path(testdata_dir, frame_dir_name, sample_id, person_idx)
        audio_path = resolve_audio_path(testdata_dir, audio_dir_name, sample_id, person_idx)
        if frame_path is None or audio_path is None:
            break
        refs.append((frame_path, audio_path))
    return refs


def existing_sample_ids(
    testdata_dir: Path,
    prompts: dict[str, str],
    mode: str,
    frame_dir_name: str,
    audio_dir_name: str,
    min_persons: int = 2,
) -> list[str]:
    sample_ids = []
    for sample_id in sorted(prompts):
        if mode == "text_only":
            sample_ids.append(sample_id)
            continue

        if mode == "id2v":
            frame_path = resolve_frame_path(testdata_dir, frame_dir_name, sample_id)
            audio_path = resolve_audio_path(testdata_dir, audio_dir_name, sample_id)
            if frame_path is not None and audio_path is not None:
                sample_ids.append(sample_id)
            continue

        # multiperson
        refs = resolve_person_refs(testdata_dir, sample_id, frame_dir_name, audio_dir_name)
        if len(refs) >= min_persons:
            sample_ids.append(sample_id)
    return sample_ids


def write_csv(
    testdata_dir: Path,
    output_path: Path,
    prompts: dict[str, str],
    mode: str,
    frame_dir_name: str,
    audio_dir_name: str,
) -> None:
    sample_ids = existing_sample_ids(
        testdata_dir,
        prompts,
        mode=mode,
        frame_dir_name=frame_dir_name,
        audio_dir_name=audio_dir_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["text_prompt"]
    if mode == "id2v":
        fields.extend(["ip_image_path", "ip_audio_path"])
    elif mode == "multiperson":
        # semicolon separated, one entry per reference person (slot order = person order)
        fields.extend(["ip_image_paths", "ip_audio_paths"])

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample_id in sample_ids:
            row = {"text_prompt": prompts[sample_id]}
            if mode == "id2v":
                frame_path = resolve_frame_path(testdata_dir, frame_dir_name, sample_id)
                audio_path = resolve_audio_path(testdata_dir, audio_dir_name, sample_id)
                if frame_path is None:
                    raise FileNotFoundError(f"No frame image found for sample {sample_id} in {testdata_dir / frame_dir_name}")
                row["ip_image_path"] = str(frame_path.resolve())
                row["ip_audio_path"] = str(audio_path.resolve())
            elif mode == "multiperson":
                refs = resolve_person_refs(testdata_dir, sample_id, frame_dir_name, audio_dir_name)
                row["ip_image_paths"] = ";".join(str(frame.resolve()) for frame, _ in refs)
                row["ip_audio_paths"] = ";".join(str(audio.resolve()) for _, audio in refs)
            writer.writerow(row)

    print(f"Wrote {len(sample_ids)} samples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build new_infer.py-compatible prompt CSVs from evaluation/testdata.")
    parser.add_argument("--testdata-dir", type=Path, default=Path("evaluation/testdata"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["text_only", "id2v", "multiperson"], required=True)
    parser.add_argument("--prompt-dir-name", default="full_video_prompt")
    parser.add_argument("--frame-dir-name", default="frames")
    parser.add_argument("--audio-dir-name", default="audio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    testdata_dir = args.testdata_dir.resolve()
    prompts = load_prompts(testdata_dir, args.prompt_dir_name)
    write_csv(
        testdata_dir,
        args.output,
        prompts,
        mode=args.mode,
        frame_dir_name=args.frame_dir_name,
        audio_dir_name=args.audio_dir_name,
    )


if __name__ == "__main__":
    main()
