from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SPEECH_RE = re.compile(r"<S>(.*?)<E>", re.DOTALL)


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


def parse_sample_ids(raw_sample_ids: str | None) -> set[str] | None:
    if not raw_sample_ids:
        return None
    sample_ids = {sample_id.strip() for sample_id in raw_sample_ids.split(",") if sample_id.strip()}
    return sample_ids or None


def split_full_prompt(full_prompt: str) -> tuple[str, str]:
    speech_match = SPEECH_RE.search(full_prompt)
    if speech_match is None:
        raise ValueError(f"Prompt does not contain <S>...<E>: {full_prompt[:120]}")

    speech_content = " ".join(speech_match.group(1).split())
    video_prompt = SPEECH_RE.sub("", full_prompt)
    video_prompt = re.sub(
        r"\b(?:He|She|They|The man|The woman)\s+"
        r"(?:says|shares|declares|exclaims|whispers|remarks|murmurs)\s+warmly,?\s*$",
        "",
        video_prompt.strip(),
        flags=re.IGNORECASE,
    )
    video_prompt = " ".join(video_prompt.split())
    return video_prompt, speech_content


def build_sora2mini_csv(
    testdata_dir: Path,
    output_csv: Path,
    prompt_dir_name: str,
    frame_dir_name: str,
    audio_dir_name: str,
    asr_dir_name: str,
    sample_ids: set[str] | None,
    limit: int | None,
    lang: str,
) -> None:
    prompts = load_prompts(testdata_dir, prompt_dir_name)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "data_id",
                "ref_image_path",
                "ref_audio_path",
                "ref_speech_content",
                "video_path",
                "audio_path",
                "speech_content",
                "prompt",
                "lang",
            ],
        )
        writer.writeheader()

        for sample_id in sorted(prompts):
            if sample_ids is not None and sample_id not in sample_ids:
                continue

            frame_path = resolve_frame_path(testdata_dir, frame_dir_name, sample_id)
            audio_path = testdata_dir / audio_dir_name / f"{sample_id}_audio.wav"
            asr_path = testdata_dir / asr_dir_name / f"{sample_id}_asr.txt"
            if frame_path is None or not audio_path.is_file():
                continue

            video_prompt, speech_content = split_full_prompt(prompts[sample_id])
            ref_speech_content = asr_path.read_text(encoding="utf-8").strip() if asr_path.is_file() else ""
            writer.writerow(
                {
                    "data_id": sample_id,
                    "ref_image_path": str(frame_path.resolve()),
                    "ref_audio_path": str(audio_path.resolve()),
                    "ref_speech_content": " ".join(ref_speech_content.split()),
                    "video_path": "",
                    "audio_path": "",
                    "speech_content": speech_content,
                    "prompt": video_prompt,
                    "lang": lang,
                }
            )
            count += 1
            if limit is not None and count >= limit:
                break

    print(f"Wrote {count} Sora2-mini RAVG samples to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sora2-mini RAVG CSV from evaluation/testdata.")
    parser.add_argument("--testdata-dir", type=Path, default=Path("evaluation/testdata"))
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--prompt-dir-name", default="full_video_prompt")
    parser.add_argument("--frame-dir-name", default="frames")
    parser.add_argument("--audio-dir-name", default="audio")
    parser.add_argument("--asr-dir-name", default="ASR")
    parser.add_argument("--sample-ids", default=None, help="Comma-separated sample ids, e.g. 00001,00002.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lang", default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_sora2mini_csv(
        testdata_dir=args.testdata_dir.resolve(),
        output_csv=args.output_csv.resolve(),
        prompt_dir_name=args.prompt_dir_name,
        frame_dir_name=args.frame_dir_name,
        audio_dir_name=args.audio_dir_name,
        asr_dir_name=args.asr_dir_name,
        sample_ids=parse_sample_ids(args.sample_ids),
        limit=args.limit,
        lang=args.lang,
    )


if __name__ == "__main__":
    main()
