from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


MEDIA_EXTS = {".mp4", ".png"}


def format_prompt_for_filename(text: str) -> str:
    no_tags = re.sub(r"<.*?>", "", text)
    safe = no_tags.replace(" ", "_").replace("/", "_")
    return safe[:50]


def load_caption_prompts(dreamid_data_dir: Path) -> dict[str, str]:
    caption_root = dreamid_data_dir / "oneip" / "captions"
    if not caption_root.is_dir():
        raise FileNotFoundError(f"DreamID caption directory does not exist: {caption_root}")
    prompts = {}
    for caption_path in sorted(caption_root.glob("*.json")):
        sample_id = caption_path.stem
        with caption_path.open(encoding="utf-8") as f:
            prompts[sample_id] = json.load(f)
    if not prompts:
        raise RuntimeError(f"No DreamID caption json files found in {caption_root}")
    return prompts


def build_prompt_index(prompts: dict[str, str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    prompt_to_id = {}
    collisions: dict[str, list[str]] = {}
    for sample_id, prompt in prompts.items():
        key = format_prompt_for_filename(" ".join(prompt.split()))
        if key in prompt_to_id:
            collisions.setdefault(key, [prompt_to_id[key]]).append(sample_id)
        else:
            prompt_to_id[key] = sample_id
    for key in collisions:
        prompt_to_id.pop(key, None)
    return prompt_to_id, collisions


def replace_leading_id(filename: str, sample_id: str) -> str:
    if re.match(r"^\d{5}(?=_)", filename):
        return re.sub(r"^\d{5}", sample_id, filename, count=1)
    return f"{sample_id}_{filename}"


def unique_path(path: Path) -> Path:
    if not path.exists() and not path.is_symlink():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(1, 1000):
        candidate = path.with_name(f"{stem}_dup{idx}{suffix}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(f"Cannot create a unique path for {path}")


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst = unique_path(dst)
    if copy_files:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def recover_outputs(
    source_dir: Path,
    output_dir: Path,
    dreamid_data_dir: Path,
    copy_files: bool,
) -> None:
    prompts = load_caption_prompts(dreamid_data_dir)
    prompt_to_id, collisions = build_prompt_index(prompts)
    media_files = sorted(path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in MEDIA_EXTS)
    matched = 0
    unmatched = []

    for media_path in media_files:
        matches = [(key, sample_id) for key, sample_id in prompt_to_id.items() if key and key in media_path.name]
        if len(matches) != 1:
            unmatched.append(str(media_path))
            continue
        _, sample_id = matches[0]
        rel_parent = media_path.parent.relative_to(source_dir)
        corrected_name = replace_leading_id(media_path.name, sample_id)
        link_or_copy(media_path, output_dir / rel_parent / corrected_name, copy_files=copy_files)
        matched += 1

    print(f"Source files: {len(media_files)}")
    print(f"Recovered files: {matched}")
    print(f"Unmatched files: {len(unmatched)}")
    print(f"Output directory: {output_dir}")

    report = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "dreamid_data_dir": str(dreamid_data_dir),
        "source_files": len(media_files),
        "recovered_files": matched,
        "unmatched_files": unmatched,
        "prompt_key_collisions": collisions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recovery_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if unmatched:
        print("Some files could not be matched. See recovery_report.json for details.")
    if collisions:
        print("Some prompt filename keys are not unique. See recovery_report.json for details.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover DreamID-Omni output filenames to testdata sample ids.")
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/comparison/dreamid_omni_testdata"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison/dreamid_omni_testdata_recovered"))
    parser.add_argument(
        "--dreamid-data-dir",
        type=Path,
        default=Path("comparsion/DreamID-Omni/generated_testdata/evaluation_testdata"),
    )
    parser.add_argument("--copy-files", action="store_true", help="Copy media instead of creating symlinks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recover_outputs(
        source_dir=args.source_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        dreamid_data_dir=args.dreamid_data_dir.resolve(),
        copy_files=args.copy_files,
    )


if __name__ == "__main__":
    main()
