from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from frechet_audio_distance import FrechetAudioDistance


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def infer_sample_id(path: Path) -> str | None:
    stem = path.stem
    if re.match(r"^\d{5}(?:_|$)", stem):
        return stem[:5]
    match = re.search(r"(?<!\d)(\d{5})(?!\d)", stem)
    return match.group(1) if match else None


def collect_audio_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_matched_audio_dirs(background_dir: str, eval_dir: str, tmp_root: Path) -> tuple[str, str, dict[str, int]]:
    background_files = collect_audio_files(Path(background_dir))
    eval_files = collect_audio_files(Path(eval_dir))
    background_by_id = {}
    eval_by_id = {}
    for path in background_files:
        sid = infer_sample_id(path)
        if sid and sid not in background_by_id:
            background_by_id[sid] = path
    for path in eval_files:
        sid = infer_sample_id(path)
        if sid and sid not in eval_by_id:
            eval_by_id[sid] = path

    common_ids = sorted(set(background_by_id) & set(eval_by_id))
    if common_ids:
        matched_background = tmp_root / "background"
        matched_eval = tmp_root / "eval"
        for sid in common_ids:
            link_or_copy(background_by_id[sid], matched_background / f"{sid}_audio{background_by_id[sid].suffix.lower()}")
            link_or_copy(eval_by_id[sid], matched_eval / f"{sid}_audio{eval_by_id[sid].suffix.lower()}")
        return (
            str(matched_background),
            str(matched_eval),
            {
                "fad_background_files": len(background_files),
                "fad_eval_files": len(eval_files),
                "fad_count": len(common_ids),
            },
        )

    if len(background_files) == len(eval_files):
        return (
            background_dir,
            eval_dir,
            {
                "fad_background_files": len(background_files),
                "fad_eval_files": len(eval_files),
                "fad_count": len(background_files),
            },
        )

    raise RuntimeError(f"没有找到可匹配的音频: background={len(background_files)}, eval={len(eval_files)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Frechet Audio Distance for two audio directories.")
    parser.add_argument("--background_dir", required=True)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model_name", default="vggish")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--use_activation", action="store_true")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--hf_cache", default="")
    args = parser.parse_args()

    if args.hf_cache:
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_cache
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    with tempfile.TemporaryDirectory(prefix="fad_matched_") as tmp_dir:
        background_dir, eval_dir, match_meta = prepare_matched_audio_dirs(
            args.background_dir,
            args.eval_dir,
            Path(tmp_dir),
        )
        print("Computing FAD")
        print(f"background_dir: {background_dir}")
        print(f"eval_dir: {eval_dir}")
        print(
            "FAD匹配音频: "
            f"background={match_meta['fad_background_files']}, "
            f"eval={match_meta['fad_eval_files']}, "
            f"matched={match_meta['fad_count']}"
        )

        frechet = FrechetAudioDistance(
            model_name=args.model_name,
            sample_rate=args.sample_rate,
            use_pca=args.use_pca,
            use_activation=args.use_activation,
            verbose=args.verbose,
        )
        score = frechet.score(background_dir, eval_dir, dtype=args.dtype)
    output = {"fad": float(score), **match_meta}
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"FAD: {output['fad']}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
