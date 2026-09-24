"""Archive the inputs of an inference run next to its outputs.

    python evaluation/dump_inference_inputs.py \
        --prompt-csv <output_dir>/../testdata_multiperson/generated_prompts/multiperson_prompts.csv \
        --output-dir ./outputs/full_ovi_multiperson_5s \
        --run-config /tmp/multiperson_infer.XXXX.yaml

Writes into the result folder:

    <output_dir>/inputs/<sample_id>/caption.txt          推理用的提示词（含 <F00X>: <S>…<E>）
    <output_dir>/inputs/<sample_id>/ref_face_p0.jpg      第 0 个参考人的脸（复制，不是软链）
    <output_dir>/inputs/<sample_id>/ref_audio_p0.wav     第 0 个参考人的音频
    <output_dir>/inputs/<sample_id>/source_paths.txt     这些素材的原始绝对路径与槽位约定
    <output_dir>/inputs/manifest.csv                     sample_id ↔ 参考素材 ↔ 生成的 mp4
    <output_dir>/run_config.yaml                         本次推理的超参快照（可选）

生成结果的文件名里只有 prompt 片段，没有样本 id，所以 manifest 用同一个
`format_prompt_for_filename` 规则把 mp4 反查回样本（`utils/processing_utils` 里那份会 import
torch，没装时用本文末尾的等价实现兜底）。
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:                                        # 推理环境：直接用与 new_infer.py 同一份实现
    from utils.processing_utils import format_prompt_for_filename
except Exception:                           # noqa: BLE001 - 只装了标准库（自检/离线）时兜底
    def format_prompt_for_filename(text: str) -> str:
        """与 utils/processing_utils.format_prompt_for_filename 等价（那里 import torch）。"""
        no_tags = re.sub(r"<.*?>", "", text)
        return no_tags.replace(" ", "_").replace("/", "_")[:50]


def sample_id_of(image_paths: list[str]) -> str | None:
    """`<testdata>/frames/<id>_frame_p0.jpg` → `<id>`（多人 testdata 的命名约定）。"""
    if not image_paths:
        return None
    name = Path(image_paths[0]).name
    match = re.match(r"(.+)_frame_p\d+$", Path(name).stem)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description="把推理用的参考素材归档到结果目录")
    parser.add_argument("--prompt-csv", type=Path, required=True, help="--config-file 里 text_prompt 指向的 CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="new_infer.py 的 output_dir")
    parser.add_argument("--run-config", type=Path, default=None, help="本次推理的运行配置（会复制一份进去）")
    parser.add_argument("--inputs-dir-name", default="inputs")
    parser.add_argument("--no-copy", action="store_true",
                        help="只写 manifest 与 source_paths，不复制素材（省空间）")
    args = parser.parse_args()

    with open(args.prompt_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"❌ {args.prompt_csv} 里没有样本")
        return 1

    inputs_dir = args.output_dir / args.inputs_dir_name
    inputs_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index, row in enumerate(rows, start=1):
        caption = (row.get("text_prompt") or "").strip()
        image_paths = [p for p in (row.get("ip_image_paths") or row.get("ip_image_path") or "").split(";") if p]
        audio_paths = [p for p in (row.get("ip_audio_paths") or row.get("ip_audio_path") or "").split(";") if p]
        sample_id = sample_id_of(image_paths) or f"sample{index:05d}"
        sample_dir = inputs_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")

        face_names, audio_names = [], []
        for slot, image_path in enumerate(image_paths):
            source = Path(image_path)
            target = sample_dir / f"ref_face_p{slot}{source.suffix or '.jpg'}"
            if not args.no_copy:
                shutil.copyfile(source, target)
            face_names.append(target.name if not args.no_copy else str(source))
        for slot, audio_path in enumerate(audio_paths):
            source = Path(audio_path)
            target = sample_dir / f"ref_audio_p{slot}{source.suffix or '.wav'}"
            if not args.no_copy:
                shutil.copyfile(source, target)
            audio_names.append(target.name if not args.no_copy else str(source))

        (sample_dir / "source_paths.txt").write_text(
            f"sample_id: {sample_id}\n"
            f"参考槽位顺序 = 文件名 p0/p1/… 的顺序（与训练时的 slot 约定一致）\n\n"
            f"caption:\n{caption}\n\n"
            + "".join(f"参考人脸 p{slot}: {path}\n" for slot, path in enumerate(image_paths))
            + "".join(f"参考音频 p{slot}: {path}\n" for slot, path in enumerate(audio_paths)),
            encoding="utf-8")

        # 反查本次生成的 mp4：文件名里嵌的是 format_prompt_for_filename(caption)
        formatted = format_prompt_for_filename(caption)
        outputs = sorted(str(p.relative_to(args.output_dir))
                         for p in args.output_dir.rglob(f"*_{formatted}_*.mp4"))
        manifest_rows.append({
            "sample_id": sample_id,
            "outputs": ";".join(outputs),
            "caption_file": str((sample_dir / "caption.txt").relative_to(args.output_dir)),
            "face_files": ";".join(face_names),
            "audio_files": ";".join(audio_names),
            "source_faces": ";".join(image_paths),
            "source_audios": ";".join(audio_paths),
        })
        print(f"  {sample_id}: {len(image_paths)} 人脸 / {len(audio_paths)} 音频"
              + (f" → 命中 {len(outputs)} 个输出" if outputs else " → 暂无对应输出"))

    manifest_path = inputs_dir / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    if args.run_config is not None and Path(args.run_config).is_file():
        shutil.copyfile(args.run_config, args.output_dir / "run_config.yaml")
        print(f"  超参快照: {args.output_dir / 'run_config.yaml'}")

    print(f"\n归档完成：{len(manifest_rows)} 个样本 → {inputs_dir}")
    print(f"  manifest: {manifest_path}（sample_id ↔ 参考素材 ↔ 生成的 mp4）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
