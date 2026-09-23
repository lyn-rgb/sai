#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_clipscore.py  --overwrite 强制覆盖旧结果
"""
import os, cv2, csv, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import snapshot_download

IMAGE_EXT = {".mp4", ".gif", ".MP4", ".GIF"}

# ----------- 工具函数 -----------
def find_video(p: Path):
    if p.is_file() and p.suffix in IMAGE_EXT:
        return p
    for ext in IMAGE_EXT:
        for f in p.glob(f"*{ext}"):
            return f
    return None

def load_prompt(p: Path):
    return p.read_text(encoding="utf-8").strip()

@torch.no_grad()
def compute_clip_score(video_path, model, processor, prompt, device, num_frames=16):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release(); return None
    idxs = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret: frames.append(frame)
    cap.release()
    if not frames: return None
    inputs = processor(text=prompt, images=frames, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return model(**inputs).logits_per_image.mean().item()

def build_pairs(video_root: Path, prompt_root: Path):
    prompt_map = {p.stem.split("_")[0]: p for p in prompt_root.glob("*_caption.txt")}
    pairs = []
    # 平铺
    for vid in video_root.iterdir():
        if vid.suffix not in IMAGE_EXT:
            continue
        for k in prompt_map:
            if vid.stem.startswith(k):
                pairs.append((vid, prompt_map[k], k))
                break
    # 子目录
    for sub in video_root.iterdir():
        if not sub.is_dir():
            continue
        k = next((k for k in prompt_map if k in sub.name), None)
        if k and find_video(sub):
            pairs.append((find_video(sub), prompt_map[k], k))
    pairs.sort(key=lambda x: int(x[2]))
    return pairs

# ---------------- 主流程 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", required=True, type=Path)
    parser.add_argument("--prompt_root", required=True, type=Path)
    parser.add_argument("--model_path", default="../ckpts/data_process/clip-vit-base-patch32", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--results_csv", default="results.csv", type=Path)
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete old csv and start fresh")
    args = parser.parse_args()

    # 1. 模型
    if not args.model_path.exists():
        print("Model not found, downloading ...")
        snapshot_download(repo_id="openai/clip-vit-base-patch32", local_dir=args.model_path)
    model = CLIPModel.from_pretrained(args.model_path).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_path)

    # 2. 配对
    pairs = build_pairs(args.video_root, args.prompt_root)
    print(f"Found {len(pairs)} video-prompt pairs.")
    if not pairs:
        exit("No pairs found.")

    # 3. 覆盖模式：直接删除旧文件
    if args.overwrite and args.results_csv.exists():
        args.results_csv.unlink()
        print("Old csv removed (--overwrite).")

    # 4. 写详情 + 收集分数
    scores = []
    with open(args.results_csv, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["prompt_id", "video_path", "prompt_path", "prompt_text", "clip_score"])
        for vid_path, prompt_path, pid in tqdm(pairs, desc="CLIP"):
            score = compute_clip_score(vid_path, model, processor, load_prompt(prompt_path),
                                       args.device, args.num_frames)
            if score is None:
                tqdm.write(f"Skip {vid_path}")
                continue
            writer.writerow([pid, str(vid_path), str(prompt_path), load_prompt(prompt_path), score])
            scores.append(score)

    # 5. 汇总（先清旧汇总行，再追加）
    if not scores:
        exit("No valid scores.")
    mean_, std_, count = np.mean(scores), np.std(scores), len(scores)
    rows = []
    if args.results_csv.exists():
        with open(args.results_csv, newline='') as f:
            for r in csv.reader(f):
                if r and r[0] != "STATISTICS":
                    rows.append(r)
    with open(args.results_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        writer.writerow(["STATISTICS", "", "", "", ""])
        writer.writerow(["count", count, "", "", ""])
        writer.writerow(["mean", f"{mean_:.4f}", "", "", ""])
        writer.writerow(["std", f"{std_:.4f}", "", "", ""])

    print("\n【Overall】")
    print(f"  count : {count}")
    print(f"  mean  : {mean_:.4f}")
    print(f"  std   : {std_:.4f}")
    print("Results →", args.results_csv)

if __name__ == "__main__":
    main()