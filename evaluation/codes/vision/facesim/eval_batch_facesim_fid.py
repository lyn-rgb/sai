import os
from pathlib import Path
import csv
from tqdm import tqdm
import numpy as np
from facesim_fid_evaluator import FaceSimFID_Evaluator


IMAGE_EXT = ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')
VIDEO_EXT = (".mp4", ".gif", ".MP4", ".GIF")


def find_video(p: Path):
    if p.is_file() and p.suffix in VIDEO_EXT:
        return p
    for ext in VIDEO_EXT:
        for f in p.glob(f"*{ext}"):
            return f
    return None


def build_pairs(video_root: Path, image_root: Path):
    # 支持四种扩展名
    img_map = {}
    for ext in IMAGE_EXT:
        for p in image_root.glob(f"*_frame.{ext}"):
            key = p.stem.split("_")[0]          # 00001_frame.png -> 00001
            img_map[key] = p
    pairs = []
    # 平铺
    for vid in video_root.iterdir():
        if vid.suffix not in VIDEO_EXT:
            continue
        for k in img_map:
            if vid.stem.startswith(k):
                pairs.append((vid, img_map[k], k))
                break
    # 子目录
    for sub in video_root.iterdir():
        if not sub.is_dir():
            continue
        k = next((k for k in img_map if k in sub.name), None)
        if k and find_video(sub):
            pairs.append((find_video(sub), img_map[k], k))
    pairs.sort(key=lambda x: int(x[2]))
    return pairs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", required=True, type=str,
                        default="/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/benchmark1/processed/frames")
    parser.add_argument("--video_root", required=True, type=str)
    parser.add_argument("--face_model_path", default="./ckpts", type=str)
    parser.add_argument("--inception_weights_path", default="./ckpts/face_encoder/inception_v3_google-0cc3c7bd.pth", type=str)
    
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--results_csv", default="./results_facesim.csv", type=str)
    parser.add_argument("--overwrite", action="store_true", help="delete old csv")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index for parallel evaluation")
    parser.add_argument("--num-shards", type=int, default=1, help="total number of shards for parallel evaluation")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    
    image_root = Path(args.image_root)
    video_root = Path(args.video_root)
    results_csv = Path(args.results_csv)
    overwrite = args.overwrite
    
    # 1. 创建模型
    facesim_fid_evaluator = FaceSimFID_Evaluator(
        device = args.device,
        face_model_path=args.face_model_path,
        inception_weights_path=args.inception_weights_path
    )
     
    # 2. 配对
    all_pairs = build_pairs(video_root, image_root)
    pairs = all_pairs[args.shard_index::args.num_shards]
    print(
        f"Found {len(all_pairs)} video-image pairs. "
        f"Shard {args.shard_index + 1}/{args.num_shards} will process {len(pairs)} pairs."
    )
    if not pairs:
        exit("No pairs found.")
        
    # 3. 覆盖模式
    if overwrite and results_csv.exists():
        results_csv.unlink()
        print("Old csv removed (--overwrite).")

    # 4. 批量计算
    scores = {"cur": [], "arc": [], "fid": []}
    with open(results_csv, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["prompt_id", "video_path", "image_path", "cur_score", "arc_score", "fid_score"])
        for vid_path, img_path, pid in tqdm(pairs, desc="FaceSim-FID"):
            ret = facesim_fid_evaluator.process(img_path, vid_path)
            if ret is None:
                tqdm.write(f"Skip {vid_path} (no face)")
                continue
            writer.writerow([pid, str(vid_path), str(img_path),
                             ret["cur_score"], ret["arc_score"], ret["fid_score"]])
            f.flush()
            scores["cur"].append(ret["cur_score"])
            scores["arc"].append(ret["arc_score"])
            scores["fid"].append(ret["fid_score"])

    # 5. 汇总
    if not scores["cur"]:
        exit("No valid scores.")
    # 清旧汇总行
    rows = []
    if results_csv.exists():
        with open(results_csv, newline='') as f:
            for r in csv.reader(f):
                if r and r[0] != "STATISTICS":
                    rows.append(r)
    with open(results_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        writer.writerow(["STATISTICS", "", "", "", "", ""])
        writer.writerow(["count", len(scores["cur"]), "", "", "", ""])
        writer.writerow(["cur_mean", f"{np.mean(scores['cur']):.4f}", "", "", "", ""])
        writer.writerow(["arc_mean", f"{np.mean(scores['arc']):.4f}", "", "", "", ""])
        writer.writerow(["fid_mean", f"{np.mean(scores['fid']):.4f}", "", "", "", ""])
        writer.writerow(["cur_std", f"{np.std(scores['cur']):.4f}", "", "", "", ""])
        writer.writerow(["arc_std", f"{np.std(scores['arc']):.4f}", "", "", "", ""])
        writer.writerow(["fid_std", f"{np.std(scores['fid']):.4f}", "", "", "", ""])

    print("\n【Overall】")
    print(f"  count : {len(scores['cur'])}")
    print(f"  cur   : {np.mean(scores['cur']):.4f} ± {np.std(scores['cur']):.4f}")
    print(f"  arc   : {np.mean(scores['arc']):.4f} ± {np.std(scores['arc']):.4f}")
    print(f"  fid   : {np.mean(scores['fid']):.4f} ± {np.std(scores['fid']):.4f}")
    print("Results →", results_csv)


if __name__ == "__main__":
    main()
