import os
import os.path as osp
from pathlib import Path
import subprocess
import gc
import multiprocessing as mp
from multiprocessing import Manager
from tqdm import tqdm

import torch  # NOTE, import torch before decord to avoid bug occurs in decord
import decord
decord.bridge.set_bridge("torch")    # Read to torch.Tensor directly

import librosa


failed_lst = Manager().list()


def list_files_recursive(directory, sufix=".mp4"):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(sufix):
                yield os.path.join(root, file).replace(directory+"/", "")


def get_file_list(data_dir, sufix=".mp4"):
    files = list(list_files_recursive(data_dir, sufix))
    return files


def collect_paths(src_dir, dst_dir, gpu_id=0):
    files = get_file_list(src_dir)
    paths = []
    for file in tqdm(files, desc="collect paths"):
        src_path = osp.join(src_dir, file)
        dst_path = osp.join(dst_dir, file)
        if not osp.exists(osp.dirname(dst_path)):
            os.makedirs(osp.dirname(dst_path))
        paths.append({
            "src": src_path,
            "dst": dst_path,
            "gpu_id": gpu_id,
        })
    return paths
    
                                
def reformat_video(path_info):
    src_path = Path(path_info['src'])
    dst_path = Path(path_info['dst'])
    gpu_id = path_info['gpu_id']
    
    ar=16000
    fps=24
    if not dst_path.exists() or dst_path.stat().st_size < 1024:
        command = [
            'ffmpeg', '-y',
            '-i', str(src_path),
            '-c:v', 'h264_nvenc',
            '-c:a', 'aac',
            '-ar', str(ar),
            '-ac', '2',
            '-r', str(fps),
            '-gpu', str(gpu_id),
            '-strict', 'experimental',
            '-shortest', 
            str(dst_path)
        ]
        try: 
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            #print(f"[FFMPEG] Error, source video: {str(src_path)}")
            if dst_path.exists():
                os.remove(str(dst_path))
            failed_lst.append(str(src_path))
    
def multi_run(paths, num_workers=16, tag="batch_01"):
    print(f"Total have {len(paths)} videos need to be reformated.")
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(reformat_video, paths), total=len(paths), desc=f"[{tag}]"):
                pass
    else:
        for path_info in tqdm(paths, desc=f"[{tag}]"):
            reformat_video(path_info)
        
    with open(f"./reformat_failed_lst_{tag}.txt", "w") as f:
        for path in failed_lst:
            f.write(f"{path}\n")

    # 写入之后就清空failed_lst
    failed_lst[:] = []
    
    
if __name__ == "__main__":
    task = "reformat"
    import argparse
    
    parser = argparse.ArgumentParser("Reformatting videos")
    parser.add_argument("--start_batch", type=int, default=1)
    parser.add_argument("--end_batch", type=int, default=11)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    
    if task == "reformat":
        src_dir = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/sample_1M_batches_20_complete/videos"
        dst_dir = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/sample_1M_batches_20_complete/videos_nv264_24fps"
        cpu_count = mp.cpu_count()
        num_processes = min(cpu_count, 8)  # 保守的进程数
        
        for batch_id in range(args.start_batch, args.end_batch):
            batch_name = f"batch_{batch_id:02d}"
            src_batch_dir = osp.join(src_dir, batch_name)
            dst_batch_dir = osp.join(dst_dir, batch_name)
            paths = collect_paths(src_batch_dir, dst_batch_dir, gpu_id=args.gpu_id)
            
            print(f"[{batch_id - args.start_batch + 1}/{args.end_batch - args.start_batch}] [GPU:{args.gpu_id}] Processing Batch {batch_name} Processing {len(paths)} video files with {num_processes} processes, cpus {cpu_count}.")
        
            multi_run(paths, num_workers=num_processes, tag=batch_name)
            
            # 清理垃圾
            del paths
            gc.collect()
    else:
        src_path = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/debug/output.mp4"
        src_path = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/sample_1M_batches_20_complete/videos/batch_02/I2DgAYXUm8A_1920x1080_full_video_110.mp4"
        dst_path = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/debug/I2DgAYXUm8A_1920x1080_full_video_110_reformat.mp4"
        aud_path = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/sample_1000_guaranteed/audios/28tQYkd57eo_1920x1080_full_video_009.wav"
        aud_path = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/sample_1M_batches_20_complete/audios/batch_02/I2DgAYXUm8A_1920x1080_full_video_110.wav"
        
        # load audio
        wave_data, sample_rate = librosa.load(aud_path, sr=16000, mono=True)
        assert sample_rate == 16000
        print(f"audio duration: {len(wave_data)/sample_rate} secs.")
        
        vr = decord.VideoReader(src_path)
        print(f"source video len: {len(vr)}, fps: {vr.get_avg_fps()}, duration: {len(vr)/vr.get_avg_fps()} secs.")
        
        reformat_video({"src":src_path, 'dst':dst_path, 'gpu_id':0})
        if osp.exists(dst_path):
            vr = decord.VideoReader(dst_path)
            print(f"reformated video len: {len(vr)}, fps: {vr.get_avg_fps()}, duration: {len(vr)/vr.get_avg_fps()} secs.")