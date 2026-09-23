""" 对数据进行预处理，按照时长、音视频长度对齐、人脸检测筛选满足要求的数据，并将人脸单独保存为视频，人脸ID、音色特征单独保存为pt文件
"""

import os, gc, time
import os.path as osp
from typing import Tuple, List, Union, Optional, Any

import torch

import numpy as np
import cv2
import pandas as pd
import decord
import librosa
from videoio import VideoWriter
import imageio

import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from tqdm import tqdm

import sys
sys.path.append("../../")

# face cropper
from modules.face_cropper.cropper import Cropper
from modules.face_cropper.crop_config import CropConfig
# face embedder
from insightface.app import FaceAnalysis
import warnings

warnings.filterwarnings("ignore")


face_reconizer_ckpt_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/ckpts/InsightFace"
DEBUG=False


def makedirs(file_path):
    file_dir = osp.dirname(file_path)
    if not osp.exists(file_dir):
        try:
            os.makedirs(file_dir, exist_ok=False)
        except Exception as e:
            print(f"文件夹已由其他进程创建, Error < {e} >")
        
        
def file_exists_and_readable(filepath):
    """检查文件是否存在、非空且可读"""
    # 检查存在性
    if not os.path.exists(filepath):
        #print(f"文件不存在: {filepath}")
        return False
    
    # 检查是否是文件（而非目录）
    if not os.path.isfile(filepath):
        #print(f"路径不是文件: {filepath}")
        return False
    
    # 检查文件大小（防止空文件）
    if os.path.getsize(filepath) == 0:
        #print(f"文件为空: {filepath}")
        return False
    
    # 检查读取权限
    if not os.access(filepath, os.R_OK):
        #print(f"文件不可读: {filepath}")
        return False
    
    return True


class VideoProcesser:
    def __init__(self, device_ids=[0, 1], num_workers_per_device=8, flag_rot=True):
        self.device_ids = device_ids
        self.num_workers_per_device = num_workers_per_device
        self.audio_sr = 16000
        self.fps = 24
        self.flag_rot = flag_rot
        
        self.failed_lst = mp.Manager().list()
    
    def read_video(self, video_path, bbox=None, audio_path=None):
        video_info = {
            "num_frames": 0,
            "fps": 0,
            "height": 0,
            "width": 0
        }
        try:
            video_reader = decord.VideoReader(video_path)
            fps = video_reader.get_avg_fps()
            num_frames = len(video_reader)
            indices = np.arange(num_frames)
            frames = video_reader.get_batch(indices).asnumpy()  # T,H,W,C
            height, width = frames.shape[1:3]
            if bbox is not None:
                # only got the cropped part
                left, top, right, down = bbox.split(',')
                left = float(left)
                top = float(top)
                right = float(right)
                down = float(down)
                
                left, top  = max(0, int(left * width)), max(0, int(top * height))
                right, down = min(int(right * width), width), min(int(down * height), height)
                
                frames = frames[:, top:down, left:right]
            
            video_info["num_frames"] = num_frames
            video_info["fps"] = fps
            video_info["height"] = height
            video_info["width"] = width
            
            # 过滤音视频长度偏差大的视频
            if audio_path is not None:
                audio, sr = librosa.load(audio_path, sr=self.audio_sr, mono=True)
                audio_len = len(audio)
                audio_unit = self.audio_sr / fps
                audio_num_frames = audio_len / audio_unit
                
                if abs(audio_num_frames - num_frames) > 7:
                    print(f"{video_path}, (video len, audio len)=({audio_num_frames}, {num_frames})")
                    self.failed_lst.append(f"{video_path}, (video len, audio len)=({audio_num_frames}, {num_frames})")
                    frames = []
                    video_info["num_frames"] = 0
                    video_info["fps"] = 0
                    video_info["height"] = 0
                    video_info["width"] = 0
        except Exception as e:
            print(f"{video_path}, error ( {e} )")
            self.failed_lst.append(f"{video_path}, error ( {e} )")
            frames = []
            video_info["num_frames"] = 0
            video_info["fps"] = 0
            video_info["height"] = 0
            video_info["width"] = 0
            
        return frames, video_info
    
    def crop_video(self, cropper: Cropper, frames: List[np.ndarray]):
        """ video, lst of frames, RGB
        """
        #print(f"frames type: {type(frames)}, shape: {len(frames)},{frames[0].shape}")
        try:
            cropped = cropper.crop_driving_video(frames, flag_rot=self.flag_rot)
            #cropped = cropper.crop_source_video(frames, cropper.crop_cfg)
            frame_crop_lst = cropped["frame_crop_lst"]
        except Exception as e:
            frame_crop_lst = []
        
        return frame_crop_lst
            
    def get_face_feat(self, embedder: FaceAnalysis, frame_crop_lst: List[np.ndarray]):
        feats = []
        valids = torch.ones(len(frame_crop_lst))
        for i, frame_crop in enumerate(frame_crop_lst):
            try:
                face_info = embedder.get(cv2.cvtColor(frame_crop.copy(), cv2.COLOR_RGB2BGR))
                face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
                face_emb = torch.from_numpy(face_info['embedding']).to(torch.float32)
            except Exception as e:
                #print(f"!!! Error <{e}> occurred during extracting features")
                face_emb = torch.zeros(512, dtype=torch.float32)
                valids[i] = 0
            feats.append(face_emb)
            
        return feats, valids
    
    def save_video(self, frames, save_path):
        height, width = frames[0].shape[:2]
        makedirs(save_path)
        with imageio.get_writer(save_path, fps=self.fps, codec="libx264") as writer:
        #with VideoWriter(save_path, fps=self.fps, resolution=(width, height)) as writer:
            for frame in frames:
                #print(f"frame shape: {frame.shape}")
                #writer.write(frame.astype(np.uint8))
                writer.append_data(frame.astype(np.uint8))
            
    def run(self, meta_names, src_meta_dir, data_root, dst_meta_dir, dst_face_dir, dst_feat_dir, gpu_id, process_id):
        tic = time.time()
        # 0. 构建模型
        cuda_provider_options = {
            'device_id': gpu_id,  # 指定 GPU 设备 ID，默认为 0
            'arena_extend_strategy': 'kNextPowerOfTwo', # 内存分配策略
            'cudnn_conv_algo_search': 'EXHAUSTIVE', # 卷积算法搜索
            'do_copy_in_default_stream': True,
        }
        face_cropper = Cropper(crop_cfg=CropConfig, device_id=gpu_id, cuda_provider_options=cuda_provider_options)
        face_embedder = FaceAnalysis(name='antelopev2', root=face_reconizer_ckpt_dir, providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider'])
        face_embedder.prepare(ctx_id=gpu_id, det_size=(640, 640))
        # warm up
        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        face_embedder.get(img_bgr)
        
        # 对每一个meta文件
        # 1. 读取有效的meta信息
        for i, meta_name in enumerate(meta_names):
            print(f"[{i+1}/{len(meta_names)}][GPU ID: {gpu_id}, PID: {process_id}] Processing {meta_name}")
            src_meta_path = os.path.join(src_meta_dir, meta_name)
            src_meta_data = pd.read_csv(src_meta_path)
            
            dst_meta_path = os.path.join(data_root, dst_meta_dir, meta_name)
            # 避免重复处理数据
            if file_exists_and_readable(dst_meta_path):
                print(f"[{i+1}/{len(meta_names)}][GPU ID: {gpu_id}, PID: {process_id}] {dst_meta_path} already exists, skip.")
                continue
            columns = src_meta_data.columns.to_list()
            #print(columns)
            columns.append("face_path")
            columns.append("feat_path")
            columns.append("num_frames")
            columns.append("height")
            columns.append("width")
            columns.append("fps")
            dst_meta_data = pd.DataFrame(columns=columns)
            
            for j, data in tqdm(src_meta_data.iterrows(), total=src_meta_data.shape[0], disable=(process_id > 0)):
                # 对每一条数据
                vid = data.vid
                video_path = os.path.join(data_root, data.video_path)
                audio_path = os.path.join(data_root, data.audio_path)
                bbox = data.bbox
                
                # 2. 加载视频，返回视频时长和尺寸，可能返回None
                frames, video_info = self.read_video(video_path, bbox, audio_path)
                #print(f"{video_path}, num frames: {len(frames)}, {video_info}")
                if len(frames) > 0:
                    dst_face_path = dst_face_dir + data.video_path.split("/", 1)[-1]   # /batchxx/xxx.mp4
                    dst_feat_path = dst_feat_dir + data.video_path.split("/", 1)[-1].replace(".mp4", ".pt")
                    
                    if file_exists_and_readable(osp.join(data_root, dst_face_path)) and file_exists_and_readable(osp.join(data_root, dst_feat_path)):
                        # 如果文件已经存在，则直接写入数据
                        data["face_path"] = dst_face_path
                        data["feat_path"] = dst_feat_path
                        data["num_frames"] = video_info["num_frames"]
                        data["height"] = video_info["height"]
                        data["width"] = video_info["width"]
                        data["fps"] = video_info["fps"]
                        dst_meta_data.loc[len(dst_meta_data)] = data
                    else:
                        # 3. 对视频crop人脸，过滤没有人脸的数据(threshold=0.5)，保存到dst_face_dir
                        frame_crop_lst = self.crop_video(face_cropper, frames)
                        if len(frame_crop_lst) > 0:
                            # 4. 提取crop后人脸的特征，并过滤掉出错的数据，保存人脸特征和每帧和帧间的相似度（可选），保存到dst_feat_dir
                            face_feats, valids = self.get_face_feat(face_embedder, frame_crop_lst)
                            face_feats = torch.stack(face_feats)  # N,512
                            
                            self.save_video(frame_crop_lst, osp.join(data_root, dst_face_path))
                            # save face feats
                            makedirs(osp.join(data_root, dst_feat_path))
                            torch.save({"face_embs": face_feats, "valids": valids}, osp.join(data_root, dst_feat_path))
                            
                            data["face_path"] = dst_face_path
                            data["feat_path"] = dst_feat_path
                            data["num_frames"] = video_info["num_frames"]
                            data["height"] = video_info["height"]
                            data["width"] = video_info["width"]
                            data["fps"] = video_info["fps"]
                                    
                            dst_meta_data.loc[len(dst_meta_data)] = data
                        else:
                            print(f"[{i+1}/{len(meta_names)}][GPU ID: {gpu_id}, PID: {process_id}] {video_path}, failed cropping faces")
                            self.failed_lst.append(f"{video_path}, failed cropping faces")
                else:
                    print(f"[{i+1}/{len(meta_names)}][GPU ID: {gpu_id}, PID: {process_id}] Skip video {video_path}, failed to load.")
                #if DEBUG and j > 9:
                #    break
                        
            # 5. 将过滤后的数据重新组织mete信息并保存到dst_meta_dir                
            dst_meta_data.to_csv(dst_meta_path, index=False)
            del src_meta_data
            del dst_meta_data
            gc.collect()
        # 删除缓存和模型
        del face_cropper, face_embedder
        torch.cuda.empty_cache()
        
        # 6. 结束进程
        print(f"[GPU ID: {gpu_id}, PID: {process_id}] Done. Total costs {time.time() - tic} secs.")
    
    def multi_processing(self, src_meta_dir, data_root, dst_meta_dir, dst_face_dir, dst_feat_dir, machine_id=0, num_machines=1):
        # 1. 获取所有meta文件列表
        meta_names = sorted([meta_name for meta_name in os.listdir(src_meta_dir) if meta_name.endswith(".csv")])
        # 2. 根据machine_id 和 num_machines分配要处理的数据
        meta_names = meta_names[machine_id::num_machines]
        print("="*15 + "元信息" + "="*15)
        print(f"=> 机器数量 {num_machines} ")
        print(f"=> 机器ID  {machine_id}")
        print(f"=> 文件数  {len(meta_names)}")
        print(f"=> 设备ID  {self.device_ids}")
        print(f"=> 总进程  {len(self.device_ids) * self.num_workers_per_device}")
        print("="*15 + "处理中" + "="*15)
        # 2. 分配数据
        total_num_processess = len(self.device_ids) * self.num_workers_per_device
        sub_meta_names = [[] for _ in range(total_num_processess)]
        for i, meta_name in enumerate(meta_names):
            sub_meta_names[i % total_num_processess].append(meta_name)
        
        # 3. 启动子进程
        processess = []
        for i, gpu_id in enumerate(self.device_ids):
            for j in range(self.num_workers_per_device):
                process_id = i * self.num_workers_per_device + j
                process = mp.Process(
                    target=self.run,
                    args=(
                        sub_meta_names[process_id],
                        src_meta_dir,
                        data_root,
                        dst_meta_dir,
                        dst_face_dir,
                        dst_feat_dir,
                        gpu_id,
                        process_id
                    ),
                )
                process.start()
                processess.append(process)
        
        for process in processess:
            process.join()
                
        with open("preprocess_failed_lst.txt", "w") as f:
            f.writelines(self.failed_lst)
            self.failed_lst[:] = []
            
        print(f"全部进程处理完毕.")


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(f"Preprocessing Videos")
    argparser.add_argument("--num_machines", type=int, default=1)
    argparser.add_argument("--machine_id", type=int, default=0)
    argparser.add_argument("--num_processes_per_machine", type=int, default=4)
    args = argparser.parse_args()
    
    data_root = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/sample_1M_batches_20_complete"
    src_meta_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/data/meta/sample_1M_batches_20_complete"
    dst_meta_dir = "processed/csv/"
    dst_face_dir = "processed/face/"
    dst_feat_dir = "processed/feat/"
    
    video_processer = VideoProcesser(device_ids=list(range(torch.cuda.device_count())), num_workers_per_device=args.num_processes_per_machine)
    
    video_processer.multi_processing(src_meta_dir, data_root, dst_meta_dir, dst_face_dir, dst_feat_dir, machine_id=args.machine_id, num_machines=args.num_machines)
    