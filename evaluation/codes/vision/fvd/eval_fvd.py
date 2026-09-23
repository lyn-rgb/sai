import numpy as np
import torch
from tqdm import tqdm
import os
import cv2
import re
from glob import glob
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
    """视频数据集类"""
    def __init__(self, video_dir=None, video_paths=None, max_frames=30, frame_size=(224, 224)):  # 改为224x224
        if video_paths is not None:
            self.video_paths = [str(path) for path in video_paths]
        else:
            self.video_paths = sorted(glob(os.path.join(video_dir, '*.mp4')) + 
                                      glob(os.path.join(video_dir, '*.avi')) + 
                                      glob(os.path.join(video_dir, '*.mov')) + 
                                      glob(os.path.join(video_dir, '*.mkv')) +
                                      glob(os.path.join(video_dir, '*.webm')))
        
        if video_paths is None and len(self.video_paths) == 0:
            # 如果没有找到视频文件，尝试查找子目录中的视频
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        self.video_paths.append(os.path.join(root, file))
                if self.video_paths:  # 如果找到了文件，停止搜索
                    break
        source = video_dir if video_dir is not None else "provided video list"
        print(f"在 {source} 中找到 {len(self.video_paths)} 个视频")
        self.max_frames = max_frames
        self.frame_size = frame_size
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        return self.load_video(video_path)
    
    def load_video(self, video_path):
        """加载视频文件并预处理"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # 读取视频帧
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR转RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 调整大小
            frame = cv2.resize(frame, self.frame_size)
            # 归一化到[0,1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()

        if not frames:
            raise RuntimeError(f"无法读取视频帧: {video_path}")
        
        # 如果视频帧数不足，重复最后一帧
        while len(frames) < self.max_frames:
            frames.append(frames[-1])
        
        # 转换为tensor: [frames, height, width, channels]
        video = np.stack(frames, axis=0)
        # 转换为 [frames, channels, height, width]
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        
        return video


def infer_sample_id(video_path):
    stem = os.path.splitext(os.path.basename(str(video_path)))[0]
    if re.match(r"^\d{5}(?:_|$)", stem):
        return stem[:5]
    match = re.search(r"(?<!\d)(\d{5})(?!\d)", stem)
    return match.group(1) if match else None


def collect_video_paths(video_dir):
    video_paths = sorted(glob(os.path.join(video_dir, '*.mp4')) +
                         glob(os.path.join(video_dir, '*.avi')) +
                         glob(os.path.join(video_dir, '*.mov')) +
                         glob(os.path.join(video_dir, '*.mkv')) +
                         glob(os.path.join(video_dir, '*.webm')))
    if video_paths:
        return video_paths
    nested = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                nested.append(os.path.join(root, file))
    return sorted(nested)


def match_video_paths(real_dir, gen_dir):
    real_paths = collect_video_paths(real_dir)
    gen_paths = collect_video_paths(gen_dir)
    real_by_id = {}
    gen_by_id = {}
    for path in real_paths:
        sid = infer_sample_id(path)
        if sid and sid not in real_by_id:
            real_by_id[sid] = path
    for path in gen_paths:
        sid = infer_sample_id(path)
        if sid and sid not in gen_by_id:
            gen_by_id[sid] = path

    common_ids = sorted(set(real_by_id) & set(gen_by_id))
    if common_ids:
        return (
            [real_by_id[sid] for sid in common_ids],
            [gen_by_id[sid] for sid in common_ids],
            {"real_videos": len(real_paths), "gen_videos": len(gen_paths), "matched_videos": len(common_ids)},
        )

    if len(real_paths) == len(gen_paths):
        return (
            real_paths,
            gen_paths,
            {"real_videos": len(real_paths), "gen_videos": len(gen_paths), "matched_videos": len(real_paths)},
        )

    raise RuntimeError(f"没有找到可匹配的视频: real={len(real_paths)}, gen={len(gen_paths)}")

def trans(x):
    """视频格式转换函数"""
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)
    
    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 
    
    return x

def calculate_fvd_from_dirs(real_dir, gen_dir, device, method='styleganv', 
                            only_final=False, batch_size=4, max_frames=30, 
                            frame_size=(224, 224), i3d_weights_path=None):
    """
    计算两个视频目录之间的FVD
    
    Args:
        real_dir: 真实视频目录路径
        gen_dir: 生成视频目录路径
        device: 计算设备
        method: 'styleganv' 或 'videogpt'
        only_final: 是否只计算完整视频的FVD
        batch_size: 批处理大小
        max_frames: 每个视频的最大帧数
        frame_size: 帧尺寸 (height, width)
        i3d_weights_path: I3D权重路径，优先使用配置传入的路径
    """
    
    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
    
    print(f"计算FVD...")
    print(f"真实视频目录: {real_dir}")
    print(f"生成视频目录: {gen_dir}")
    
    real_paths, gen_paths, match_meta = match_video_paths(real_dir, gen_dir)
    print(
        "FVD匹配视频: "
        f"real={match_meta['real_videos']}, gen={match_meta['gen_videos']}, "
        f"matched={match_meta['matched_videos']}"
    )

    # 创建数据集和数据加载器
    real_dataset = VideoDataset(video_paths=real_paths, max_frames=max_frames, frame_size=frame_size)
    gen_dataset = VideoDataset(video_paths=gen_paths, max_frames=max_frames, frame_size=frame_size)
    
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False)
    
    all_feats1 = []
    all_feats2 = []
    
    if method == 'videogpt':
        # 对于VideoGPT，我们直接在CPU上运行所有计算
        print("使用CPU模式计算VideoGPT FVD...")
        
        # 加载模型到CPU，并确保不使用DataParallel
        i3d = load_i3d_pretrained(device='cpu', weights_path=i3d_weights_path)
        
        # 如果模型是DataParallel，获取其内部模块
        if hasattr(i3d, 'module'):
            i3d = i3d.module
        
        # 确保模型在CPU上
        i3d = i3d.to('cpu')
        i3d.eval()
        
        # 批量提取特征
        print("提取视频特征...")
        with torch.no_grad():
            for real_batch, gen_batch in tqdm(zip(real_loader, gen_loader),
                                              total=len(real_loader),
                                              desc="FVD videos",
                                              unit="video" if batch_size == 1 else "batch"):
                # 确保两个批次大小相同
                if real_batch.shape[0] != gen_batch.shape[0]:
                    min_batch = min(real_batch.shape[0], gen_batch.shape[0])
                    real_batch = real_batch[:min_batch]
                    gen_batch = gen_batch[:min_batch]
                
                # 数据已经在CPU上，不需要.to(device)
                # 预处理格式
                real_batch = trans(real_batch)
                gen_batch = trans(gen_batch)
                
                # 提取特征 - 直接在CPU上计算
                feats1 = get_fvd_feats(real_batch, i3d=i3d, device='cpu')
                feats2 = get_fvd_feats(gen_batch, i3d=i3d, device='cpu')
                
                all_feats1.append(feats1)
                all_feats2.append(feats2)
    else:
        # 对于styleganv，使用GPU
        i3d = load_i3d_pretrained(device=device, weights_path=i3d_weights_path)
        
        print("使用GPU模式计算StyleGANV FVD...")
        with torch.no_grad():
            for real_batch, gen_batch in tqdm(zip(real_loader, gen_loader),
                                              total=len(real_loader),
                                              desc="FVD videos",
                                              unit="video" if batch_size == 1 else "batch"):
                if real_batch.shape[0] != gen_batch.shape[0]:
                    min_batch = min(real_batch.shape[0], gen_batch.shape[0])
                    real_batch = real_batch[:min_batch]
                    gen_batch = gen_batch[:min_batch]
                
                real_batch = real_batch.to(device)
                gen_batch = gen_batch.to(device)
                real_batch = trans(real_batch)
                gen_batch = trans(gen_batch)
                
                feats1 = get_fvd_feats(real_batch, i3d=i3d, device=device)
                feats2 = get_fvd_feats(gen_batch, i3d=i3d, device=device)
                
                all_feats1.append(feats1.cpu())
                all_feats2.append(feats2.cpu())
    
    # 合并所有特征
    all_feats1 = torch.cat(all_feats1, dim=0)
    all_feats2 = torch.cat(all_feats2, dim=0)
    
    # 计算FVD
    print("计算FVD距离...")
    fvd_results = []
    
    if only_final:
        assert all_feats1.shape[1] >= 10, "视频帧数必须 >= 10"
        fvd_results.append(frechet_distance(all_feats1, all_feats2))
    else:
        # 计算不同时间步长的FVD
        for t in tqdm(range(10, all_feats1.shape[1] + 1)):
            feats1_t = all_feats1[:, :t]
            feats2_t = all_feats2[:, :t]
            fvd_results.append(frechet_distance(feats1_t, feats2_t))
    
    result = {
        "value": fvd_results,
        "num_videos": len(real_dataset),
        "real_videos": match_meta["real_videos"],
        "gen_videos": match_meta["gen_videos"],
        "matched_videos": match_meta["matched_videos"],
        "video_info": {
            "max_frames": max_frames,
            "frame_size": frame_size
        }
    }
    
    return result

def main():
    # 配置参数
    real_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/ConsisID/eval/new_metric/sample_70_real_video"
    # gen_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/HunyuanCustom/out_one_file"
    # gen_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/HuMo/out_one_file"
    gen_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/ID-Animator/ID-Animator/output_mp4"
    #gen_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/outputs/old_best_full/ip_image_True_ip_audio_False"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    
    # 计算FVD
    result = calculate_fvd_from_dirs(
        real_dir=real_dir,
        gen_dir=gen_dir,
        device=device,
        method='videogpt',  # 或 'styleganv'
        only_final=False,   # True只返回最终FVD，False返回各时间步的FVD
        batch_size=2,       # 减小batch_size以避免内存问题
        max_frames=30,      # 每个视频处理的最大帧数
        frame_size=(224, 224)  # I3D模型需要224x224输入
    )
    
    print("\n=== FVD计算结果 ===")
    print(f"视频数量: {result['num_videos']}")
    print(f"视频信息: {result['video_info']}")
    
    if len(result['value']) == 1:
        print(f"最终FVD: {result['value'][0]:.4f}")
    else:
        print(f"各时间步FVD (从10帧到{len(result['value'])+9}帧):")
        for i, fvd in enumerate(result['value']):
            print(f"  帧数 {i+10}: {fvd:.4f}")
        print(f"平均FVD: {np.mean(result['value']):.4f}")
        print(f"最终FVD: {result['value'][-1]:.4f}")

if __name__ == "__main__":
    main()
