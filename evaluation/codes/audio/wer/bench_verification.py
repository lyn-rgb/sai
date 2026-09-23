# bench_verification.py
import argparse
import soundfile as sf
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL
import os
import glob
from pathlib import Path
import subprocess
import numpy as np
import re
from tqdm import tqdm

MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat',
              "wavlm_base_plus", "wavlm_large"]

# ---------- 以下全部代码与之前相同，仅 batch_verification 函数签名与一行改动 ----------
def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=None)
    elif model_name == 'wavlm_large':
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
    elif model_name == 'hubert_large':
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=None)
    elif model_name == 'wav2vec2_xlsr':
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=None)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model

def extract_audio_from_video(video_path, output_audio_path):
    cmd = ['ffmpeg', '-i', video_path,
           '-vn', '-acodec', 'pcm_s16le',
           '-ar', '16000', '-ac', '1', '-y', output_audio_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return True

def detect_audio2_structure(audio2_path):
    audio2_path = Path(audio2_path)
    if not audio2_path.exists():
        raise ValueError(f"Audio2 path does not exist: {audio2_path}")
    items = list(audio2_path.iterdir())
    if not items:
        raise ValueError(f"No files or directories found in {audio2_path}")
    subdirs = [item for item in items if item.is_dir()]
    files   = [item for item in items if item.is_file()]
    video_ext = {'.mp4', '.avi', '.mov', '.mkv'}
    audio_ext = {'.wav', '.mp3', '.flac', '.m4a'}

    if subdirs:
        first_sub = subdirs[0]
        sub_files = list(first_sub.iterdir())
        has_v = any(f.suffix.lower() in video_ext for f in sub_files)
        has_a = any(f.suffix.lower() in audio_ext for f in sub_files)
        if has_v: return "subdirs_with_videos"
        if has_a: return "subdirs_with_audios"
        raise ValueError(f"Unsupported file types in subdirectory {first_sub}")
    else:
        has_v = any(f.suffix.lower() in video_ext for f in files)
        has_a = any(f.suffix.lower() in audio_ext for f in files)
        if has_v and has_a: return "mixed_files"
        if has_v: return "direct_videos"
        if has_a: return "direct_audios"
        raise ValueError(f"Unsupported file types in directory {audio2_path}")

def prepare_audio2_files(audio2_path):
    structure = detect_audio2_structure(audio2_path)
    audio2_path = Path(audio2_path)
    video_ext = {'.mp4', '.avi', '.mov', '.mkv'}
    audio_ext = {'.wav', '.mp3', '.flac', '.m4a'}

    if structure == "subdirs_with_videos":
        extract_dir = audio2_path.parent / 'extract_audio2'
        extract_dir.mkdir(exist_ok=True)
        audio_files = []
        for sub in sorted([d for d in audio2_path.iterdir() if d.is_dir()]):
            sid = sub.name.split('_')[0] if '_' in sub.name else sub.name
            v_files = [f for f in sub.iterdir() if f.suffix.lower() in video_ext]
            if not v_files: continue
            v = sorted(v_files)[0]
            out = extract_dir / f"{sid}_audio.wav"
            if not out.exists():
                extract_audio_from_video(str(v), str(out))
            audio_files.append(out)
        return extract_dir, audio_files

    if structure == "subdirs_with_audios":
        audio_files = []
        for sub in sorted([d for d in audio2_path.iterdir() if d.is_dir()]):
            a_files = [f for f in sub.iterdir() if f.suffix.lower() in audio_ext]
            if a_files: audio_files.append(sorted(a_files)[0])
        return audio2_path, audio_files

    if structure == "direct_videos":
        extract_dir = audio2_path.parent / 'extract_audio2'
        extract_dir.mkdir(exist_ok=True)
        audio_files = []
        for v in sorted([f for f in audio2_path.iterdir() if f.suffix.lower() in video_ext]):
            stem = v.stem
            m = re.search(r'(\d{5})', stem)
            vid = m.group(1) if m else stem
            out = extract_dir / f"{vid}_audio.wav"
            if not out.exists():
                extract_audio_from_video(str(v), str(out))
            audio_files.append(out)
        return extract_dir, audio_files

    if structure == "direct_audios":
        return audio2_path, sorted([f for f in audio2_path.iterdir() if f.suffix.lower() in audio_ext])

    # mixed_files
    extract_dir = audio2_path.parent / 'extract_audio2'
    extract_dir.mkdir(exist_ok=True)
    audio_files = []
    for f in sorted(audio2_path.iterdir()):
        if f.suffix.lower() in audio_ext:
            audio_files.append(f)
        elif f.suffix.lower() in video_ext:
            stem = f.stem
            m = re.search(r'(\d{5})', stem)
            fid = m.group(1) if m else stem
            out = extract_dir / f"{fid}_audio.wav"
            if not out.exists():
                extract_audio_from_video(str(f), str(out))
            audio_files.append(out)
    return extract_dir, audio_files

def find_matching_pairs(audio1_dir, audio2_files):
    audio1_dir = Path(audio1_dir)
    audio1_files = sorted(audio1_dir.glob('*.wav'))
    pairs = []
    for a1 in audio1_files:
        a1_id = a1.stem.split('_')[0]
        for a2 in audio2_files:
            if a1_id in a2.stem:
                pairs.append((a1, a2, a1_id))
                break
        else:
            print(f"Warning: No matching audio2 file found for {a1.name}")
    return pairs

def process_audio_pair(model, wav1_path, wav2_path, use_gpu=True):
    w1, sr1 = sf.read(wav1_path)
    w2, sr2 = sf.read(wav2_path)
    w1 = torch.from_numpy(w1).unsqueeze(0).float()
    w2 = torch.from_numpy(w2).unsqueeze(0).float()
    w1 = Resample(sr1, 16000)(w1)
    w2 = Resample(sr2, 16000)(w2)
    if use_gpu:
        w1, w2 = w1.cuda(), w2.cuda()
    model.eval()
    with torch.no_grad():
        e1, e2 = model(w1), model(w2)
    return F.cosine_similarity(e1, e2)[0].item()

def verification(model_name, wav1, wav2, use_gpu=True, checkpoint=None):
    assert model_name in MODEL_LIST
    model = init_model(model_name, checkpoint)
    w1, sr1 = sf.read(wav1)
    w2, sr2 = sf.read(wav2)
    w1 = torch.from_numpy(w1).unsqueeze(0).float()
    w2 = torch.from_numpy(w2).unsqueeze(0).float()
    w1 = Resample(sr1, 16000)(w1)
    w2 = Resample(sr2, 16000)(w2)
    if use_gpu:
        model, w1, w2 = model.cuda(), w1.cuda(), w2.cuda()
    model.eval()
    with torch.no_grad():
        e1, e2 = model(w1), model(w2)
    sim = F.cosine_similarity(e1, e2)
    print(f"The similarity score between two audios is {sim[0].item():.4f} (-1.0, 1.0).")

# ------------------------- 唯一需要改动的函数 -------------------------
def batch_verification(model_name, audio1_dir, audio2_dir,
                       output_file='./audio_sim_cosy1.txt',   # ← 新增参数
                       use_gpu=True, checkpoint=None):
    assert model_name in MODEL_LIST
    model = init_model(model_name, checkpoint)
    if use_gpu:
        model = model.cuda()
    model.eval()

    audio2_path, audio2_files = prepare_audio2_files(audio2_dir)
    print(f"Found {len(audio2_files)} audio files in audio2 directory")
    pairs = find_matching_pairs(audio1_dir, audio2_files)
    print(f"Found {len(pairs)} matching audio pairs")
    if not pairs:
        print("No matching audio pairs found!")
        return

    results = []
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ID\tAudio1\tAudio2\tSimilarity\n")
        f.write("-" * 50 + "\n")
        for a1, a2, pid in tqdm(pairs, desc="AudioSim", unit="pair"):
            try:
                score = process_audio_pair(model, a1, a2, use_gpu)
                results.append(score)
                line = f"{pid}\t{a1.name}\t{a2.name}\t{score:.4f}"
                tqdm.write(line)
                f.write(line + "\n")
                f.flush()
            except Exception as e:
                tqdm.write(f"Error processing pair {pid}: {e}")
                results.append(0.0)

    if results:
        avg, std, mx, mn = np.mean(results), np.std(results), np.max(results), np.min(results)
        print("\n" + "=" * 50)
        print("Batch Verification Results:")
        print(f"Total pairs: {len(results)}")
        print(f"Average: {avg:.4f}  Std: {std:.4f}  Max: {mx:.4f}  Min: {mn:.4f}")
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"Total pairs: {len(results)}\n")
            f.write(f"Average similarity: {avg:.4f}\n")
            f.write(f"Std similarity: {std:.4f}\n")
            f.write(f"Max similarity: {mx:.4f}\n")
            f.write(f"Min similarity: {mn:.4f}\n")
    print(f"\nResults saved to: {output_file}")

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def main():
    parser = argparse.ArgumentParser(description="Speaker verification utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verification")
    verify_parser.add_argument("--model_name", required=True, choices=MODEL_LIST)
    verify_parser.add_argument("--wav1", required=True)
    verify_parser.add_argument("--wav2", required=True)
    verify_parser.add_argument("--use_gpu", type=str2bool, default=True)
    verify_parser.add_argument("--checkpoint", default=None)

    batch_parser = subparsers.add_parser("batch_verification")
    batch_parser.add_argument("--model_name", required=True, choices=MODEL_LIST)
    batch_parser.add_argument("--audio1_dir", required=True)
    batch_parser.add_argument("--audio2_dir", required=True)
    batch_parser.add_argument("--output_file", default="./audio_sim_cosy1.txt")
    batch_parser.add_argument("--use_gpu", type=str2bool, default=True)
    batch_parser.add_argument("--checkpoint", default=None)

    args = parser.parse_args()
    if args.command == "verification":
        verification(args.model_name, args.wav1, args.wav2, args.use_gpu, args.checkpoint)
    elif args.command == "batch_verification":
        batch_verification(
            args.model_name,
            args.audio1_dir,
            args.audio2_dir,
            output_file=args.output_file,
            use_gpu=args.use_gpu,
            checkpoint=args.checkpoint,
        )


if __name__ == "__main__":
    main()





####在代码里写入输出结果
'''
import soundfile as sf
import torch
import fire
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL
import os
import glob
from pathlib import Path
import subprocess
import numpy as np

MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]

def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model

def extract_audio_from_video(video_path, output_audio_path):
    """从视频文件中提取音频"""
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', output_audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False

def detect_audio2_structure(audio2_path):
    """检测audio2目录的结构类型"""
    audio2_path = Path(audio2_path)
    
    if not audio2_path.exists():
        raise ValueError(f"Audio2 path does not exist: {audio2_path}")
    
    # 获取所有文件和子目录
    items = list(audio2_path.iterdir())
    
    if not items:
        raise ValueError(f"No files or directories found in {audio2_path}")
    
    # 检查是否包含子目录
    subdirs = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    
    if subdirs:
        # 情况1或2：包含子目录
        print(f"Found {len(subdirs)} subdirectories")
        
        # 检查第一个子目录中的文件类型
        first_subdir = subdirs[0]
        subdir_files = list(first_subdir.iterdir())
        
        if not subdir_files:
            raise ValueError(f"No files found in subdirectory {first_subdir}")
        
        # 检查子目录中是否包含视频文件
        has_video_in_subdir = any(f.suffix.lower() in video_extensions for f in subdir_files)
        # 检查子目录中是否包含音频文件
        has_audio_in_subdir = any(f.suffix.lower() in audio_extensions for f in subdir_files)
        
        if has_video_in_subdir:
            return "subdirs_with_videos"
        elif has_audio_in_subdir:
            return "subdirs_with_audios"
        else:
            raise ValueError(f"Unsupported file types in subdirectory {first_subdir}")
    
    else:
        # 情况3或4：直接包含文件
        print(f"Found {len(files)} files directly in directory")
        
        has_video = any(f.suffix.lower() in video_extensions for f in files)
        has_audio = any(f.suffix.lower() in audio_extensions for f in files)
        
        if has_video and has_audio:
            print("Warning: Mixed audio and video files found")
            return "mixed_files"
        elif has_video:
            return "direct_videos"
        elif has_audio:
            return "direct_audios"
        else:
            raise ValueError(f"Unsupported file types in directory {audio2_path}")

def prepare_audio2_files(audio2_path):
    """根据目录结构准备audio2音频文件"""
    structure_type = detect_audio2_structure(audio2_path)
    audio2_path = Path(audio2_path)
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    
    if structure_type == "subdirs_with_videos":
        # 情况2：子目录中包含视频文件
        extract_dir = audio2_path.parent / 'extract_audio2'
        extract_dir.mkdir(exist_ok=True)
        print(f"Extracting audio from subdirectory videos to {extract_dir}")
        
        audio_files = []
        subdirs = sorted([d for d in audio2_path.iterdir() if d.is_dir()])
        
        for subdir in subdirs:
            # 从子目录名中提取ID
            subdir_id = subdir.name.split('_')[0] if '_' in subdir.name else subdir.name
            
            # 在子目录中查找视频文件
            video_files = [f for f in subdir.iterdir() if f.suffix.lower() in video_extensions]
            if not video_files:
                print(f"Warning: No video files found in {subdir}")
                continue
                
            # 取第一个视频文件
            video_file = sorted(video_files)[0]
            output_audio_path = extract_dir / f"{subdir_id}_audio.wav"
            
            if not output_audio_path.exists():
                print(f"Extracting audio from {video_file}...")
                success = extract_audio_from_video(str(video_file), str(output_audio_path))
                if not success:
                    print(f"Failed to extract audio from {video_file}")
                    continue
            else:
                print(f"Audio already exists: {output_audio_path.name}")
            
            audio_files.append(output_audio_path)
        
        return extract_dir, audio_files
    
    elif structure_type == "subdirs_with_audios":
        # 情况1：子目录中包含音频文件
        print("Processing subdirectories with audio files")
        
        audio_files = []
        subdirs = sorted([d for d in audio2_path.iterdir() if d.is_dir()])
        
        for subdir in subdirs:
            # 从子目录名中提取ID
            subdir_id = subdir.name.split('_')[0] if '_' in subdir.name else subdir.name
            
            # 在子目录中查找音频文件
            audio_file_list = [f for f in subdir.iterdir() if f.suffix.lower() in audio_extensions]
            if not audio_file_list:
                print(f"Warning: No audio files found in {subdir}")
                continue
                
            # 取第一个音频文件
            audio_file = sorted(audio_file_list)[0]
            audio_files.append(audio_file)
        
        return audio2_path, audio_files
    
    elif structure_type == "direct_videos":
        # 情况3：直接包含视频文件
        extract_dir = audio2_path.parent / 'extract_audio2'
        extract_dir.mkdir(exist_ok=True)
        print(f"Extracting audio from direct videos to {extract_dir}")
        
        video_files = [f for f in audio2_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
        audio_files = []
        
        for video_file in sorted(video_files):
            # 从文件名中提取ID
            stem = video_file.stem
            import re
            match = re.search(r'(\d{5})', stem)
            if match:
                audio_id = match.group(1)
            else:
                audio_id = stem
            
            output_audio_path = extract_dir / f"{audio_id}_audio.wav"
            
            if not output_audio_path.exists():
                print(f"Extracting audio from {video_file.name}...")
                success = extract_audio_from_video(str(video_file), str(output_audio_path))
                if not success:
                    print(f"Failed to extract audio from {video_file}")
                    continue
            else:
                print(f"Audio already exists: {output_audio_path.name}")
            
            audio_files.append(output_audio_path)
        
        return extract_dir, audio_files
    
    elif structure_type == "direct_audios":
        # 情况4：直接包含音频文件
        print("Processing direct audio files")
        audio_files = [f for f in audio2_path.iterdir() if f.is_file() and f.suffix.lower() in audio_extensions]
        return audio2_path, sorted(audio_files)
    
    else:  # mixed_files
        # 混合情况：同时包含音频和视频文件
        print("Processing mixed audio and video files")
        
        extract_dir = audio2_path.parent / 'extract_audio2'
        extract_dir.mkdir(exist_ok=True)
        
        audio_files = []
        files = sorted([f for f in audio2_path.iterdir() if f.is_file()])
        
        for file_path in files:
            if file_path.suffix.lower() in audio_extensions:
                # 直接使用音频文件
                audio_files.append(file_path)
            elif file_path.suffix.lower() in video_extensions:
                # 提取视频文件的音频
                stem = file_path.stem
                import re
                match = re.search(r'(\d{5})', stem)
                audio_id = match.group(1) if match else stem
                
                output_audio_path = extract_dir / f"{audio_id}_audio.wav"
                
                if not output_audio_path.exists():
                    print(f"Extracting audio from {file_path.name}...")
                    success = extract_audio_from_video(str(file_path), str(output_audio_path))
                    if success:
                        audio_files.append(output_audio_path)
                else:
                    audio_files.append(output_audio_path)
        
        return extract_dir, audio_files

def find_matching_pairs(audio1_dir, audio2_files):
    """找到匹配的音频对"""
    audio1_dir = Path(audio1_dir)
    audio1_files = sorted(audio1_dir.glob('*.wav'))
    
    pairs = []
    
    for audio1_file in audio1_files:
        # 从audio1文件名中提取ID (00001_audio.wav -> 00001)
        stem = audio1_file.stem
        audio1_id = stem.split('_')[0]  # 获取00001部分
        
        # 在audio2文件中查找匹配的ID
        matching_audio2 = None
        for audio2_file in audio2_files:
            audio2_stem = audio2_file.stem
            # 检查是否包含相同的ID
            if audio1_id in audio2_stem:
                matching_audio2 = audio2_file
                break
        
        if matching_audio2:
            pairs.append((audio1_file, matching_audio2, audio1_id))
        else:
            print(f"Warning: No matching audio2 file found for {audio1_file.name}")
    
    return pairs

def process_audio_pair(model, wav1_path, wav2_path, use_gpu=True):
    """处理单个音频对并计算相似度"""
    wav1, sr1 = sf.read(wav1_path)
    wav2, sr2 = sf.read(wav2_path)

    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
    
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)

    if use_gpu:
        wav1 = wav1.cuda()
        wav2 = wav2.cuda()

    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    return sim[0].item()

def verification(model_name, wav1, wav2, use_gpu=True, checkpoint=None):
    """单对音频验证（保持原有接口）"""
    assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    model = init_model(model_name, checkpoint)

    wav1, sr1 = sf.read(wav1)
    wav2, sr2 = sf.read(wav2)

    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)

    if use_gpu:
        model = model.cuda()
        wav1 = wav1.cuda()
        wav2 = wav2.cuda()

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    print("The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(sim[0].item()))

def batch_verification(model_name, audio1_dir, audio2_dir, use_gpu=True, checkpoint=None):
    """批次化音频验证"""
    
    assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    
    # 初始化模型
    model = init_model(model_name, checkpoint)
    if use_gpu:
        model = model.cuda()
    model.eval()
    
    # 准备audio2文件
    audio2_path, audio2_files = prepare_audio2_files(audio2_dir)
    print(f"Found {len(audio2_files)} audio files in audio2 directory")
    
    # 找到匹配的音频对
    pairs = find_matching_pairs(audio1_dir, audio2_files)
    print(f"Found {len(pairs)} matching audio pairs")
    
    if not pairs:
        print("No matching audio pairs found!")
        return
    
    # 处理所有音频对
    results = []
    #output_file = Path(audio2_dir).parent / 'audio_sim.txt'
    output_file = './audio_sim_cosy1.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ID\tAudio1\tAudio2\tSimilarity\n")
        f.write("-" * 50 + "\n")
        
        for audio1_file, audio2_file, pair_id in pairs:
            try:
                similarity = process_audio_pair(model, audio1_file, audio2_file, use_gpu)
                results.append(similarity)
                
                # 写入结果
                line = f"{pair_id}\t{audio1_file.name}\t{audio2_file.name}\t{similarity:.4f}"
                print(line)
                f.write(line + "\n")
                f.flush()
                
            except Exception as e:
                print(f"Error processing pair {pair_id}: {e}")
                results.append(0.0)  # 错误时设为0
    
    # 计算统计信息
    if results:
        avg_similarity = np.mean(results)
        std_similarity = np.std(results)
        max_similarity = np.max(results)
        min_similarity = np.min(results)
        
        print("\n" + "=" * 50)
        print("Batch Verification Results:")
        print(f"Total pairs processed: {len(results)}")
        print(f"Average similarity: {avg_similarity:.4f}")
        print(f"Std similarity: {std_similarity:.4f}")
        print(f"Max similarity: {max_similarity:.4f}")
        print(f"Min similarity: {min_similarity:.4f}")
        
        # 写入统计信息
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"Total pairs: {len(results)}\n")
            f.write(f"Average similarity: {avg_similarity:.4f}\n")
            f.write(f"Std similarity: {std_similarity:.4f}\n")
            f.write(f"Max similarity: {max_similarity:.4f}\n")
            f.write(f"Min similarity: {min_similarity:.4f}\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    fire.Fire({
        'verification': verification,
        'batch_verification': batch_verification
    })
'''
