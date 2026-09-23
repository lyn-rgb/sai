#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import argparse
import glob
import sys
import traceback
import subprocess

# 添加当前路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SyncNetInstance import *
import numpy as np
import json
from tqdm import tqdm


def tensor_to_scalar(value, value_type=float):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().view(-1)[0].item()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1)[0].item()
    return value_type(value)


def evaluate_video_file(s, opt, videofile, source_video, track_file=None):
    result_tuple = s.evaluate(opt, videofile=videofile)
    if result_tuple[0] is None:
        return None

    offset, conf, dist = result_tuple
    return {
        'video_file': source_video,
        'sync_input_file': videofile,
        'track_file': track_file,
        'offset': tensor_to_scalar(offset, int),
        'confidence': tensor_to_scalar(conf, float),
        'min_dist': tensor_to_scalar(dist, float),
    }


def run_crop_pipeline(
    videofile,
    reference,
    data_dir,
    facedet_scale=0.25,
    crop_scale=0.40,
    min_track=100,
    frame_rate=25,
    num_failed_det=25,
    min_face_size=100,
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    s3fd_weight = os.path.join(script_dir, "detectors", "s3fd", "weights", "sfd_face.pth")
    if not os.path.exists(s3fd_weight):
        raise FileNotFoundError(
            f"S3FD face detector weight not found: {s3fd_weight}. "
            "Run evaluation/codes/syncnet/download_model.sh or place sfd_face.pth there."
        )
    cmd = [
        sys.executable,
        os.path.join(script_dir, "run_pipeline.py"),
        "--videofile",
        videofile,
        "--reference",
        reference,
        "--data_dir",
        data_dir,
        "--facedet_scale",
        str(facedet_scale),
        "--crop_scale",
        str(crop_scale),
        "--min_track",
        str(min_track),
        "--frame_rate",
        str(frame_rate),
        "--num_failed_det",
        str(num_failed_det),
        "--min_face_size",
        str(min_face_size),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=script_dir, check=True)
    crop_dir = os.path.join(data_dir, "pycrop", reference)
    return sorted(glob.glob(os.path.join(crop_dir, "*.avi")))


def batch_evaluate(
    video_dir,
    output_file,
    initial_model,
    tmp_base_dir,
    batch_size=10,
    vshift=15,
    use_crop_pipeline=True,
    allow_full_frame_fallback=False,
    facedet_scale=0.25,
    crop_scale=0.40,
    min_track=100,
    frame_rate=25,
    num_failed_det=25,
    min_face_size=100,
):
    """
    批量处理视频文件，计算同步指标
    """
    
    # 初始化SyncNet
    print("Initializing SyncNet...")
    try:
        s = SyncNetInstance()
        s.loadParameters(initial_model)
        print(f"Model {initial_model} loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return
    
    # 获取所有视频文件
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        video_files.extend(glob.glob(os.path.join(video_dir, ext.upper())))
    
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        print(f"在目录 {video_dir} 中未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"前5个视频: {[os.path.basename(f) for f in video_files[:5]]}")
    
    # 存储所有结果
    all_results = []
    
    # 创建临时文件根目录
    os.makedirs(tmp_base_dir, exist_ok=True)
    
    # 处理每个视频
    successful = 0
    failed = 0
    
    for i, videofile in enumerate(tqdm(video_files, desc="Processing videos")):
        try:
            video_name = os.path.splitext(os.path.basename(videofile))[0]
            safe_name = "".join(c for c in video_name if c.isalnum() or c in ['_', '-']).rstrip()
            if not safe_name:
                safe_name = f"video_{i}"
            
            # 创建临时opt对象
            class Opt:
                pass
            
            opt = Opt()
            opt.initial_model = initial_model
            opt.batch_size = batch_size
            opt.vshift = vshift
            opt.videofile = videofile
            opt.tmp_dir = tmp_base_dir
            opt.reference = f"tmp_{safe_name}_{i}"
            
            print(f"\n[{i+1}/{len(video_files)}] 处理: {os.path.basename(videofile)}")

            sync_inputs = []
            crop_error = ""
            if use_crop_pipeline:
                try:
                    pipeline_data_dir = os.path.join(tmp_base_dir, "pipeline")
                    crop_files = run_crop_pipeline(
                        videofile=videofile,
                        reference=opt.reference,
                        data_dir=pipeline_data_dir,
                        facedet_scale=facedet_scale,
                        crop_scale=crop_scale,
                        min_track=min_track,
                        frame_rate=frame_rate,
                        num_failed_det=num_failed_det,
                        min_face_size=min_face_size,
                    )
                    sync_inputs = [(crop_file, crop_file) for crop_file in crop_files]
                    print(f"  -> 检测到 {len(sync_inputs)} 条人脸轨迹")
                except Exception as e:
                    crop_error = str(e)
                    print(f"  -> 人脸裁剪失败: {crop_error}")

                if not sync_inputs and allow_full_frame_fallback:
                    print("  -> fallback 到整帧 SyncNet（不建议用于正式指标）")
                    sync_inputs = [(videofile, None)]
            else:
                sync_inputs = [(videofile, None)]

            track_results = []
            for track_idx, (sync_input_file, track_file) in enumerate(sync_inputs):
                opt.reference = f"tmp_{safe_name}_{i}_track_{track_idx}"
                track_result = evaluate_video_file(
                    s=s,
                    opt=opt,
                    videofile=sync_input_file,
                    source_video=videofile,
                    track_file=track_file,
                )
                if track_result is not None:
                    track_results.append(track_result)

            if not track_results:
                print(f"  -> 处理失败：无法计算指标")
                failed += 1
                result = {
                    'video_file': videofile,
                    'video_name': video_name,
                    'offset': None,
                    'confidence': None,
                    'min_dist': None,
                    'num_tracks': len(sync_inputs),
                    'status': 'failed',
                    'error': crop_error or 'Evaluation returned None'
                }
            else:
                best = max(track_results, key=lambda item: item["confidence"])
                result = {
                    'video_file': videofile,
                    'video_name': video_name,
                    'sync_input_file': best.get('sync_input_file'),
                    'track_file': best.get('track_file'),
                    'offset': best['offset'],
                    'confidence': best['confidence'],
                    'min_dist': best['min_dist'],
                    'num_tracks': len(sync_inputs),
                    'track_results': track_results,
                    'status': 'success'
                }
                
                successful += 1
                print(
                    f"  -> 结果: tracks={len(sync_inputs)}, "
                    f"offset={best['offset']}, confidence={best['confidence']:.3f}, "
                    f"distance={best['min_dist']:.3f}"
                )
            
            all_results.append(result)
            
        except Exception as e:
            print(f"处理视频 {os.path.basename(videofile)} 时出错: {str(e)}")
            traceback.print_exc()
            
            failed += 1
            result = {
                'video_file': videofile,
                'video_name': video_name if 'video_name' in locals() else os.path.basename(videofile),
                'offset': None,
                'confidence': None,
                'min_dist': None,
                'status': 'failed',
                'error': str(e)
            }
            all_results.append(result)
    
    # 计算平均指标
    valid_results = [r for r in all_results if r['status'] == 'success']
    
    print("\n" + "="*60)
    print(f"处理完成！成功: {successful}, 失败: {failed}, 总计: {len(video_files)}")
    
    if valid_results:
        offsets = [r['offset'] for r in valid_results]
        confidences = [r['confidence'] for r in valid_results]
        distances = [r['min_dist'] for r in valid_results]
        
        summary = {
            'total_videos': len(video_files),
            'successful_videos': len(valid_results),
            'failed_videos': len(video_files) - len(valid_results),
            'average_offset': float(np.mean(offsets)),
            'average_confidence': float(np.mean(confidences)),
            'average_distance': float(np.mean(distances)),
            'std_offset': float(np.std(offsets)),
            'std_confidence': float(np.std(confidences)),
            'std_distance': float(np.std(distances)),
            'median_offset': float(np.median(offsets)),
            'median_confidence': float(np.median(confidences)),
            'median_distance': float(np.median(distances)),
            'offset_range': [float(np.min(offsets)), float(np.max(offsets))],
            'confidence_range': [float(np.min(confidences)), float(np.max(confidences))],
            'distance_range': [float(np.min(distances)), float(np.max(distances))]
        }
        summary['preprocess_version'] = 3
        summary['use_crop_pipeline'] = bool(use_crop_pipeline)
        
        print("\n平均指标：")
        print(f"  平均AV偏移: {summary['average_offset']:.2f} 帧")
        print(f"  平均置信度: {summary['average_confidence']:.3f}")
        print(f"  平均距离: {summary['average_distance']:.3f}")
        print(f"  偏移范围: [{summary['offset_range'][0]}, {summary['offset_range'][1]}]")
        print("="*60)
    else:
        summary = {
            'total_videos': len(video_files),
            'successful_videos': 0,
            'failed_videos': len(video_files),
            'preprocess_version': 3,
            'use_crop_pipeline': bool(use_crop_pipeline)
        }
        print("\n没有成功处理的视频！")
        print("可能的原因：")
        print("1. 模型文件可能不是SyncNet v2模型")
        print("2. 视频格式可能不兼容")
        print("3. 需要先进行人脸检测和嘴部裁剪")
    
    # 保存结果
    output_data = {
        'summary': summary,
        'detailed_results': all_results
    }
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description='Batch SyncNet Evaluation')
    
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频文件目录路径')
    parser.add_argument('--output_file', type=str, default='syncnet_results.json',
                        help='输出结果文件路径')
    parser.add_argument('--initial_model', type=str, 
                        default='/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/ConsisID/eval/new_metric/Wav2Lip/evaluation/syncnet_python/data/syncnet_v2.model',
                        help='预训练模型路径')
    parser.add_argument('--tmp_base_dir', type=str, default='./tmp',
                        help='临时文件基础目录')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='批处理大小')
    parser.add_argument('--vshift', type=int, default=15,
                        help='偏移范围')
    parser.add_argument('--no_crop_pipeline', action='store_true',
                        help='不做人脸检测/裁剪，直接整帧评测（仅用于调试，不建议用于正式指标）')
    parser.add_argument('--allow_full_frame_fallback', action='store_true',
                        help='人脸裁剪失败时 fallback 到整帧评测（不建议用于正式指标）')
    parser.add_argument('--facedet_scale', type=float, default=0.25)
    parser.add_argument('--crop_scale', type=float, default=0.40)
    parser.add_argument('--min_track', type=int, default=100)
    parser.add_argument('--frame_rate', type=int, default=25)
    parser.add_argument('--num_failed_det', type=int, default=25)
    parser.add_argument('--min_face_size', type=int, default=100)
    
    opt = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(opt.initial_model):
        print(f"错误：模型文件不存在: {opt.initial_model}")
        sys.exit(1)
    
    if not os.path.exists(opt.video_dir):
        print(f"错误：视频目录不存在: {opt.video_dir}")
        sys.exit(1)
    
    print("="*60)
    print("SyncNet 批量评估工具")
    print("="*60)
    print(f"视频目录: {opt.video_dir}")
    print(f"输出文件: {opt.output_file}")
    print(f"模型文件: {opt.initial_model}")
    print(f"临时目录: {opt.tmp_base_dir}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Vshift: {opt.vshift}")
    print(f"Use crop pipeline: {not opt.no_crop_pipeline}")
    print("="*60)
    
    batch_evaluate(
        video_dir=opt.video_dir,
        output_file=opt.output_file,
        initial_model=opt.initial_model,
        tmp_base_dir=opt.tmp_base_dir,
        batch_size=opt.batch_size,
        vshift=opt.vshift,
        use_crop_pipeline=not opt.no_crop_pipeline,
        allow_full_frame_fallback=opt.allow_full_frame_fallback,
        facedet_scale=opt.facedet_scale,
        crop_scale=opt.crop_scale,
        min_track=opt.min_track,
        frame_rate=opt.frame_rate,
        num_failed_det=opt.num_failed_det,
        min_face_size=opt.min_face_size,
    )

if __name__ == "__main__":
    main()
