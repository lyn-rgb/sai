import os
import argparse
import json
import numpy as np
from pathlib import Path
import torch
import time
import re
from typing import Dict, List, Tuple
from tqdm import tqdm

try:
    import whisper
except ImportError:
    whisper = None

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except ImportError:
    FasterWhisperModel = None

HAS_OPENAI_WHISPER = whisper is not None and hasattr(whisper, "load_model")

def normalize(text: str) -> str:
    """统一大小写、去标点、去多余空格"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # 去掉标点
    text = re.sub(r"\s+", " ", text).strip()
    return text

class WhisperASR:
    def __init__(self, model_path_or_name, device="cuda"):
        self.device = device
        self.backend = None
        print(f"Loading Whisper model: {model_path_or_name}")
        model_path = Path(str(model_path_or_name)).expanduser()
        prefer_faster_whisper = model_path.exists() and model_path.is_dir()
        if prefer_faster_whisper and FasterWhisperModel is not None:
            self.backend = "faster-whisper"
            compute_type = "float16" if device.startswith("cuda") and torch.cuda.is_available() else "float32"
            self.model = FasterWhisperModel(str(model_path), device="cuda" if device.startswith("cuda") else "cpu", compute_type=compute_type)
        elif HAS_OPENAI_WHISPER:
            self.backend = "openai-whisper"
            self.model = whisper.load_model(model_path_or_name, device=device)
        elif FasterWhisperModel is not None:
            self.backend = "faster-whisper"
            compute_type = "float16" if device.startswith("cuda") and torch.cuda.is_available() else "float32"
            self.model = FasterWhisperModel(str(model_path_or_name), device="cuda" if device.startswith("cuda") else "cpu", compute_type=compute_type)
        else:
            whisper_path = getattr(whisper, "__file__", None) if whisper is not None else None
            raise ImportError(
                "WER requires either faster-whisper or openai-whisper. "
                "Install one of them, for example: pip install faster-whisper. "
                f"Current whisper module is not OpenAI Whisper: {whisper_path}"
            )
        print(f"Model loaded successfully with {self.backend} on {device}")
    
    def transcribe_audio(self, audio_path, language="en"):
        try:
            if self.backend == "faster-whisper":
                segments, _ = self.model.transcribe(str(audio_path), language=language)
                return " ".join(segment.text.strip() for segment in segments).strip()
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                fp16=torch.cuda.is_available(),
                verbose=None
            )
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""

def calculate_wer_details(reference: str, hypothesis: str) -> Dict:
    """
    计算WER及详细错误统计
    """
    # 文本标准化
    ref_normalized = normalize(reference)
    hyp_normalized = normalize(hypothesis)
    
    ref_words = ref_normalized.split()
    hyp_words = hyp_normalized.split()
    
    # 计算编辑距离
    n_ref = len(ref_words)
    n_hyp = len(hyp_words)
    
    if n_ref == 0:
        return {
            'wer': 1.0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': n_hyp,
            'total_words': 0,
            'total_errors': n_hyp,
            'normalized_reference': ref_normalized,
            'normalized_hypothesis': hyp_normalized
        }
    
    # 初始化编辑距离矩阵
    dp = np.zeros((n_ref + 1, n_hyp + 1))
    for i in range(n_ref + 1):
        dp[i][0] = i
    for j in range(n_hyp + 1):
        dp[0][j] = j
    
    # 填充编辑距离矩阵
    for i in range(1, n_ref + 1):
        for j in range(1, n_hyp + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # 删除
                    dp[i][j-1] + 1,    # 插入  
                    dp[i-1][j-1] + 1   # 替换
                )
    
    # 回溯计算错误类型
    substitutions = deletions = insertions = 0
    i, j = n_ref, n_hyp
    
    while i > 0 and j > 0:
        if ref_words[i-1] == hyp_words[j-1]:
            i -= 1
            j -= 1
        else:
            if dp[i][j] == dp[i-1][j-1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + 1:
                deletions += 1
                i -= 1
            else:
                insertions += 1
                j -= 1
    
    # 处理剩余部分
    while i > 0:
        deletions += 1
        i -= 1
    while j > 0:
        insertions += 1
        j -= 1
    
    # 计算WER
    total_errors = substitutions + deletions + insertions
    wer = total_errors / n_ref
    
    return {
        'wer': wer,
        'substitutions': substitutions,
        'deletions': deletions, 
        'insertions': insertions,
        'total_words': n_ref,
        'total_errors': total_errors,
        'normalized_reference': ref_normalized,
        'normalized_hypothesis': hyp_normalized
    }

def find_matching_audio_files(audio_dir, base_name):
    """在音频目录中查找匹配的音频文件"""
    audio_dir = Path(audio_dir)
    matching_files = list(audio_dir.glob(f"{base_name}*.wav"))
    
    if not matching_files:
        matching_files = list(audio_dir.glob(f"{base_name}*.flac")) + \
                        list(audio_dir.glob(f"{base_name}*.mp3"))
    
    return matching_files[0] if matching_files else None

def load_reference_texts(gt_dir):
    """加载参考文本"""
    references = {}
    gt_dir = Path(gt_dir)
    
    text_patterns = ["*_gt.txt", "*_reference.txt", "*.txt"]
    
    for pattern in text_patterns:
        for gt_file in gt_dir.glob(pattern):
            base_name = gt_file.stem.replace('_gt', '').replace('_reference', '')
            if base_name in references:
                continue
                
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                references[base_name] = text
            except Exception as e:
                print(f"Error reading {gt_file}: {e}")
    
    print(f"Loaded {len(references)} reference texts from {gt_dir}")
    return references

def process_single_file(args):
    """处理单个文件对"""
    base_name, reference, audio_dir, asr_model, lang = args
    
    # 查找匹配的音频文件
    audio_file = find_matching_audio_files(audio_dir, base_name)
    if not audio_file:
        print(f"No audio file found for {base_name}")
        return None
    
    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        return None
    
    print(f"Processing: {base_name}")
    print(f"  Original Ref: '{reference}'")
    
    # 运行ASR获取识别结果
    start_time = time.time()
    hypothesis = asr_model.transcribe_audio(str(audio_file), language=lang)
    asr_time = time.time() - start_time
    
    print(f"  Original Hyp: '{hypothesis}'")
    
    # 计算WER
    wer_details = calculate_wer_details(reference, hypothesis)
    
    print(f"  Normalized Ref: '{wer_details['normalized_reference']}'")
    print(f"  Normalized Hyp: '{wer_details['normalized_hypothesis']}'")
    print(f"  WER = {wer_details['wer']:.4f}, ASR time = {asr_time:.2f}s")
    print(f"  Errors: S={wer_details['substitutions']}, D={wer_details['deletions']}, I={wer_details['insertions']}")
    print("-" * 80)
    
    result = {
        'base_name': base_name,
        'audio_file': str(audio_file),
        'reference': reference,
        'hypothesis': hypothesis,
        'asr_time': asr_time,
        'wer': wer_details['wer'],
        'wer_details': wer_details
    }
    
    return result

def calculate_wer_batch(gt_dir, audio_dir, model_path, lang="en"):
    """批量计算WER"""
    # 加载参考文本
    references = load_reference_texts(gt_dir)
    
    if not references:
        print("No reference texts found!")
        return []
    
    # 检查音频目录
    audio_files = list(Path(audio_dir).glob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")
    
    # 初始化ASR模型
    asr_model = WhisperASR(model_path, "cuda")
    results = []
    
    # 处理每个文件
    for base_name, reference in tqdm(references.items(), desc="WER", unit="audio"):
        task = (base_name, reference, audio_dir, asr_model, lang)
        result = process_single_file(task)
        if result is not None:
            results.append(result)
    
    return results

def calculate_basic_statistics(results):
    """计算基础统计信息"""
    if not results:
        return {}
    
    wer_scores = [r['wer'] for r in results]
    asr_times = [r['asr_time'] for r in results]
    
    total_substitutions = sum(r['wer_details']['substitutions'] for r in results)
    total_deletions = sum(r['wer_details']['deletions'] for r in results)
    total_insertions = sum(r['wer_details']['insertions'] for r in results)
    total_words = sum(r['wer_details']['total_words'] for r in results)
    
    avg_wer = np.mean(wer_scores)
    avg_asr_time = np.mean(asr_times)
    
    return {
        'average_wer': avg_wer,
        'wer_percentage': avg_wer * 100,
        'min_wer': min(wer_scores),
        'max_wer': max(wer_scores),
        'std_wer': np.std(wer_scores),
        'total_substitutions': total_substitutions,
        'total_deletions': total_deletions,
        'total_insertions': total_insertions,
        'total_words': total_words,
        'total_errors': total_substitutions + total_deletions + total_insertions,
        'average_asr_time': avg_asr_time,
        'total_asr_time': sum(asr_times)
    }

def calculate_robust_wer_metrics(results):
    """
    计算修正后的WER统计，排除WER>100%的异常值
    """
    # 分离正常文件和异常文件
    normal_results = [r for r in results if r['wer'] <= 1.0]
    abnormal_results = [r for r in results if r['wer'] > 1.0]
    
    if not normal_results:
        print("警告：所有文件的WER都超过100%！")
        return None
    
    # 计算正常文件的统计
    normal_wer_scores = [r['wer'] for r in normal_results]
    normal_asr_times = [r['asr_time'] for r in normal_results]
    
    # 错误统计
    total_substitutions = sum(r['wer_details']['substitutions'] for r in normal_results)
    total_deletions = sum(r['wer_details']['deletions'] for r in normal_results)
    total_insertions = sum(r['wer_details']['insertions'] for r in normal_results)
    total_words = sum(r['wer_details']['total_words'] for r in normal_results)
    total_errors = total_substitutions + total_deletions + total_insertions
    
    # 计算整体WER（合并所有文本）
    all_references = " ".join([r['reference'] for r in normal_results])
    all_hypotheses = " ".join([r['hypothesis'] for r in normal_results])
    overall_wer_details = calculate_wer_details(all_references, all_hypotheses)
    
    # 计算统计指标
    robust_metrics = {
        'original_total_files': len(results),
        'normal_files_count': len(normal_results),
        'abnormal_files_count': len(abnormal_results),
        'abnormal_file_ids': [r['base_name'] for r in abnormal_results],
        
        # WER统计
        'robust_avg_wer': np.mean(normal_wer_scores),
        'robust_wer_percentage': np.mean(normal_wer_scores) * 100,
        'overall_wer': overall_wer_details['wer'],
        'overall_wer_percentage': overall_wer_details['wer'] * 100,
        'min_wer': min(normal_wer_scores),
        'max_wer': max(normal_wer_scores),
        'std_wer': np.std(normal_wer_scores),
        
        # 错误统计
        'total_substitutions': total_substitutions,
        'total_deletions': total_deletions,
        'total_insertions': total_insertions,
        'total_words': total_words,
        'total_errors': total_errors,
        'overall_total_words': overall_wer_details['total_words'],
        'overall_total_errors': overall_wer_details['total_errors'],
        
        # 时间统计
        'avg_asr_time': np.mean(normal_asr_times),
        'total_asr_time': sum(normal_asr_times),
        
        # 原始统计（用于对比）
        'original_avg_wer': np.mean([r['wer'] for r in results]),
        'original_wer_percentage': np.mean([r['wer'] for r in results]) * 100
    }
    
    return robust_metrics

def analyze_abnormal_cases(abnormal_results):
    """
    分析异常案例的详细信息
    """
    if not abnormal_results:
        return "没有异常案例"
    
    analysis = "异常案例分析:\n"
    analysis += "=" * 50 + "\n"
    
    for result in abnormal_results:
        analysis += f"文件: {result['base_name']}\n"
        analysis += f"WER: {result['wer']:.1%}\n"
        analysis += f"参考文本: '{result['reference']}'\n"
        analysis += f"识别结果: '{result['hypothesis']}'\n"
        analysis += f"错误详情: S={result['wer_details']['substitutions']}, "
        analysis += f"D={result['wer_details']['deletions']}, "
        analysis += f"I={result['wer_details']['insertions']}\n"
        analysis += f"参考词数: {result['wer_details']['total_words']}\n"
        analysis += "-" * 50 + "\n"
    
    return analysis

def generate_comprehensive_report(results_file_path, output_dir=None):
    """
    生成完整的修正报告
    """
    # 读取结果文件
    with open(results_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 计算修正指标
    robust_metrics = calculate_robust_wer_metrics(results)
    
    if robust_metrics is None:
        print("无法计算修正指标")
        return
    
    # 确定输出路径
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(results_file_path).stem
        output_path = output_dir / f"{base_name}_corrected_report.txt"
        corrected_output_path = output_dir / f"{base_name}_corrected.json"
    else:
        output_path = results_file_path.replace('.json', '_corrected_report.txt')
        corrected_output_path = results_file_path.replace('.json', '_corrected.json')
    
    # 生成报告
    report = []
    report.append("F5-TTS WER评估综合报告")
    report.append("=" * 60)
    report.append(f"数据文件: {Path(results_file_path).name}")
    report.append(f"原始总文件数: {robust_metrics['original_total_files']}")
    report.append(f"正常文件数: {robust_metrics['normal_files_count']}")
    report.append(f"异常文件数: {robust_metrics['abnormal_files_count']}")
    
    if robust_metrics['abnormal_files_count'] > 0:
        report.append(f"异常文件ID: {', '.join(robust_metrics['abnormal_file_ids'])}")
    
    report.append("\nWER统计对比:")
    report.append("-" * 40)
    report.append(f"原始平均WER: {robust_metrics['original_avg_wer']:.4f} ({robust_metrics['original_wer_percentage']:.2f}%)")
    report.append(f"文件级平均WER: {robust_metrics['robust_avg_wer']:.4f} ({robust_metrics['robust_wer_percentage']:.2f}%)")
    report.append(f"整体WER: {robust_metrics['overall_wer']:.4f} ({robust_metrics['overall_wer_percentage']:.2f}%)")
    report.append(f"WER范围: {robust_metrics['min_wer']:.4f} - {robust_metrics['max_wer']:.4f}")
    report.append(f"标准差: {robust_metrics['std_wer']:.4f}")
    
    report.append("\n错误分析:")
    report.append("-" * 40)
    report.append(f"总词数: {robust_metrics['total_words']} (文件级) / {robust_metrics['overall_total_words']} (整体)")
    report.append(f"总错误数: {robust_metrics['total_errors']} (文件级) / {robust_metrics['overall_total_errors']} (整体)")
    if robust_metrics['total_errors'] > 0:
        report.append(f"替换错误(S): {robust_metrics['total_substitutions']} ({robust_metrics['total_substitutions']/robust_metrics['total_errors']*100:.1f}%)")
        report.append(f"删除错误(D): {robust_metrics['total_deletions']} ({robust_metrics['total_deletions']/robust_metrics['total_errors']*100:.1f}%)")
        report.append(f"插入错误(I): {robust_metrics['total_insertions']} ({robust_metrics['total_insertions']/robust_metrics['total_errors']*100:.1f}%)")
    else:
        report.append("替换错误(S): 0 (0%)")
        report.append("删除错误(D): 0 (0%)")
        report.append("插入错误(I): 0 (0%)")
    
    report.append("\n性能评估:")
    report.append("-" * 40)
    wer_to_evaluate = min(robust_metrics['robust_wer_percentage'], robust_metrics['overall_wer_percentage'])
    if wer_to_evaluate <= 5.0:
        report.append("✅ 优秀水平 (WER ≤ 5%) - 接近商业TTS系统")
    elif wer_to_evaluate <= 10.0:
        report.append("✅ 良好水平 (WER ≤ 10%) - 研究级TTS系统")
    elif wer_to_evaluate <= 15.0:
        report.append("⚠️  一般水平 (WER ≤ 15%) - 可用的TTS系统")
    else:
        report.append("❌ 需要改进 (WER > 15%)")
    
    report.append("\n性能统计:")
    report.append("-" * 40)
    report.append(f"平均ASR时间: {robust_metrics['avg_asr_time']:.2f}s")
    report.append(f"总ASR时间: {robust_metrics['total_asr_time']:.2f}s")
    
    # 异常案例分析
    abnormal_results = [r for r in results if r['wer'] > 1.0]
    report.append("\n" + analyze_abnormal_cases(abnormal_results))
    
    # 保存报告
    report_text = "\n".join(report)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 同时保存修正后的JSON
    corrected_data = data.copy()
    
    # 确保statistics字段存在
    if 'statistics' not in corrected_data:
        corrected_data['statistics'] = {}
    
    corrected_data['statistics']['robust_metrics'] = robust_metrics
    corrected_data['statistics']['overall_wer_analysis'] = {
        'overall_wer': robust_metrics['overall_wer'],
        'overall_wer_percentage': robust_metrics['overall_wer_percentage'],
        'overall_total_words': robust_metrics['overall_total_words'],
        'overall_total_errors': robust_metrics['overall_total_errors']
    }
    
    with open(corrected_output_path, 'w', encoding='utf-8') as f:
        json.dump(corrected_data, f, ensure_ascii=False, indent=2)
    
    print(report_text)
    print(f"\n报告已保存至: {output_path}")
    print(f"修正数据已保存至: {corrected_output_path}")
    
    return robust_metrics

def main():
    parser = argparse.ArgumentParser(description='TTS系统WER计算与评估工具')
    
    # 主要模式选择
    parser.add_argument('--mode', type=str, choices=['calculate', 'analyze', 'both'], default='both',
                       help='运行模式: calculate(仅计算), analyze(仅分析), both(计算+分析)')
    
    # 计算模式参数
    parser.add_argument('--gt_dir', type=str, 
                       help='Directory containing ground truth text files')
    parser.add_argument('--audio_dir', type=str,
                       help='Directory containing generated audio files')
    parser.add_argument('--model_path', type=str, default='large-v3',
                       help='Path to Whisper model or model name')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'zh'],
                       help='Language of the audio files')
    
    # 分析模式参数
    parser.add_argument('--input', '-i', type=str,
                       help='输入的WER结果JSON文件路径')
    
    # 通用参数
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出目录路径')
    parser.add_argument('--output_file', type=str, default='tts_wer_results.json',
                       help='输出JSON文件名')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    if args.mode in ['calculate', 'both']:
        if not args.gt_dir or not args.audio_dir:
            print("错误: 计算模式需要 --gt_dir 和 --audio_dir 参数")
            return
        
        print("开始计算WER...")
        # 计算WER
        results = calculate_wer_batch(
            args.gt_dir, 
            args.audio_dir, 
            args.model_path, 
            args.lang
        )
        
        if not results:
            print("没有生成结果！")
            return
        
        # 确定输出文件路径
        if output_dir:
            output_file = output_dir / args.output_file
        else:
            output_file = args.output_file
        
        # 计算基础统计信息
        basic_stats = calculate_basic_statistics(results)
        total_time = sum(r['asr_time'] for r in results)
        
        # 保存原始结果
        output_data = {
            'config': {
                'gt_dir': args.gt_dir,
                'audio_dir': args.audio_dir,
                'model_path': args.model_path,
                'lang': args.lang,
                'total_files': len(results),
                'processing_time': total_time
            },
            'results': results,
            'statistics': basic_stats  # 添加基础统计信息
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"原始结果已保存至: {output_file}")
        
        # 如果模式是both，则继续分析
        if args.mode == 'both':
            args.input = output_file
    
    if args.mode in ['analyze', 'both']:
        if not args.input:
            print("错误: 分析模式需要 --input 参数")
            return
        
        print("\n开始分析WER结果...")
        robust_metrics = generate_comprehensive_report(args.input, args.output)
        
        if robust_metrics:
            print(f"\n🎯 最终评估结果:")
            print(f"   文件级平均WER: {robust_metrics['robust_wer_percentage']:.2f}%")
            print(f"   整体WER: {robust_metrics['overall_wer_percentage']:.2f}%")
            print(f"   基于 {robust_metrics['normal_files_count']} 个正常文件")
            print(f"   排除 {robust_metrics['abnormal_files_count']} 个异常文件")
            
            # 性能评级
            wer_to_evaluate = min(robust_metrics['robust_wer_percentage'], robust_metrics['overall_wer_percentage'])
            if wer_to_evaluate <= 5.0:
                print("   🏆 性能评级: 优秀")
            elif wer_to_evaluate <= 10.0:
                print("   ✅ 性能评级: 良好") 
            else:
                print("   ⚠️  性能评级: 一般")

if __name__ == "__main__":
    main()































######增加了wer.txt保存机制#######
'''
import os
import re
import glob
import argparse
import numpy as np
from tqdm import tqdm
import jiwer
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import datetime

class TextNormalizer:
    def __init__(self, remove_fillers=True):
        self.remove_fillers = remove_fillers
        
        # 口语化表达映射
        self.contraction_map = {
            "gotta": "got to",
            "wanna": "want to", 
            "gonna": "going to",
            "kinda": "kind of",
            "sorta": "sort of",
            "lemme": "let me",
            "gimme": "give me",
            "dunno": "do not know",
            "ain't": "is not",
            "ya": "you",
            "schoolroom": "school room",
            "livingroom": "living room", 
            "bedroom": "bed room",
            "bathroom": "bath room",
            "classroom": "class room",
        }
        
        # 语气助词列表
        self.filler_words = [
            "uh", "um", "ah", "er", "hm", "hmm", 
            "like", "you know", "i mean", "actually",
            "basically", "literally", "honestly",
            "okay", "ok", "right", "so", "well",
            "anyway", "anyways", "yeah", "yes", "no",
            "uh-huh", "uh-uh", "mm-hmm", "huh"
        ]
        
        # 创建语气助词的正则表达式模式
        self.filler_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(filler) for filler in self.filler_words) + r')\b',
            re.IGNORECASE
        )
    
    def normalize(self, text):
        """规范化文本"""
        if not text or not text.strip():
            return ""
            
        text = text.lower()
        
        # 第一步：替换口语化表达
        words = text.split()
        normalized_words = []
        
        for word in words:
            # 移除标点但保留单词
            clean_word = re.sub(r'[^\w\s-]', '', word)
            if clean_word in self.contraction_map:
                normalized_words.extend(self.contraction_map[clean_word].split())
            else:
                normalized_words.append(clean_word)
        
        normalized_text = ' '.join(normalized_words)
        
        # 第二步：移除语气助词
        if self.remove_fillers:
            normalized_text = self.filler_pattern.sub('', normalized_text)
        
        # 第三步：清理多余空格
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        
        return normalized_text

class TTSWEREvaluator:
    def __init__(self, model_size="large-v3", language="zh"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.language = language
        
        print(f"加载Whisper模型: {model_size}")
        model_id = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/pretrain/openai/whisper-large-v3"
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        # 初始化文本规范化器
        self.normalizer = TextNormalizer(remove_fillers=True)
    
    def load_ground_truth(self, ground_truth_file):
        """加载ground truth文本"""
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truths = [line.strip() for line in f if line.strip()]
        return ground_truths
    
    def recognize_audio_batch(self, audio_files):
        """批量识别音频"""
        texts = []
        for audio_file in tqdm(audio_files, desc="语音识别"):
            try:
                result = self.pipe(
                    audio_file,
                    generate_kwargs={"language": self.language} if self.language else None
                )
                texts.append(result["text"].strip())
            except Exception as e:
                print(f"识别错误 {audio_file}: {e}")
                texts.append("")
        return texts
    
    def calculate_wer(self, ground_truths, recognized_texts, audio_files):
        """计算WER - 包含完整的文本规范化"""
        if len(ground_truths) != len(recognized_texts):
            print(f"警告: ground truth数量({len(ground_truths)})与识别结果数量({len(recognized_texts)})不匹配")
            min_len = min(len(ground_truths), len(recognized_texts))
            ground_truths = ground_truths[:min_len]
            recognized_texts = recognized_texts[:min_len]
            audio_files = audio_files[:min_len]
        
        # 应用文本规范化
        print("应用文本规范化...")
        normalized_ground_truths = [self.normalizer.normalize(gt) for gt in ground_truths]
        normalized_recognized = [self.normalizer.normalize(rec) for rec in recognized_texts]
        
        # 标准预处理
        if self.language in ["zh", "cn"]:
            transformation = jiwer.Compose([
                jiwer.RemoveWhiteSpace(replace_by_space=False),
                jiwer.RemoveEmptyStrings(),
            ])
        else:
            transformation = jiwer.Compose([
                jiwer.RemovePunctuation(),
                jiwer.ToLowerCase(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ])
        
        gt_processed = transformation(normalized_ground_truths)
        rec_processed = transformation(normalized_recognized)
        
        # 计算每个样本的WER
        individual_results = []
        individual_wers = []
        
        for i, (gt, rec, audio_file) in enumerate(zip(gt_processed, rec_processed, audio_files)):
            if gt:  # 避免除零错误
                wer_score = jiwer.wer(gt, rec)
                individual_wers.append(wer_score)
                
                # 保存每个样本的详细结果
                individual_results.append({
                    'audio_file': os.path.basename(audio_file),
                    'ground_truth': ground_truths[i],
                    'recognized': recognized_texts[i],
                    'normalized_gt': normalized_ground_truths[i],
                    'normalized_rec': normalized_recognized[i],
                    'wer': wer_score
                })
            else:
                individual_results.append({
                    'audio_file': os.path.basename(audio_file),
                    'ground_truth': ground_truths[i],
                    'recognized': recognized_texts[i],
                    'normalized_gt': normalized_ground_truths[i],
                    'normalized_rec': normalized_recognized[i],
                    'wer': float('inf')  # 表示无效结果
                })
        
        # 计算平均WER（只计算有效的样本）
        valid_wers = [w for w in individual_wers if w != float('inf')]
        if valid_wers:
            avg_wer = np.mean(valid_wers)
            wer_std = np.std(valid_wers)
            wer_min = np.min(valid_wers)
            wer_max = np.max(valid_wers)
        else:
            avg_wer = float('inf')
            wer_std = 0
            wer_min = float('inf')
            wer_max = float('inf')
        
        return {
            "avg_wer": avg_wer,
            "wer_std": wer_std,
            "wer_min": wer_min,
            "wer_max": wer_max,
            "num_samples": len(ground_truths),
            "num_valid_samples": len(valid_wers),
            "individual_results": individual_results,
            "individual_wers": individual_wers,
        }
    
    def save_results_to_file(self, metrics, output_file="wer.txt"):
        """保存详细结果到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入头部信息
            f.write("TTS音色克隆WER评估结果\n")
            f.write("=" * 80 + "\n")
            f.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"语言: {self.language}\n")
            f.write(f"总样本数: {metrics['num_samples']}\n")
            f.write(f"有效样本数: {metrics['num_valid_samples']}\n")
            f.write("\n")
            
            # 写入总体统计
            f.write("总体统计:\n")
            f.write("-" * 40 + "\n")
            f.write(f"平均 WER: {metrics['avg_wer']:.4f} ({metrics['avg_wer']*100:.2f}%)\n")
            f.write(f"WER 标准差: {metrics['wer_std']:.4f}\n")
            f.write(f"最小 WER: {metrics['wer_min']:.4f}\n")
            f.write(f"最大 WER: {metrics['wer_max']:.4f}\n")
            f.write("\n")
            
            # 写入WER分布
            wers = [result['wer'] for result in metrics['individual_results'] if result['wer'] != float('inf')]
            ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]
            range_labels = ["0-10%", "10-20%", "20-30%", "30-50%", "50-100%"]
            
            f.write("WER分布:\n")
            for (low, high), label in zip(ranges, range_labels):
                count = len([w for w in wers if low <= w < high])
                percentage = (count / len(wers)) * 100 if wers else 0
                f.write(f"  {label}: {count}个样本 ({percentage:.1f}%)\n")
            f.write("\n")
            
            # 写入每个样本的详细结果
            f.write("每个样本的详细结果:\n")
            f.write("=" * 80 + "\n")
            
            for i, result in enumerate(metrics['individual_results']):
                f.write(f"样本 {i+1}:\n")
                f.write(f"  音频文件: {result['audio_file']}\n")
                f.write(f"  WER: {result['wer']:.4f}\n")
                f.write(f"  原始GT: {result['ground_truth']}\n")
                f.write(f"  原始识别: {result['recognized']}\n")
                f.write(f"  规范化GT: {result['normalized_gt']}\n")
                f.write(f"  规范化识别: {result['normalized_rec']}\n")
                f.write("-" * 80 + "\n")
            
            # 写入总结
            f.write("\n总结:\n")
            f.write("-" * 40 + "\n")
            if metrics['avg_wer'] < 0.1:
                f.write("WER < 10%: 优秀 - TTS系统语音清晰度很好\n")
            elif metrics['avg_wer'] < 0.2:
                f.write("WER 10-20%: 良好 - TTS系统语音清晰度较好\n")
            elif metrics['avg_wer'] < 0.3:
                f.write("WER 20-30%: 中等 - TTS系统语音清晰度一般\n")
            elif metrics['avg_wer'] < 0.4:
                f.write("WER 30-40%: 较差 - TTS系统需要改进\n")
            else:
                f.write("WER > 40%: 很差 - TTS系统语音清晰度问题严重\n")
        
        print(f"\n详细结果已保存到: {output_file}")
    
    def evaluate(self, audio_dir, ground_truth_file, output_file="wer.txt"):
        """执行评估"""
        print("开始TTS音色克隆WER评估")
        print("=" * 60)
        
        # 1. 加载ground truth
        ground_truths = self.load_ground_truth(ground_truth_file)
        print(f"加载 {len(ground_truths)} 条ground truth文本")
        
        # 2. 获取音频文件
        audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        if not audio_files:
            # 尝试其他音频格式
            audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.mp3"))) + \
                         sorted(glob.glob(os.path.join(audio_dir, "*.flac")))
        
        if not audio_files:
            raise ValueError(f"在 {audio_dir} 中未找到音频文件")
        print(f"找到 {len(audio_files)} 个合成音频文件")
        
        # 3. 语音识别
        recognized_texts = self.recognize_audio_batch(audio_files)
        
        # 4. 计算WER
        metrics = self.calculate_wer(ground_truths, recognized_texts, audio_files)
        
        # 5. 显示结果
        self.print_results(metrics)
        
        # 6. 保存结果到文件
        self.save_results_to_file(metrics, output_file)
        
        return metrics
    
    def print_results(self, metrics):
        """打印简要结果到控制台"""
        print("\n" + "=" * 80)
        print("TTS音色克隆WER评估结果")
        print("=" * 80)
        
        print(f"总样本数: {metrics['num_samples']}")
        print(f"有效样本数: {metrics['num_valid_samples']}")
        print(f"平均 WER: {metrics['avg_wer']:.4f} ({metrics['avg_wer']*100:.2f}%)")
        print(f"WER标准差: {metrics['wer_std']:.4f}")
        print(f"最小 WER: {metrics['wer_min']:.4f}")
        print(f"最大 WER: {metrics['wer_max']:.4f}")
        
        # 显示前几个样本的简要信息
        print(f"\n前3个样本的WER:")
        for i in range(min(3, len(metrics['individual_results']))):
            result = metrics['individual_results'][i]
            print(f"  样本 {i+1}: {result['audio_file']} - WER: {result['wer']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='TTS音色克隆WER评估（保存详细结果）')
    parser.add_argument("-i", "--input", required=True, help="TTS合成音频目录")
    parser.add_argument("-g", "--ground_truth", required=True, help="Ground truth文本文件")
    parser.add_argument("-o", "--output", default="wer.txt", help="输出结果文件")
    parser.add_argument("-l", "--language", choices=['zh', 'en'], default='en', help="语言")
    parser.add_argument("-m", "--model", default="large-v3", help="Whisper模型大小")
    
    args = parser.parse_args()
    
    evaluator = TTSWEREvaluator(model_size=args.model, language=args.language)
    results = evaluator.evaluate(args.input, args.ground_truth, args.output)

if __name__ == "__main__":
    main()
'''


######初始版本#######
'''
import os
import re
import glob
import argparse
import numpy as np
from tqdm import tqdm
import jiwer
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class TextNormalizer:
    def __init__(self, remove_fillers=True):
        self.remove_fillers = remove_fillers
        
        # 口语化表达映射
        self.contraction_map = {
            "gotta": "got to",
            "wanna": "want to", 
            "gonna": "going to",
            "kinda": "kind of",
            "sorta": "sort of",
            "lemme": "let me",
            "gimme": "give me",
            "dunno": "do not know",
            "ain't": "is not",
            "ya": "you",
            "schoolroom": "school room",
            "livingroom": "living room", 
            "bedroom": "bed room",
            "bathroom": "bath room",
            "classroom": "class room",
        }
        
        # 语气助词列表
        self.filler_words = [
            "uh", "um", "ah", "er", "hm", "hmm", 
            "like", "you know", "i mean", "actually",
            "basically", "literally", "honestly",
            "okay", "ok", "right", "so", "well",
            "anyway", "anyways", "yeah", "yes", "no",
            "uh-huh", "uh-uh", "mm-hmm", "huh"
        ]
        
        # 创建语气助词的正则表达式模式
        self.filler_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(filler) for filler in self.filler_words) + r')\b',
            re.IGNORECASE
        )
    
    def normalize(self, text):
        """规范化文本"""
        if not text or not text.strip():
            return ""
            
        text = text.lower()
        
        # 第一步：替换口语化表达
        words = text.split()
        normalized_words = []
        
        for word in words:
            # 移除标点但保留单词
            clean_word = re.sub(r'[^\w\s-]', '', word)
            if clean_word in self.contraction_map:
                normalized_words.extend(self.contraction_map[clean_word].split())
            else:
                normalized_words.append(clean_word)
        
        normalized_text = ' '.join(normalized_words)
        
        # 第二步：移除语气助词
        if self.remove_fillers:
            normalized_text = self.filler_pattern.sub('', normalized_text)
        
        # 第三步：清理多余空格
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        
        return normalized_text

class TTSWEREvaluator:
    def __init__(self, model_size="large-v3", language="zh"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.language = language
        
        print(f"加载Whisper模型: {model_size}")
        model_id = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/pretrain/openai/whisper-large-v3"
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        # 初始化文本规范化器
        self.normalizer = TextNormalizer(remove_fillers=True)
    
    def load_ground_truth(self, ground_truth_file):
        """加载ground truth文本"""
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truths = [line.strip() for line in f if line.strip()]
        return ground_truths
    
    def recognize_audio_batch(self, audio_files):
        """批量识别音频"""
        texts = []
        for audio_file in tqdm(audio_files, desc="语音识别"):
            try:
                result = self.pipe(
                    audio_file,
                    generate_kwargs={"language": self.language} if self.language else None
                )
                texts.append(result["text"].strip())
            except Exception as e:
                print(f"识别错误 {audio_file}: {e}")
                texts.append("")
        return texts
    
    def calculate_wer(self, ground_truths, recognized_texts):
        """计算WER - 包含完整的文本规范化"""
        if len(ground_truths) != len(recognized_texts):
            print(f"警告: ground truth数量({len(ground_truths)})与识别结果数量({len(recognized_texts)})不匹配")
            min_len = min(len(ground_truths), len(recognized_texts))
            ground_truths = ground_truths[:min_len]
            recognized_texts = recognized_texts[:min_len]
        
        # 应用文本规范化
        print("应用文本规范化...")
        normalized_ground_truths = [self.normalizer.normalize(gt) for gt in ground_truths]
        normalized_recognized = [self.normalizer.normalize(rec) for rec in recognized_texts]
        
        # 标准预处理
        if self.language in ["zh", "cn"]:
            transformation = jiwer.Compose([
                jiwer.RemoveWhiteSpace(replace_by_space=False),
                jiwer.RemoveEmptyStrings(),
            ])
        else:
            transformation = jiwer.Compose([
                jiwer.RemovePunctuation(),
                jiwer.ToLowerCase(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ])
        
        gt_processed = transformation(normalized_ground_truths)
        rec_processed = transformation(normalized_recognized)
        
        # 计算WER
        wer_score = jiwer.wer(gt_processed, rec_processed)
        
        # 显示规范化效果
        print(f"\n=== 文本规范化效果示例 ===")
        for i in range(min(3, len(ground_truths))):
            print(f"样本 {i+1}:")
            print(f"  原始GT: {ground_truths[i]}")
            print(f"  规范化GT: {normalized_ground_truths[i]}")
            print(f"  原始识别: {recognized_texts[i]}")
            print(f"  规范化识别: {normalized_recognized[i]}")
            
            # 计算单个样本的WER
            if normalized_ground_truths[i]:
                sample_wer = jiwer.wer(
                    transformation([normalized_ground_truths[i]])[0], 
                    transformation([normalized_recognized[i]])[0]
                )
                print(f"  规范化后WER: {sample_wer:.4f}")
            print("-" * 60)
        
        # 计算每个样本的WER
        individual_wers = []
        for gt, rec in zip(gt_processed, rec_processed):
            if gt:
                individual_wers.append(jiwer.wer(gt, rec))
        
        return {
            "wer": wer_score,
            "num_samples": len(ground_truths),
            "individual_wers": individual_wers,
            "wer_std": np.std(individual_wers) if individual_wers else 0,
            "wer_min": np.min(individual_wers) if individual_wers else 0,
            "wer_max": np.max(individual_wers) if individual_wers else 0,
            "normalized_ground_truths": normalized_ground_truths,
            "normalized_recognized": normalized_recognized,
        }
    
    def evaluate(self, audio_dir, ground_truth_file):
        """执行评估"""
        print("开始TTS音色克隆WER评估")
        print("=" * 60)
        
        # 1. 加载ground truth
        ground_truths = self.load_ground_truth(ground_truth_file)
        print(f"加载 {len(ground_truths)} 条ground truth文本")
        
        # 2. 获取音频文件
        audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        if not audio_files:
            # 尝试其他音频格式
            audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.mp3"))) + \
                         sorted(glob.glob(os.path.join(audio_dir, "*.flac")))
        
        if not audio_files:
            raise ValueError(f"在 {audio_dir} 中未找到音频文件")
        print(f"找到 {len(audio_files)} 个合成音频文件")
        
        # 3. 语音识别
        recognized_texts = self.recognize_audio_batch(audio_files)
        
        # 4. 计算WER
        metrics = self.calculate_wer(ground_truths, recognized_texts)
        
        # 5. 显示结果
        self.print_results(metrics, ground_truths, recognized_texts, audio_files)
        
        return metrics
    
    def print_results(self, metrics, ground_truths, recognized_texts, audio_files):
        """打印详细结果"""
        print("\n" + "=" * 80)
        print("TTS音色克隆WER评估结果")
        print("=" * 80)
        
        print(f"总样本数: {metrics['num_samples']}")
        print(f"词错误率 (WER): {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
        print(f"WER标准差: {metrics['wer_std']:.4f}")
        print(f"最小 WER: {metrics['wer_min']:.4f}")
        print(f"最大 WER: {metrics['wer_max']:.4f}")
        
        # WER分布
        wers = metrics['individual_wers']
        ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]
        range_labels = ["0-10%", "10-20%", "20-30%", "30-50%", "50-100%"]
        
        print("\nWER分布:")
        for (low, high), label in zip(ranges, range_labels):
            count = len([w for w in wers if low <= w < high])
            percentage = (count / len(wers)) * 100 if wers else 0
            print(f"  {label}: {count}个样本 ({percentage:.1f}%)")
        
        # 样本对比
        print("\n样本对比详情:")
        print("=" * 80)
        for i in range(min(5, len(ground_truths))):
            print(f"样本 {i+1}:")
            print(f"  音频文件: {os.path.basename(audio_files[i])}")
            print(f"  原始GT: {ground_truths[i]}")
            print(f"  原始识别: {recognized_texts[i]}")
            print(f"  规范化GT: {metrics['normalized_ground_truths'][i]}")
            print(f"  规范化识别: {metrics['normalized_recognized'][i]}")
            if i < len(wers):
                print(f"  WER: {wers[i]:.4f}")
            print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description='TTS音色克隆WER评估（完整文本规范化版）')
    parser.add_argument("-i", "--input", required=True, help="TTS合成音频目录")
    parser.add_argument("-g", "--ground_truth", required=True, help="Ground truth文本文件")
    parser.add_argument("-l", "--language", choices=['zh', 'en'], default='en', help="语言")
    parser.add_argument("-m", "--model", default="large-v3", help="Whisper模型大小")
    
    args = parser.parse_args()
    
    evaluator = TTSWEREvaluator(model_size=args.model, language=args.language)
    results = evaluator.evaluate(args.input, args.ground_truth)

if __name__ == "__main__":
    main()
    
'''
