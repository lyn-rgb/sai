""" 使用whisper(EN)或paraformer-zh(ZH)测试错字率
"""
import os
import math
import string
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import numpy as np
import json

from jiwer import wer as compute_wer
from zhon.hanzi import punctuation
import zhconv


def load_asr_model(lang, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel

        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            # vad_model = os.path.join(ckpt_dir, "fsmn-vad"),
            # punc_model = os.path.join(ckpt_dir, "ct-punc"),
            # spk_model = os.path.join(ckpt_dir, "cam++"),
            disable_update=True,
        )  # following seed-tts setting
    elif lang == "en":
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model
    

def run_asr_wer(asr_model, audio_path, gt_text, lang="en"):
    punctuation_all = punctuation + string.punctuation
    
    
    if lang == "zh":
        res = asr_model.generate(input=audio_path, batch_size_s=300, disable_pbar=True)
        hypo = res[0]["text"]
        hypo = zhconv.convert(hypo, "zh-cn")
    elif lang == "en":
        segments, _ = asr_model.transcribe(audio_path, beam_size=5, language="en")
        hypo = ""
        for segment in segments:
            hypo = hypo + " " + segment.text

    raw_truth = gt_text
    raw_hypo = hypo

    for x in punctuation_all:
        gt_text = gt_text.replace(x, "")
        hypo = hypo.replace(x, "")
    
    gt_text = gt_text.replace("  ", " ")
    hypo = hypo.replace("  ", " ")
    
    gt_text = gt_text.replace("—", " ").replace("-", " ")
    hypo = hypo.replace("—", " ").replace("-", " ")

    if lang == "zh":
        gt_text = " ".join([x for x in gt_text])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        gt_text = gt_text.lower()
        hypo = hypo.lower()

    wer = compute_wer(gt_text, hypo)
    
    return {
            "wav": Path(audio_path).stem,
            "truth": raw_truth,
            "hypo": raw_hypo,
            "wer": wer,
        }
    

def compute_wsr(audio_text_lst, ckpt_dir, lang="en", eval_task="wer", save_dir=None):
    # get model
    asr_model = load_asr_model(lang, ckpt_dir)
    print(f"ASR Model for langurage {lang} has been loaded.")
    wer_results = []
    
    for audio_text in tqdm(audio_text_lst, total=len(audio_text_lst), desc="Compute ASR"):
        audio_path = audio_text["audio_path"]
        gt_text = audio_text["gt_text"]
        
        wer_results.append(run_asr_wer(asr_model, audio_path, gt_text, lang))
        
    metrics = []
    if save_dir is not None:
        result_path = f"{save_dir}/_{eval_task}_results.jsonl"
        with open(result_path, "w") as f:
            for line in wer_results:
                metrics.append(line[eval_task])
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            metric = round(np.mean(metrics), 5)
            f.write(f"\n{eval_task.upper()}: {metric}\n")

    print(f"\nTotal {len(metrics)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"{eval_task.upper()} results saved to {result_path}")

        
if __name__ == "__main__":
    text_a = "this is a audio and video sync generator"
    text_b = "that are a audio & video sync generator"
    print(f"wer: {compute_wer(text_a, text_b)}")
    
    audio_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/outputs/sai_1M_cropped_ip_embs_gpus_8_online_asr_sa_lora_16.4w_5s_up_res_576*992/w33/audio"
    gt_text_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/benchmark1/processed/gt_se"
    ckpt_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/weights/faster-whisper-large-v3"
    save_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/results/Ours_emb"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # load audio text pairs
    text_file_names = sorted([text_file_name for text_file_name in os.listdir(gt_text_dir) if text_file_name.endswith(".txt")])
    audio_file_names = sorted([audio_file_name for audio_file_name in os.listdir(audio_dir) if audio_file_name.endswith(".wav")])
    
    audio_text_lst = []
    for text_file_name in text_file_names:
        text_file_path = os.path.join(gt_text_dir, text_file_name)
        with open(text_file_path, "r") as f:
            gt_text = f.readline().strip()
        
        file_index = text_file_name.split("_")[0]
        for audio_file_name in audio_file_names:
            if file_index in audio_file_name:
                audio_path = os.path.join(audio_dir, audio_file_name)
                audio_text_lst.append({
                    "audio_path": audio_path,
                    "gt_text": gt_text.replace("—", " ").replace("-", " ")
                })
                break
    print(f"Total have {len(audio_text_lst)} audio-text pairs")
    compute_wsr(audio_text_lst=audio_text_lst, ckpt_dir=ckpt_dir, lang="en", eval_task="wer", save_dir=save_dir)