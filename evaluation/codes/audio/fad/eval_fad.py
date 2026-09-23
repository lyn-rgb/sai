from frechet_audio_distance import FrechetAudioDistance
'''
# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)
# # to use `PANN`
# frechet = FrechetAudioDistance(
#     model_name="pann",
#     sample_rate=16000,
#     use_pca=False, 
#     use_activation=False,
#     verbose=False
# )
# # to use `CLAP`
# frechet = FrechetAudioDistance(
#     model_name="clap",
#     sample_rate=48000,
#     submodel_name="630k-audioset",  # for CLAP only
#     verbose=False,
#     enable_fusion=False,            # for CLAP only
# )
# # to use `EnCodec`
# frechet = FrechetAudioDistance(
#     model_name="encodec",
#     sample_rate=48000,
#     channels=2,
#     verbose=False,
# )

fad_score = frechet.score(
    "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/benchmark1/processed/audio", 
    "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/outputs/resume_9.2w_oldbest_560*992/w33/audio", 
    dtype="float32"
)

'''


import os
import traceback
from frechet_audio_distance import FrechetAudioDistance

# 设置环境变量（确保使用本地缓存）
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/ConsisID/eval/new_metric/frechet-audio-distance/facebook/"

print("="*50)
print("初始化 FrechetAudioDistance...")
print("="*50)

try:
    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False, 
        use_activation=False,
        verbose=True  # 设置为 True 可以看到详细日志
    )
    print("✓ 初始化成功")
except Exception as e:
    print(f"✗ 初始化失败: {e}")
    traceback.print_exc()
    exit(1)

print("\n" + "="*50)
print("开始计算 FAD 分数...")
print("="*50)

background_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/benchmark1/processed/audio"

eval_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/outputs/resume_10.8w_oldbest_560*992/w33/audio"

#eval_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/TTS/CosyVoice/output_bench1-cosy1-sft-complete"

#eval_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/TTS/F5-TTS/output_bench1"

# eval_dir = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/TTS/fish-speech/output-bench1"




print(f"背景音频目录: {background_dir}")
print(f"评估音频目录: {eval_dir}")
print(f"目录存在 - 背景: {os.path.exists(background_dir)}")
print(f"目录存在 - 评估: {os.path.exists(eval_dir)}")
print(f"背景目录文件数: {len(os.listdir(background_dir)) if os.path.exists(background_dir) else 0}")
print(f"评估目录文件数: {len(os.listdir(eval_dir)) if os.path.exists(eval_dir) else 0}")

try:
    fad_score = frechet.score(
        background_dir, 
        eval_dir, 
        dtype="float32"
    )
    print("\n" + "="*50)
    print(f"FAD 分数: {fad_score}")
    print("="*50)
except Exception as e:
    print(f"\n✗ 计算 FAD 时出错: {e}")
    traceback.print_exc()