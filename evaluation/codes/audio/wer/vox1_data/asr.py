# import whisper

# # 加载模型
# model = whisper.load_model("medium")

# # 转录音频文件
# result = model.transcribe("mm1/Lab/TTS/evaluation/UniSpeech/downstreams/speaker_verification/vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav", language="en")

# # 打印识别结果
# print(result["text"])


import os
import whisper

# 检查文件是否存在
audio_path = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/TTS/evaluation/UniSpeech/downstreams/speaker_verification/vox1_data/David_Faustino/xTOk1Jz-F_g_0000015.wav"

if not os.path.exists(audio_path):
    print(f"错误：文件不存在 - {audio_path}")
    print(f"当前工作目录：{os.getcwd()}")
else:
    print(f"文件找到，开始转录...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="en")
    print(result["text"])