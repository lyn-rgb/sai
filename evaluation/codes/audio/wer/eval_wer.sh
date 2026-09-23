# . /inspire/hdd/global_user/zhangkaipeng-24043/maomaoli/anaconda3/etc/profile.d/conda.sh

export TORCH_HUB_USE_FORK_CHECK=0
# conda activate tts-eval
python bench_verification.py batch_verification \
--model_name wavlm_large \
--audio1_dir /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/benchmark1/processed/audio \
--audio2_dir /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/TTS/F5-TTS/output_bench1 \
--checkpoint ./ckpt/wavlm_large_finetune.pth \
--use_gpu True \
--output_file ./result/audio-sim_f5_tts.txt


#./result/audio-sim_ours_resume_9.2w_oldbest_560*992_celeb_ip-wo-audio-wo-audio.txt
#--audio2_dir /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/TTS/fish-speech/output-bench1 \
#--audio2_dir /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/TTS/F5-TTS/output_bench1 \
#--audio2_dir /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/TTS/CosyVoice/output_bench1 \
#--audio2_dir /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/outputs/contrastive_cropped/new_infer_bench/ip_image_True_ip_audio_True \






# conda activate cosyvoice1

# python wer.py \
#     --gt_dir /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/benchmark1/processed/gt_se \
#     --audio_dir /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/TTS/CosyVoice/output_bench1  \
#     --model_path large-v3 \
#     --lang en


# python verification.py \
# --model_name wavlm_large \
# --wav1 vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav \
# --wav2 vox1_data/Josh_Gad/HXUqYaOwrxA_0000015.wav \
# --checkpoint ./ckpt/wavlm_large_finetune.pth




###single
python verification.py \
    --model_name wavlm_large \
    --wav1 "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/benchmark1/processed/audio/00007_audio.wav" \
    --wav2 "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/outputs/resume_9.8w_oldbest_560*992_multiseed/07/w33/audio/07-04_audio.wav" \
    --use_gpu True \
    --checkpoint ./ckpt/wavlm_large_finetune.pth


#/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/outputs/resume_9.8w_oldbest_560*992_multiseed/10/ip_image_True_ip_audio_True/00074_crop-True_A_woman_stands_before_the_Trevi_Fountain_at_dawn,__560x992_seed1273_0.mp4
#/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/outputs/resume_5.2w_oldbest_558*992/w33/audio/00068_crop-True_A_man_stands_at_the_podium_in_OpenAI's_luxurious_c_558x992_102_0_audio.wav
#/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/outputs/sai_1M_cropped_ip_embs_gpus_8_online_asr_sa_lora_13.4w_5s_up_res_Highres/w33/audio/00068_crop-True_A_man_stands_at_the_podium_in_OpenAI's_luxurious_c_720x1280_102_0_audio.wav