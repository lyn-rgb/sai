# python demo_syncnet.py \
# --videofile /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/ConsisID/eval/new_metric/Wav2Lip/evaluation/syncnet_python/data/example.avi \
# --tmp_dir /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/ConsisID/eval/new_metric/syncnet_python/tmp 

#/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/outputs/resume_9.2w_oldbest_560*992/ip_image_True_ip_audio_True
#/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/HuMo/out_one_file
#/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/HunyuanCustom/out_one_file




python batch_syncnet.py \
  --video_dir /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/HunyuanCustom/out_one_file \
  --output_file results.json \
  --tmp_base_dir ./tmp