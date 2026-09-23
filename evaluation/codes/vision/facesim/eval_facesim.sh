#conda activate sai

CUDA_VISIBLE_DEVICES=0 python eval_batch_facesim_fid.py \
    --video_root /inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai_code/outputs/ablations/text_only_ovi/ip_image_False_ip_audio_False \
    --image_root  /inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/benchmark1/processed/frames \
    --device cuda \
    --results_csv result_ab/face_sim/eval_id_vace2.csv \


#result_new/face_sim/eval_ours_resume_15.0w_oldbest_560*992_name.csv
#/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/Phantom/out_benchmark1
#/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/VACE/out_benchmark1
#/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/ConsisID/output_benchmark
#/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/HunyuanCustom/out_benchmark1/ 
#/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/ID-Animator/ID-Animator/output_benchmark1_720_16_70example
#/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/HuMo/output