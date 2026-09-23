# !/usr/bin/bash

conda activate sai

cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/envs/nv-codec-headers && make install 

cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/envs/ffmpeg && make install

# 设置环境变量
export PATH=$PATH:/usr/local/ffmpeg/bin && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/ffmpeg/lib

mv ~/.bashrc ~/.bashrc_bk

cp /inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/envs/bashrc ~/.bashrc

#export PATH=/inspire/hdd/global_user/zhangkaipeng-24043/maomaoli/anaconda3/bin:$PATH
conda init bash && source ~/.bashrc && source activate && conda deactivate && conda activate sai

cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai

