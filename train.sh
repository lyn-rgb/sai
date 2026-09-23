
# export NCCL_NET=IB
# export NCCL_NET=Socket; # 数据传输协议，如果使用IB网卡协议，则不需要配置
# export NCCL_SOCKET_IFNAME=en,eth,em,bond;  # 指定的socket协议网口，默认是eth0
# export NCCL_SHM_DISABLE=1;  # 强制使用P2P协议，会自动使用IB协议或IP socket
# export NCCL_SOCKET_NTHREADS=4;  # socket协议线程数，默认是1,范围1-16，数字越大数据传输越快
# export NCCL_P2P_DISABLE=0;  # 关闭p2p传输，使用NVLink or PCI可配置，默认可不配置
# export NCCL_IB_DISABLE=1;  # 为1表示禁用IB协议，如果使用IB则设置为0
# export NCCL_DEBUG=INFO # DEBUG打印日志的等级
export ENABLE_COMPILE=false
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_NCCL_ENABLE_MONITORING=0
# Add these to your training script
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0  # Set to 1 for debugging
# single node
export NCCL_SOCKET_IFNAME=^lo,docker0  # Specify network interface
export NCCL_IB_DISABLE=1  # Disable InfiniBand if problematic
# multi nodes
#export NCCL_CROSS_NIC=1
#export NCCL_IB_GID_INDEX=3
#export NCCL_IB_TIMEOUT=22
#export NCCL_SOCKET_IFNAME=storage_bond # multi-nodes
#export GLOO_SOCKET_IFNAME=storage_bond
#export NCCL_NET_PLUGIN=none

CONDA_HOME=/inspire/hdd/global_user/zhangkaipeng-24043/maomaoli/anaconda3/bin
export PATH=$CONDA_HOME:$PATH
#source ~/.bashrc
conda init bash && source ~/.bashrc && source activate && conda deactivate && conda activate sai
#conda init bash && source ~/.bashrc && conda activate sai

NUM_PROCESSES=`expr $WORLD_SIZE \* $PET_NPROC_PER_NODE`
echo "NPROC_PER NODE: ${PET_NPROC_PER_NODE}, WORLD_SIZE: ${WORLD_SIZE}, RANK: ${RANK}"
#pip3 list
PROJECT_DIR=/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai
DATA_DIR=/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1

accelerate launch \
  --config_file $PROJECT_DIR/configs/train/accelerate_config_zero1.yaml \
  --machine_rank 0 \
  --main_process_port 29500 \
  train.py \
  --data_root $DATA_DIR/sample_1M_batches_20_complete \
  --meta_dir $DATA_DIR/CloneMyFaceCloneMyVoice/sai/data/meta/sample_1M_batches_20_complete \
  --model_config_path $PROJECT_DIR/configs/train/model.yaml \
  --height 480 \
  --width 864 \
  --num_frames 81 \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps 4 \
  --num_epochs 5 \
  --save_steps 2000 \
  --resume_from_ckpt "$PROJECT_DIR/logs/sai_1M_baseline_gpus_32/ckpt/step-40000.safetensors" \
  --remove_prefix_in_ckpt "pipe.model." \
  --output_path "$PROJECT_DIR/logs/sai_1M_contrastive_cropped_debug" \