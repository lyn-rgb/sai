
# export NCCL_NET=IB
# export NCCL_NET=Socket; # 数据传输协议，如果使用IB网卡协议，则不需要配置
# export NCCL_SOCKET_IFNAME=en,eth,em,bond;  # 指定的socket协议网口，默认是eth0
# export NCCL_SHM_DISABLE=1;  # 强制使用P2P协议，会自动使用IB协议或IP socket
# export NCCL_SOCKET_NTHREADS=4;  # socket协议线程数，默认是1,范围1-16，数字越大数据传输越快
# export NCCL_P2P_DISABLE=0;  # 关闭p2p传输，使用NVLink or PCI可配置，默认可不配置
# export NCCL_IB_DISABLE=1;  # 为1表示禁用IB协议，如果使用IB则设置为0
# export NCCL_DEBUG=INFO # DEBUG打印日志的等级
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_NCCL_ENABLE_MONITORING=0
# Add these to your training script
export NCCL_DEBUG=ERROR
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0  # Set to 1 for debugging

# multi nodes
export NCCL_CROSS_NIC=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
#export NCCL_SOCKET_IFNAME=en,eth,em,bond # multi-nodes, 平台内置了
#export GLOO_SOCKET_IFNAME=en,eth,em,bond
export NCCL_NET_PLUGIN=none
export ENABLE_COMPILE=false
export TOKENIZERS_PARALLELISM=false

export PATH="/inspire/hdd/global_user/zhangkaipeng-24043/maomaoli/anaconda3/bin:$PATH"

conda init bash && source ~/.bashrc && source activate && conda deactivate && conda activate sai &&

current_time=$(date '+%Y-%m-%d')

# 默认项目和数据目录
PROJECT_DIR=/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai
DATA_DIR=/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1
OUTPUT_DIR=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/logs

# 系统设置
NUM_PROCESSES=`expr $WORLD_SIZE \* $PET_NPROC_PER_NODE`
OUTPUT_PATH=$OUTPUT_DIR/baseline/${current_time}_sai_1M_cropped_ip-embs_self-lora_bs-$NUM_PROCESSES-base_960x960
GRAD_ACC_STEPS=$(echo $(( (32 / $NUM_PROCESSES) * 4 )))

# mount -o remount,size=1024G /dev/shm
echo "=== 系统信息 ==="
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}, GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME}"
echo "WORLD_SIZE: ${WORLD_SIZE}, RANK: ${RANK}"
echo "PET_NNODES: ${PET_NNODES}, PET_NODE_RANK: ${PET_NODE_RANK}, NPROC_PER NODE: ${PET_NPROC_PER_NODE}, NUM_PROCESSES: ${NUM_PROCESSES}"
echo "MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"
echo "PET_MASTER_ADDR: ${PET_MASTER_ADDR}, PET_MASTER_PORT: ${PET_MASTER_PORT}"
echo "GRAD_ACC_STEPS: ${GRAD_ACC_STEPS}"
echo "=== 文件目录 ==="
echo "PROJECT_DIR: $PROJECT_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "=== 开始训练 ==="

accelerate launch \
  --config_file $PROJECT_DIR/configs/train/accelerate_config_zero0_multinodes.yaml \
  --num_machines $WORLD_SIZE \
  --num_processes $NUM_PROCESSES \
  --machine_rank $RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  train.py \
  --data_root $DATA_DIR/sample_1M_batches_20_complete \
  --meta_dir $DATA_DIR/sample_1M_batches_20_complete/processed/csv \
  --model_config_path $PROJECT_DIR/configs/train/model_960x960.yaml \
  --height 480 \
  --width 864 \
  --num_frames 121 \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps $GRAD_ACC_STEPS \
  --num_epochs 5 \
  --save_steps 2000 \
  --remove_prefix_in_ckpt "pipe.model." \
  --resume_from_ckpt /inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/logs/2026-01-06_15-27-44_sai_1M_cropped_ip_self_lora-embs_bs-8/ckpt/step-174000.safetensors \
  --output_path $OUTPUT_PATH