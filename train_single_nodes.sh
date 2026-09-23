
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
export NCCL_DEBUG=ERROR
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0  # Set to 1 for debugging
# single node
#export NCCL_SOCKET_IFNAME=^lo,docker0  # Specify network interface
#export NCCL_IB_DISABLE=1  # Disable InfiniBand if problematic
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
conda init bash && source ~/.bashrc && source activate && conda deactivate && conda activate sai &&
#conda init bash && source ~/.bashrc && conda activate sai
#mount -o remount,size=1024G /dev/shm

check_nvidia_gpu() {
    echo "=== NVIDIA GPU检测 ==="
    
    # 检查nvidia-smi是否可用
    if command -v nvidia-smi &> /dev/null; then
        echo "[√] nvidia-smi 已安装"
        
        # 获取GPU数量
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null)
        
        if [ -z $gpu_count ] || [ $gpu_count -eq 0 ]; then
            echo "[!] nvidia-smi 未检测到GPU"
        else
            echo "[√] 检测到 $gpu_count 个NVIDIA GPU"
            
            # 显示GPU详细信息
            echo "--- GPU详细信息 ---"
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
        fi
    else
        echo "[!] nvidia-smi 未安装"
    fi
    
    echo ""
    echo "=== PCI总线检测 ==="
    
    # 通过PCI总线检测
    pci_count=$(lspci | grep -i nvidia | wc -l)
    echo "PCI总线检测到 $pci_count 个NVIDIA设备"
    
    if [ "$pci_count" -gt 0 ]; then
        echo "--- PCI设备列表 ---"
        lspci | grep -i nvidia
    fi
    
    echo ""
    echo "=== 系统信息 ==="
    
    # 检查NVIDIA驱动
    if [ -f /proc/driver/nvidia/version ]; then
        echo "NVIDIA驱动版本:"
        cat /proc/driver/nvidia/version | head -1
    else
        echo "未检测到NVIDIA驱动"
    fi
    
    # 检查CUDA
    if command -v nvcc &> /dev/null; then
        echo "CUDA版本: $(nvcc --version | grep "release" | awk '{print $6}')"
    fi
}

check_nvidia_gpu

get_rank() {
    # 按优先级检查
    if [ -n "$SLURM_PROCID" ]; then
        echo $SLURM_PROCID
    elif [ -n "$OMPI_COMM_WORLD_RANK" ]; then
        echo $OMPI_COMM_WORLD_RANK
    elif [ -n "$PMI_RANK" ]; then
        echo $PMI_RANK
    elif [ -n "$RANK" ]; then
        echo $RANK
    else
        echo "0"
    fi
}

get_local_rank() {
    if [ -n "$SLURM_LOCALID" ]; then
        echo $SLURM_LOCALID
    elif [ -n "$OMPI_COMM_WORLD_LOCAL_RANK" ]; then
        echo $OMPI_COMM_WORLD_LOCAL_RANK
    elif [ -n "$LOCAL_RANK" ]; then
        echo $LOCAL_RANK
    else
        echo "0"
    fi
}

# 获取信息
RANK=$(get_rank)
LOCAL_RANK=$(get_local_rank)
WORLD_SIZE=${WORLD_SIZE:-1}
LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-1}

PET_NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
NUM_PROCESSES=`expr $WORLD_SIZE \* $PET_NPROC_PER_NODE`

#pip3 list
PROJECT_DIR=/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai
DATA_DIR=/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1
OUTPUT_DIR=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/logs

current_time=$(date '+%Y-%m-%d_%H-%M-%S')
OUTPUT_PATH=$OUTPUT_DIR/${current_time}_sai_1M_cropped_ip_self_lora-embs_bs-$NUM_PROCESSES

# 使用
GRAD_ACC_STEPS=$(echo $(( (32 / $NUM_PROCESSES) * 4 )))

echo "NPROC_PER NODE: ${PET_NPROC_PER_NODE}, WORLD_SIZE: ${WORLD_SIZE}, RANK: ${RANK}, GRAD_ACC_STEPS: ${GRAD_ACC_STEPS}"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "OUTPUT_PATH: $OUTPUT_PATH"

accelerate launch \
  --config_file $PROJECT_DIR/configs/train/accelerate_config_zero0_1node.yaml \
  --num_machines $WORLD_SIZE \
  --num_processes $NUM_PROCESSES \
  --machine_rank $RANK \
  train.py \
  --data_root $DATA_DIR/sample_1M_batches_20_complete \
  --meta_dir $DATA_DIR/sample_1M_batches_20_complete/processed/csv \
  --model_config_path $PROJECT_DIR/configs/train/model.yaml \
  --height 480 \
  --width 864 \
  --num_frames 121 \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps $GRAD_ACC_STEPS \
  --num_epochs 5 \
  --save_steps 2000 \
  --remove_prefix_in_ckpt "pipe.model." \
  --resume_from_ckpt $PROJECT_DIR/logs/sai_1M_contrastive_cropped_gpus_32/ckpt/step-94000.safetensors \
  --output_path $OUTPUT_PATH