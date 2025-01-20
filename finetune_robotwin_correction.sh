export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
# set NCCL_SOCKET_IFNAME to the interface you want to use for NCCL communication, get from ifconfig. otherwise will encounter error("NCCL WARN Bootstrap : no socket interface found")
export NCCL_SOCKET_IFNAME=enp185s0f1np1 #enp210s0f0 #bond0 
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

now="$(date +"%Y%m%d-%H%M%S")"
lr=$1
type=$2

run_name="rdt-robotwin-finetune-lr${lr}-type${type}"
save_path="/nvme_data/liangzhi/rdt"

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="${save_path}/checkpoints/${run_name}-${now}"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/nvme_data/liangzhi/installer/flash-attention/csrc/cutlass/"

# assert CUTLASS_PATH is set
if [ -z "$CUTLASS_PATH" ]; then
    echo "CUTLASS_PATH is not set, clone git@github.com:NVIDIA/cutlass.git and export CUTLASS_PATH=/path/to/cutlass"
    exit 1
fi

if [ -z "$NCCL_SOCKET_IFNAME" ]; then
    echo "NCCL_SOCKET_IFNAME is not set, check ifconfig for the interface name"
    exit 1
fi

export WANDB_PROJECT="robotics_diffusion_transformer"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

accelerate launch main.py \
    --deepspeed="./configs/zero2.json" \
    --robot_name="cobot" \
    --run_name="rdt-robotwin-${task_name}-lr${lr}-type${type}-multi_task" \
    --pretrained_model_name_or_path="google/rdt-robotwin/" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=6 \
    --sample_batch_size=8 \
    --max_train_steps=200000 \
    --checkpointing_period=5000 \
    --sample_period=500 \
    --checkpoints_total_limit=5 \
    --lr_scheduler="constant" \
    --learning_rate=${lr} \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --robotwin \
    --data_type=${type} \
    --report_to=wandb
