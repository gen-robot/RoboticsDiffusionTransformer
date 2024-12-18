export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
# set NCCL_SOCKET_IFNAME to the interface you want to use for NCCL communication, get from ifconfig. otherwise will encounter error("NCCL WARN Bootstrap : no socket interface found")
# export NCCL_SOCKET_IFNAME=enp210s0f0 #bond0 
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

now="$(date +"%Y-%m-%d-%H-%M-%S")"

# check if arguments are exported, otherwise use default values
if [ -z "$lora_rank" ]; then
    lora_rank=0
fi
if [ -z "$bs" ]; then
    bs=5
fi
if [ -z "$task" ]; then
    task=""
    echo "[Warning] Task not specified, will train on all tasks, continue? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
        exit 1
    fi
fi
if [ -z "$max_demo" ]; then
    max_demo=200
fi
if [ -z "$pretrained" ]; then
    pretrained='rdt-1b-ft'
fi
if [ -z "$instr" ]; then
    instr='random'
fi
if [ -z "$mask_prob" ]; then
    mask_prob=0.1
fi
if [ -z "$precision" ]; then
    precision='bf16'
fi

# print the arguments
echo "%%%%%% Arguments %%%%%%"
echo "lora_rank: ${lora_rank}"
echo "bs: ${bs}"
echo "task: ${task}"
echo "max_demo: ${max_demo}"
echo "pretrained: ${pretrained}"
echo "instr: ${instr}"
echo "mask_prob: ${mask_prob}"
echo "precision: ${precision}"
echo "%%%%%%%%%%%%%%%%%%%%%%%"

run_name="cobot-${task}-${pretrained}-lora${lora_rank}-bs${bs}-max${max_demo}-${instr}-mask${mask_prob}-${precision}"
ckpt_path="google/${pretrained}"

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/${run_name}/${now}"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"

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
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

# deepspeed --hostfile=hostfile.txt

accelerate launch main.py \
    --deepspeed="./configs/zero2.json" \
    --robot_name="cobot" \
    --precomp_lang_embed \
    --lora_rank=${lora_rank} \
    --run_name=${run_name} \
    --data_path="data/datasets/agilex/cobot_data/${task}" \
    --pretrained_model_name_or_path=${ckpt_path} \
    --max_demo_per_task=${max_demo} \
    --instruction_mode=${instr} \
    --cond_mask_prob=${mask_prob} \
    --mixed_precision=${precision} \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=${bs} \
    --sample_batch_size=${bs} \
    --max_train_steps=200000 \
    --checkpointing_period=1000 \
    --sample_period=500 \
    --checkpoints_total_limit=10 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=wandb

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
    # --precomp_lang_embed \
