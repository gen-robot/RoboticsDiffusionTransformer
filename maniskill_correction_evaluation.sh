CUDA_VISIBLE_DEVICES=0 python maniskill_correction_collection/eval.py \
    --pretrained_path /nvme0n1/rdt/checkpoints/eval-checkpoints/rdt-maniskill-original.pt \
    --type original
CUDA_VISIBLE_DEVICES=1 python maniskill_correction_collection/eval.py \
    --pretrained_path /nvme0n1/rdt/checkpoints/eval-checkpoints/rdt-maniskill-mix.pt \
    --type mix
CUDA_VISIBLE_DEVICES=3 python maniskill_correction_collection/eval.py \
    --pretrained_path /nvme0n1/rdt/checkpoints/eval-checkpoints/rdt-maniskill-only_correction.pt \
    --correction \
    --type only_correction
CUDA_VISIBLE_DEVICES=4 python maniskill_correction_collection/eval.py \
    --pretrained_path /nvme0n1/rdt/checkpoints/eval-checkpoints/rdt-maniskill-all.pt \
    --correction \
    --type all