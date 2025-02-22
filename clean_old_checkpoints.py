datalist = """
cobot-all_about_coke-rdt1bft-prelang-lora0-bs10-20241212-230308
cobot-bi_pick_coke-rdt1bft-prelang-lora0-bs5-demo20-20241214-011931
cobot-bi_pick_coke-rdt1bft-prelang-lora0-bs5-demo40-20241214-011927
cobot-bi_pick_coke-rdt1bft-prelang-lora0-bs5-demo60-20241214-011925
cobot-bi_pick_coke-rdt1bft-prelang-lora32-bs5-demo60-20241214-011933
cobot-cheer_coke-rdt1bft-prelang-lora0-bs10-20241212-230235
cobot-coke-rdt1b-prelang-lora0-bs10-20241208-204308
cobot-coke-rdt1b-prelang-lora16-bs10-20241208-204255
cobot-coke-rdt1b-prelang-lora32-bs10-20241208-204344
cobot-coke-rdt1b-prelang-lora32-bs20-20241208-204357
cobot-coke-rdt1b-prelang-lora64-bs10-20241208-204350
cobot-correct_coke-rdt1bft-prelang-lora0-bs10-20241212-230110
cobot-move_coke-rdt1bft-prelang-lora0-bs10-20241212-230553
cobot-move_coke-rdt1bft-prelang-lora0-bs10-20241213-145157
cobot-move_coke-rdt1bft-prelang-lora0-bs10-20241213-145857
rdt-170m-pretrained
rdt-1b-ft
rdt-1b-pretrained
rdt-finetune-170m
rdt-finetune-170m-20241204-030003
rdt-finetune-1b
rdt-finetune-1b-20241204-025959
rdt-finetune-1b-20241205-014854
rdt_data-finetune-1b-20241204-031804
shanzhao-rdt-finetune-1b-20241205-013307
"""
ROOT_DIR = "checkpoints"

data_dir = datalist.strip().split("\n")
print(data_dir)

kept_ckpt_num = 5

import os

for dir in data_dir:
    # if True:
    ckpt_dir = os.path.join(ROOT_DIR, dir)
    # ckpt_dir = "/nvme0n1/gaofeng/rdt_data-finetune-1b-20241204-031804" #os.path.join(root_dir, dir)
    all_checkpoints = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
    if len(all_checkpoints) > 1:
        sorted_checkpoints = sorted(all_checkpoints, key=lambda x: int(x.split("-")[-1]))
        for ckpt in sorted_checkpoints[:-kept_ckpt_num]:
            ckpt_path = os.path.join(ckpt_dir, ckpt)
            print(f"Removing {ckpt_path}")
            os.system(f"rm -rf {ckpt_path}")