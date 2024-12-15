datalist = """
cobot-bi_pick_coke-rdt1bft-prelang-lora0-bs5-demo20-20241214-011931
cobot-bi_pick_coke-rdt1bft-prelang-lora0-bs5-demo40-20241214-011927
cobot-bi_pick_coke-rdt1bft-prelang-lora0-bs5-demo60-20241214-011925
cobot-bi_pick_coke-rdt1bft-prelang-lora32-bs5-demo60-20241214-011933
"""
ROOT_DIR = "checkpoints"

data_dir = datalist.strip().split("\n")
print(data_dir)

import os
import time

deploy_dir = f"checkpoints/ckpt-{time.time()}"
os.makedirs(deploy_dir)

for dir in data_dir:
    # if True:
    ckpt_dir = os.path.join(ROOT_DIR, dir)
    # ckpt_dir = "/nvme0n1/gaofeng/rdt_data-finetune-1b-20241204-031804" #os.path.join(root_dir, dir)
    all_checkpoints = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
    if len(all_checkpoints) > 1:
        sorted_checkpoints = sorted(all_checkpoints, key=lambda x: int(x.split("-")[-1]))
        # for ckpt in sorted_checkpoints[:-kept_ckpt_num]:
        #     ckpt_path = os.path.join(ckpt_dir, ckpt)
        #     print(f"Removing {ckpt_path}")
        #     os.system(f"rm -rf {ckpt_path}")
        latest_ckpt = sorted_checkpoints[-1]
        name = ckpt_dir.split("/")[-1] + "-" + latest_ckpt
        os.makedirs(os.path.join(deploy_dir, name))
        for files in ['config.json', 'model.safetensors', 'adapter_config.json', 'adapter_model.safetensors', 'pytorch_model.bin']:
            if not os.path.exists(os.path.join(ckpt_dir, latest_ckpt, files)):
                print(f"File {files} not found in {os.path.join(ckpt_dir, latest_ckpt)}")
                continue
            os.system(f"cp {os.path.join(ckpt_dir, latest_ckpt, files)} {os.path.join(deploy_dir, name)}")
    else:
        print(f"Checkpoint not found in {ckpt_dir}")

os.system(f"tar -czvf {deploy_dir}.tar.gz {deploy_dir}")
