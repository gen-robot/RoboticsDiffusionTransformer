# datalist = """
# cobot-bi_pick_coke-cheer20-rdt-1b-ft-prelang-lora0-bs5-demo60-random
# cobot-bi_pick_coke-correct20-rdt-1b-ft-prelang-lora0-bs5-demo60-random
# cobot-bi_pick_coke-correct50-rdt-1b-ft-prelang-lora0-bs5-demo60-random
# cobot-bi_pick_coke-rdt-1b-ft-prelang-lora0-bs5-demo60-expanded
# cobot-bi_pick_coke-rdt-1b-ft-prelang-lora0-bs5-demo60-nonsense
# cobot-cheer_coke-rdt-1b-ft-prelang-lora0-bs5-demo60-random
# cobot-correct_coke-rdt-1b-ft-prelang-lora0-bs5-demo60-random
# cobot-put_book-rdt-1b-ft-prelang-lora0-bs5-demo50-random
# """
ROOT_DIR = "checkpoints"
dirlist = [d.strip() for d in open("checkpoints/ckpt.txt", "r").readlines()]

import os
import time

deploy_dir = f"checkpoints/ckpt-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
os.makedirs(deploy_dir)

for run_dir in dirlist:
    sub_dir = os.listdir(os.path.join(ROOT_DIR, run_dir))
    for dir in sub_dir:
        ckpt_dir = os.path.join(ROOT_DIR, run_dir, dir)
        all_checkpoints = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
        if len(all_checkpoints) > 1:
            sorted_checkpoints = sorted(all_checkpoints, key=lambda x: int(x.split("-")[-1]))
            latest_ckpt = sorted_checkpoints[-1]
            name = run_dir + "-" + latest_ckpt + "-" + dir
            os.makedirs(os.path.join(deploy_dir, name))
            for files in ['config.json', 'model.safetensors', 'adapter_config.json', 'adapter_model.safetensors', 'pytorch_model.bin']:
                if not os.path.exists(os.path.join(ckpt_dir, latest_ckpt, files)):
                    print(f"File {files} not found in {os.path.join(ckpt_dir, latest_ckpt)}")
                    continue
                os.system(f"cp {os.path.join(ckpt_dir, latest_ckpt, files)} {os.path.join(deploy_dir, name)}")
        else:
            print(f"Checkpoint not found in {ckpt_dir}")

os.system(f"tar -czvf {deploy_dir}.tar.gz {deploy_dir}")
