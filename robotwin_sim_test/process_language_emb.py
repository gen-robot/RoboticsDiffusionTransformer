import os
import json
import sys

import torch
import yaml
from tqdm import tqdm

if __name__=="__main__":
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_path)

from models.multimodal_encoder.t5_encoder import T5Embedder
import pickle

GPU = 0
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
# Modify the TARGET_DIR to your dataset path
TARGET_DIR = "/nvme_data/liangzhi/rdt/gpt_precompute_prompt/"
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    tasks = [
        'block_hammer_beat',
        'blocks_stack_easy_part_repick_black',
        'empty_cup_place_end',
        'block_handover',
        'blocks_stack_easy_part_repick_red',
        'empty_cup_place_full',
        'blocks_stack_easy_end',
        'container_place',
        'empty_cup_place',
        'blocks_stack_easy_full',
        'diverse_bottles_pick',
        'empty_cup_place_part_repick',
        'blocks_stack_easy',
        'dual_bottles_pick_hard',
        'shoe_place'
    ]

    for task in tasks:
        task_emb_dir = os.path.join(TARGET_DIR, task)
        if not os.path.exists(task_emb_dir):
            os.makedirs(task_emb_dir)

        key = "general"

        task_detail_emb_dir = os.path.join(task_emb_dir, key)
        if not os.path.exists(task_detail_emb_dir):
            os.makedirs(task_detail_emb_dir)

        language_ins_file = '/nvme_data/embodied_agent/robotwin_data/{}_hdf5/language_ins.pkl'.format(task)
        with open(language_ins_file, 'rb') as f:
            data = pickle.load(f)
            instructions = data[key]

            tokenized_res = tokenizer(
                instructions, return_tensors="pt",
                padding="longest",
                truncation=True
                )
            tokens = tokenized_res["input_ids"].to(device)
            attn_mask = tokenized_res["attention_mask"].to(device)

            with torch.no_grad():
                text_embeds = text_encoder(
                    input_ids=tokens,
                    attention_mask=attn_mask
                )["last_hidden_state"].detach().cpu()
        
            attn_mask = attn_mask.cpu().bool()

            for i in range(len(instructions)):
                text_embed = text_embeds[i][attn_mask[i]]
                save_path = os.path.join(task_detail_emb_dir, 'lang_embed_{}.pt'.format(i))
                torch.save(text_embed, save_path)


if __name__ == "__main__":
    main()
