import os
import json

import torch
import yaml
import argparse
from tqdm import tqdm

try:
    from ..models.multimodal_encoder.t5_encoder import T5Embedder
    from ..constants import RDT_ROOT_DIR
except:
    from models.multimodal_encoder.t5_encoder import T5Embedder
    from constants import RDT_ROOT_DIR


argparser = argparse.ArgumentParser()
argparser.add_argument("--model_path", type=str, default="google/t5-v1_1-xxl")
argparser.add_argument("--config_path", type=str, default="configs/base.yaml")
argparser.add_argument("--dataset_dir", type=str, 
    default=os.path.join(RDT_ROOT_DIR, "data/datasets/agilex/cobot_data"))
argparser.add_argument("--task_name", type=str, nargs="+", default=None)

args = argparser.parse_args()


GPU = 0
MODEL_PATH = args.model_path
CONFIG_PATH = args.config_path
TARGET_DIR = args.dataset_dir
TASK_LIST = args.task_name
if TASK_LIST is None:
    TASK_LIST = os.listdir(TARGET_DIR)
elif not isinstance(TASK_LIST, list):
    TASK_LIST = [TASK_LIST]

print("%"*30)
print(f"Using model: {MODEL_PATH}")
print(f"Using config: {CONFIG_PATH}")
print(f"Using dataset: {TARGET_DIR}")
print(f"Using task name: {TASK_LIST}")
print("%"*30)

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
    
    # Get all the task paths
    task_paths = []
    for task_name in TASK_LIST:
        task_dir = os.path.join(TARGET_DIR, task_name)
        assert os.path.isdir(task_dir), f"Task directory {task_dir} does not exist."
        if not os.path.exists(os.path.join(
            task_dir, 'expanded_instruction_gpt-4-turbo.json')
        ):
            print(f"[WARNING] Task {task_name} does not contain expanded instructions. Skipped.")
            continue
        task_paths.append(task_dir)

    print(f"Found {len(task_paths)} tasks to encode instructions for.")

    # For each task, encode the instructions
    for task_path in tqdm(task_paths):
        task_name = os.path.basename(task_path)
        # Load the instructions corresponding to the task from the directory
        with open(os.path.join(task_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = [instruction_dict['instruction']] + instruction_dict['simplified_instruction'] + \
            instruction_dict['expanded_instruction']
    
        # Encode the instructions
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

        if not os.path.exists(os.path.join(task_path, "precomp_lang_embeds")):
            os.makedirs(os.path.join(task_path, "precomp_lang_embeds"))

        # Save the embeddings for training use
        for i in range(len(instructions)):
            text_embed = text_embeds[i][attn_mask[i]]
            save_path = os.path.join(task_path, "precomp_lang_embeds", f"lang_embed_{i}.pt")
            # torch.save(text_embed, save_path)
            torch.save({
                "name": task_name,
                "instruction": instructions[i],
                "embeddings": text_embed
            }, save_path)

if __name__ == "__main__":
    main()
