import time
import os
import sys
import json
import math
import random
from tqdm import tqdm
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import transformers

language_precompute_dir = '/nvme_data/liangzhi/rdt/gpt_precompute_prompt/'

if __name__ == "__main__":
    task_list = os.listdir(language_precompute_dir)

    for task in task_list:
        task_dir = os.path.join(language_precompute_dir, f"{task}/general")
        num_language_ins = len(os.listdir(task_dir))

        for i in tqdm(range(num_language_ins), desc=f"Checking {task}:"):
            ins_emb_dir = os.path.join(task_dir, 'lang_embed_{}.pt'.format(i))
            if ins_emb_dir[-1] == ".":
                ins_emb_dir = ins_emb_dir[:-1]
            
            if isinstance(torch.load(ins_emb_dir), torch.Tensor):
                pass
            else:
                print(f"Error in {ins_emb_dir}")
                break