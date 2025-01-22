import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
sys.path.append("/home/liangzhi/work-space/RoboTwin/")

from scripts.robotwin_model import create_model, RoboticDiffusionTransformerModel
from scripts.robotwin_adapter import RDTRobotwinAdapter
from argparse import ArgumentParser
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image as PImage

from robotwin_sim_test.eval_rdt_model import RDT

import matplotlib.pyplot as plt

import pickle
from configs.state_vec import STATE_VEC_IDX_MAPPING

UNI_STATE_INDICES = [50, 51, 52, 53, 54, 55, 60, 0, 1, 2, 3, 4, 5, 10]

AGILEX_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["left_gripper_open"]
] + [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
]

def get_action(data_dict, agent, text_embeds, usr_args):
    proprio = torch.from_numpy(data_dict["state"][:, UNI_STATE_INDICES]).to(device=agent.policy.device)
    images = [data_dict["cam_high"][0], data_dict["cam_left_wrist"][0], data_dict["cam_right_wrist"][0], 
              data_dict["cam_high"][1], data_dict["cam_left_wrist"][1], data_dict["cam_right_wrist"][1]]
    images = [PImage.fromarray(img) if img is not None else None for img in images]
    action_chunk = agent.policy.step(proprio, images, text_embeds)
    if len(action_chunk.shape) > 2:
        action_chunk = action_chunk.squeeze(0)
    np_action_chunk = action_chunk.detach().cpu().numpy()
    return np_action_chunk

def main(usr_args):
    rdt_args = usr_args
    agent = RDT(rdt_args, device=usr_args.device)

    expert_data_set = pickle.load(open("./robotwin_sim_test/outs/ex_check_data_dict.pkl", "rb"))
    check_data_set = pickle.load(open("./robotwin_sim_test/outs/check_data_dict.pkl", "rb"))

    test_ids = expert_data_set.keys()

    for id in test_ids:
        ex_data = expert_data_set[id][0]
        ch_data = check_data_set[id][0]

        text_embeds = torch.load(ch_data["meta"]["instruction"]).to(usr_args.device).unsqueeze(0)

        gt_action = ex_data['actions'][:, AGILEX_STATE_INDICES]
        ex_state_action = get_action(ex_data, agent, text_embeds, usr_args)
        ch_state_action = get_action(ch_data, agent, text_embeds, usr_args)

        plot_files = f'./robotwin_sim_test/outs/expert/{id}_plot.png'
        sub_img = plt.figure(figsize=(16, 3.5))
        for i in range(gt_action.shape[-1]):
            plt.subplot(2, 7, i+1)
            plt.plot(gt_action[:, i], label='gt', linestyle='dashed')
            plt.plot(ex_state_action[:, i], label='original')
            plt.plot(ch_state_action[:, i], label='process')
            plt.title(f'joint {i}')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_files)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--task_name', type=str, default='block_hammer_beat')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', type=bool, default=False, help='Render the environment')
    parser.add_argument('--render_freq', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=30)

    parser.add_argument('--chunk_size', action='store', type=int, help='Action chunk size', default=64)
    parser.add_argument('--img_history_size', action='store', type=int, help='Image history size', default=2)
    
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    
    parser.add_argument('--deploy_chunk_size', type=int, default=16, help='Action chunk size for deployment')
    parser.add_argument('--correction', action='store_true', help='Correction flag')
    
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=False, 
                        default='/nvme_data/liangzhi/rdt/checkpoints/rdt-robotwin-finetune-lr1e-4-typeoriginal-20250120-150814/checkpoint-5000/', 
                        help='rdt model check points dir')
    parser.add_argument('--ctrl_freq', type=int, default=15, help='Control frequency')
    parser.add_argument('--eval_episodes', type=int, default=100, help='Number of episodes to evaluate')
    parser.add_argument('--general_ins', type=bool, default=True, help='General instruction flag')
    parser.add_argument('--language_embeding_name', type=str, default=None, help='Language embedding name')
    
    usr_args = parser.parse_args()

    main(usr_args)