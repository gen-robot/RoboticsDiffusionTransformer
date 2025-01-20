from scripts.robotwin_model import create_model, RoboticDiffusionTransformerModel
from scripts.robotwin_adapter import RDTRobotwinAdapter
import importlib
import sys
import traceback
import yaml
from argparse import ArgumentParser
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
import os
sys.path.append('/home/liangzhi/work-space/RoboTwin/')

class RDT:
    def __init__(self, rdt_args, device='cuda:0'):
        pretrained_vision_encoder_name_or_path = "/home/liangzhi/work-space/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384"
        self.textembeding = None
        self.deploy_chunk_size = rdt_args.deploy_chunk_size
        self.device = device

        with open(rdt_args.config_path, "r") as fp:
            config = yaml.safe_load(fp)
        
        self.policy = create_model(args=config,
                                   device=device,
                                   dtype=torch.bfloat16,
                                   pretrained=rdt_args.pretrained_model_name_or_path,
                                   pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
                                   control_frequency=rdt_args.ctrl_freq,
                                   )
        self.runner = RDTRobotwinAdapter(rdt_args, task_name=rdt_args.task_name, eval_episodes=rdt_args.eval_episodes,
                                         max_steps=400)
        
    def update_textembeding(self, textembeding_dir):
        self.textembeding = torch.load(textembeding_dir).to(self.device).unsqueeze(0)

    def update_obs(self, observation):
        self.runner.update_obs(observation)    # observation 是一个字典，包含了image以及proprio
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, self.textembeding, observation)
        return action[:self.deploy_chunk_size, :]
    
    def get_last_obs(self):
        return self.runner.obs[-1]

def class_decorator(task_name):
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance