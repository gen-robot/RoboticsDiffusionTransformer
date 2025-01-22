import argparse
import sys
import threading
import time
import yaml
from collections import deque
import numpy as np
import torch
import cv2
from .robotwin_model import create_model, RoboticDiffusionTransformerModel
from typing import Dict, Callable, List
from PIL import Image as PImage

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

class RDTRobotwinAdapter:
    def __init__(self,
                 rdt_args,
                 task_name = None,
                 eval_episodes=20,
                 max_steps=400,
                 tqdm_interval_sec=5.0,
                 
                 ):
        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.n_obs_steps = rdt_args.img_history_size
        #self.n_action_steps = rdt_args.chunk_size
        # self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.obs = deque(maxlen=self.n_obs_steps + 1)
        self.env = None
    
    def stack_last_n_obs(self, all_obs, n_steps):
        assert(len(all_obs) > 0)
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]                      # 在 对起始状态的处理是 pad 到 obs_len
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
        return result
    
    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        # obs['agent_pos']: np.array(), (14,) 注意，还没有scale过
        # obs['observation']['head_camera']: np.array(), (640, 480, 3)
        # obs['observation']['front_camera']: np.array(), (640, 480, 3)
        # obs['observation']['left_camera']: np.array(), (640, 480, 3)
        # obs['observation']['right_camera']: np.array(), (640, 480, 3)
        self.obs.append(current_obs)

    def get_n_steps_obs(self,):
        assert(len(self.obs) > 0), 'no observation is recorded, please update obs first'
        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps
            )

        return result
    
    def get_action(self, policy: RoboticDiffusionTransformerModel, text_embeds, observation=None):
        device = policy.device
        if observation is not None:
            self.obs.append(observation)
        obs = self.get_n_steps_obs()
        
        #create obs dict
        np_obs_dict = dict(obs)
        #obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

        # run policy
        proprio = torch.from_numpy(np_obs_dict['agent_pos'][-1, :]).unsqueeze(0).to(device=device)      #tensor (1, 14)               # 当前时刻的关节位姿
        ext_pre = np_obs_dict['front_cam'][-2, :, :, :]              # 前一时刻的图像
        ext_curr = np_obs_dict['front_cam'][-1, :, :, :]             # 当前时刻的图像
        right_wrist_pre = np_obs_dict['right_cam'][-2, :, :, :]      # 前一时刻右手腕的图像
        right_wrist_curr = np_obs_dict['right_cam'][-1, :, :, :]     # 当前时刻右手腕的图像
        left_wrist_pre = np_obs_dict['left_cam'][-2, :, :, :]        # 前一时刻左手腕的图像
        left_wrist_curr = np_obs_dict['left_cam'][-1, :, :, :]       # 当前时刻左手腕的图像

        images = [ext_pre, right_wrist_pre, left_wrist_pre, ext_curr, right_wrist_curr, left_wrist_curr]
        images = [PImage.fromarray((arr.transpose((1, 2, 0)) * 255).astype(np.uint8)) if arr is not None else None for arr in images]
        action_chunk = policy.step(proprio, images, text_embeds)
        if len(action_chunk.shape) > 2:
            action_chunk = action_chunk.squeeze(0)
        np_action_chunk = action_chunk.detach().cpu().numpy()
        return np_action_chunk