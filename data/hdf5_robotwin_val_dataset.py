import os
import pickle
import numpy as np
import cv2
import yaml
import h5py
import copy
import torch

import sys
if __name__=="__main__":
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_path)

from configs.state_vec import STATE_VEC_IDX_MAPPING

class RoboTwinVLADataset:
    '''
    This class is used to sample episodes from RoboTwin dataset
    '''
    def __init__(self, type="all",
                 use_precompute_language_emb=True,
                 ) -> None:
        self.DATASET_NAME = 'RoboTwin'
        self.dataset_root = '/nvme_data/embodied_agent/robotwin_data'
        self.use_precompute_language_emb = use_precompute_language_emb
        self.pre_compute_emb_root = '/nvme_data/liangzhi/rdt/gpt_precompute_prompt/'
        
        self.type = type
        assert self.type in ['all', 'all_no_head', 'mix', 'mix_no_head', 'only_correction', 'original']

        self.task_detail_dict = {
            'blocks_stack_easy': ['general', 'left-left', 'left-right', 'right-left', 'right-right'],
            'shoe_place': ['general', 'left', 'right'],
            'block_hammer_beat': ['general', 'left', 'right'],
            'container_place': ['general', 'left', 'right'],
            'empty_cup_place': ['general', 'left', 'right'],
            'dual_bottles_pick_hard': ['general'],
            'diverse_bottles_pick': ['general'],
            'block_handover': ['general'],
        }

        self.task_sample_num_dict = {
            'blocks_stack_easy': 1000,
            'shoe_place': 1000,
            'block_hammer_beat': 1000,
            'container_place': 1000,
            'empty_cup_place': 600,
            'dual_bottles_pick_hard': 800,
            'diverse_bottles_pick': 600,
            'block_handover': 600,
        }

        if self.type == 'all' or self.type == 'mix':
            self.task_detail_dict['blocks_stack_easy_full'] = ['general']
            self.task_sample_num_dict['blocks_stack_easy_full'] = 500
            self.task_detail_dict['empty_cup_place_full'] = ['general']
            self.task_sample_num_dict['empty_cup_place_full'] = 500
        if self.type == 'all_no_head' or self.type == 'mix_no_head':
            self.task_detail_dict['blocks_stack_easy_end'] = ['general']
            self.task_sample_num_dict['blocks_stack_easy_end'] = 500
            self.task_detail_dict['empty_cup_place_end'] = ['general']
            self.task_sample_num_dict['empty_cup_place_end'] = 500
        if self.type == 'all' or self.type == 'all_no_head' or self.type == 'only_correction':
            self.task_detail_dict['blocks_stack_easy_part_repick_black'] = ['general']
            self.task_sample_num_dict['blocks_stack_easy_part_repick_black'] = 615
            self.task_detail_dict['blocks_stack_easy_part_repick_red'] = ['general']
            self.task_sample_num_dict['blocks_stack_easy_part_repick_red'] = 385
            self.task_detail_dict['empty_cup_place_part_repick'] = ['general']
            self.task_sample_num_dict['empty_cup_place_part_repick'] = 1000

        self.indices = self._make_idxes()
        
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
    def _make_idxes(self,):
        indices = []
        for task in self.task_detail_dict.keys():
            #task = task_detail[0]
            task_root_dir = os.path.join(self.dataset_root, task + '_hdf5')
            language_ins_dir = os.path.join(task_root_dir, 'language_ins.pkl')
            for episode_idx in range(self.task_sample_num_dict[task]):
            #for item in os.listdir(task_root_dir):
                #if item.split('.')[-1] == 'h5':
                    task_episode_dir = os.path.join(task_root_dir, f'episode{episode_idx}.h5')
                    # print(task_episode_dir)
                    with h5py.File(task_episode_dir, 'r') as f:
                        # print("f:", f['end_pos'][:].shape)
                        # traj_len = f['traj_len'][()]
                        traj_len = f['end_pos'][:].copy().shape[0]
                    for begin_id in range(0, traj_len-1):   #这里begin_id 指当前观测的位置，出现边界条件都采用 重复 padding
                        indices.append((task_episode_dir, begin_id, language_ins_dir, task))
        return indices

    def __len__(self):
        return len(self.indices)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False, check_data=False):
        while True:
            if index is None:
                rnd_index = np.random.randint(0, self.__len__())
                episode_file_path, begin_idx, language_ins_dir, task = self.indices[rnd_index]
            else:
                episode_file_path, begin_idx, language_ins_dir, task = self.indices[index]
            valid, sample = self._parse_file(episode_file_path, begin_idx, language_ins_dir, task) if not state_only else self._parse_file_state_only(episode_file_path, begin_idx, language_ins_dir)
            if valid:
                if not check_data:
                    return sample
                else:
                    return sample, episode_file_path, begin_idx
            else:
                index = np.random.randint(0, len(self.indices))

    def _parse_file(self, episode_file_path, begin_idx, language_ins_dir, task):
        gripper_scaler_1, gripper_scaler_2 = 0.045, 0.045
        

        with h5py.File(episode_file_path, 'r') as f:
            joint_poses = copy.deepcopy(f['joint_action'][:])
            task_detail = 'general'
            # task_detail = f['task_detail'][()].decode('utf-8')
            num_steps = joint_poses.shape[0]

            joint_poses[:, [6, 13]] = (joint_poses[:, [6, 13]] > 0) * (joint_poses[:, [6, 13]]) / np.array([[gripper_scaler_1, gripper_scaler_2]]) + (joint_poses[:, [6, 13]] <= 0) * joint_poses[:, [6, 13]]

            joint_actions = copy.deepcopy(joint_poses)

            state = copy.deepcopy(joint_poses[begin_idx:begin_idx + 1, :])
            state_std = np.std(joint_poses, axis=0)
            state_mean = np.mean(joint_poses, axis=0)
            state_norm = np.sqrt(np.mean(joint_poses ** 2, axis=0))
            actions = copy.deepcopy(joint_actions[begin_idx + 1: min(begin_idx + 1 + self.CHUNK_SIZE, num_steps)])
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)
            
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING["left_gripper_open"]
                ] + [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec

            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            actions = fill_in_state(actions)

            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(begin_idx - self.IMG_HISORY_SIZE + 1, 0), begin_idx + 1):
                    img = copy.deepcopy(f[key][i])
                    imgs.append(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR))
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs
            
            cam_high = parse_img('front_camera_list')
            valid_len = min(begin_idx + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array([False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len)
            cam_left_wrist = parse_img('left_camera_list')
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img('right_camera_list')
            cam_right_wrist_mask = cam_high_mask.copy()
        
        if not self.use_precompute_language_emb:
            with open(language_ins_dir, 'rb') as f:
                instruction_dict = pickle.load(f)       # {{'general': [ins1, ins2, ...], 'left-left': [ins1, ins2, ...], ...}}
            assert task_detail in self.task_detail_dict[task]
            
            if task_detail != 'general':
                ins_type = np.random.randint(2)
                if ins_type == 0:
                    task_ins_key = 'general'
                else:
                    task_ins_key = task_detail
            else:
                task_ins_key = task_detail
            instruction = np.random.choice(instruction_dict[task_ins_key])

        else:
            assert task_detail in self.task_detail_dict[task]
            if task_detail != 'general':
                ins_type = np.random.randint(2)
                if ins_type == 0:
                    task_ins_key = 'general'
                else:
                    task_ins_key = task_detail
            else:
                task_ins_key = task_detail

            language_ins_emb_dir = os.path.join(self.pre_compute_emb_root, task, task_ins_key)
            num_language_ins = len(os.listdir(language_ins_emb_dir))
            ins_emb_dir = os.path.join(language_ins_emb_dir, 'lang_embed_{}.pt'.format(np.random.randint(num_language_ins)))
            instruction = ins_emb_dir

        meta = {
            'dataset_name': self.DATASET_NAME,
            '#steps': num_steps,
            'step_id': begin_idx,
            'instruction': instruction
            }

        return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
                } 

    def _parse_file_state_only(self, episode_file_path, begin_idx, language_ins_dir):
        gripper_scaler_1, gripper_scaler_2 = 0.045, 0.045
        with h5py.File(episode_file_path, 'r') as f:
            joint_poses = copy.deepcopy(f['joint_action'][:])
            num_steps = joint_poses.shape[0]
            joint_poses[:, [6, 13]] = (joint_poses[:, [6, 13]] > 0) * (joint_poses[:, [6, 13]]) / np.array([[gripper_scaler_1, gripper_scaler_2]]) + (joint_poses[:, [6, 13]] <= 0) * joint_poses[:, [6, 13]]

            joint_actions = copy.deepcopy(joint_poses)
            state = copy.deepcopy(joint_poses[begin_idx:begin_idx + 1, :])
            actions = copy.deepcopy(joint_actions[begin_idx: min(begin_idx + self.CHUNK_SIZE, num_steps)])
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING["left_gripper_open"]
                ] + [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec

            state = fill_in_state(state)
            actions = fill_in_state(actions)

            return True, {
                "state": state,
                "action": actions
            }
        
if __name__ == "__main__":
    ds = RoboTwinVLADataset(type="only_correction")
    print('##################################')
    print(len(ds))
    data = ds.get_item(0)
    print(data['cam_high'][0])
    save_file_path = './test.png'
    cv2.imwrite(save_file_path, data['cam_high'][0])
    # for i in range(len(ds)):
    #     print(f"Processing episode {i}/{len(ds)}...")
    #     collect_list = []
    #     sample, episode_file_path, begin_idx = ds.get_item(i, debug=True)
    #     if sample['cam_high'].shape[1] != 480:
    #         collect_list.append((episode_file_path, begin_idx, 'cam_high'))
        
    #     if sample['cam_left_wrist'].shape[1] != 480:
    #         collect_list.append((episode_file_path, begin_idx, 'cam_left'))
        
    #     if sample['cam_right_wrist'].shape[1] != 480:
    #         collect_list.append((episode_file_path, begin_idx, 'cam_right'))