import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

try:
    from ..configs.state_vec import STATE_VEC_IDX_MAPPING
    from ..constants import RDT_ROOT_DIR, RDT_CONFIG_DIR
    from .rotation_utils import quaternion_to_ortho6d
except ImportError:
    from configs.state_vec import STATE_VEC_IDX_MAPPING
    from constants import RDT_ROOT_DIR, RDT_CONFIG_DIR
    from rotation_utils import quaternion_to_rotation_matrix, rotation_matrix_to_ortho6d


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(
            self, 
            data_path: str=None, 
            robot_name: str='rdt', 
            use_precomp_lang_embed: bool=False,
            max_demo_per_task: int=None,
            instruction_mode: str="random",
            enable_eef_obs=False,
            enable_eef_action=False
    ) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        if data_path is None:
            HDF5_DIR = f"{RDT_ROOT_DIR}/data/datasets/agilex/cobot_data/"
        else:
            HDF5_DIR = data_path
        if max_demo_per_task is None:
            max_demo_per_task = 999

        self.DATASET_NAME = "agilex"
        self.use_precomp_lang_embed = use_precomp_lang_embed
        self.instruction_mode = instruction_mode
        self.enable_eef_obs = enable_eef_obs
        self.enable_eef_action = enable_eef_action

        with open(f'{RDT_CONFIG_DIR}/gripper_scale.json', 'r') as gs_file:
            self.gs_dict = json.load(gs_file)
            assert robot_name in self.gs_dict, f"Robot name {robot_name} not found in gripper scale dict."
            self.gripper_qpos_scale = self.gs_dict[robot_name]['qpos']
            self.gripper_action_scale = self.gs_dict[robot_name]['action']

        assert os.path.exists(HDF5_DIR), f"Dataset directory {HDF5_DIR} does not exist."
        
        self.file_paths = []
        self.invalid_file_paths = []
        for root, _, files in os.walk(HDF5_DIR, followlinks=True):
            for filename in sorted(
                fnmatch.filter(files, '*.hdf5'), 
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            ):
                episode_id = int(filename.split('/')[-1].split('_')[-1].split('.')[0])
                if episode_id >= max_demo_per_task:
                    continue
                file_path = os.path.join(root, filename)
                file_dir = os.path.dirname(file_path)
                if self.use_precomp_lang_embed:
                    assert os.path.exists(os.path.join(file_dir, "precomp_lang_embeds")), \
                        f"Language embeddings not found for {file_path} for precomputed language embeddings."
                try:
                    f = h5py.File(file_path, 'r')
                    self.file_paths.append(file_path)
                    f.close()
                except:
                    self.invalid_file_paths.append(file_path)
                    print(f"Failed to open {file_path}.")

        # write the invalid file paths to a file
        with open(f'{RDT_CONFIG_DIR}/invalid_file_paths.txt', 'w') as f:
            for file_path in self.invalid_file_paths:
                f.write(f"{file_path}\n")

        print(f"Found {len(self.file_paths)} HDF5 files in the dataset directory.")

        assert len(self.file_paths) > 0, "No HDF5 files found in the dataset directory."

        # Load the config
        with open(f'{RDT_CONFIG_DIR}/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, step_id: int=None, instr_mode: str=None, state_only: bool=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            step_id (int, optional): the index of the sampled step.
                If not provided, a random timestep will be selected.
            instr_mode (str, optional): the instruction mode.
                It can be "normal", "simplified", or "expanded". Defaults to "normal".
                If not provided, a random mode will be selected from ["normal", "simplified", "expanded"].
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            if instr_mode is None:
                instr_mode = self.instruction_mode
            valid, sample = self.parse_hdf5_file(file_path, step_id=step_id, instr_mode=instr_mode) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path, step_id: int=None, instr_mode: str="normal"):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")

            # We randomly sample a timestep if step_id is not provided
            if step_id is None:
                step_id = np.random.randint(first_idx-1, num_steps)
            
            # Load the instruction
            dir_path = os.path.dirname(file_path)
            with open(os.path.join(dir_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
                instruction_dict = json.load(f_instr)
            # We have 1/3 prob to use original instruction,
            # 1/3 to use simplified instruction,
            # and 1/3 to use expanded instruction.
            if instr_mode == "normal":
                instruction_type = 'instruction'
            elif instr_mode == "simplified":
                instruction_type = 'simplified_instruction'
            elif instr_mode == "expanded":
                instruction_type = 'expanded_instruction'
            elif instr_mode == "nonsense":
                instruction_type = None
            else:
                instruction_type = np.random.choice([
                    'instruction', 'simplified_instruction', 'expanded_instruction'])
            if instruction_type is None:
                instruction = "This is a meaningless instruction that has nothing to do with any real task and is only used to test the effect of the language instruction."
            else:
                instruction = instruction_dict[instruction_type]
                if isinstance(instruction, list):
                    instruction = np.random.choice(instruction)
            # FIXME: use_precomp_lang_embed will cover the instr_mode and randomly sample an instruction
            # You can also use precomputed language embeddings (recommended)
            if self.use_precomp_lang_embed:
                # Load the precomputed language embeddings
                embeds_dir = os.path.join(dir_path, "precomp_lang_embeds")
                all_embeds = [
                    os.path.join(embeds_dir, p) for p in sorted(os.listdir(embeds_dir))
                    if p.endswith(".pt")
                ]
                assert len(all_embeds) > 0, \
                    "No language embeddings found in {}.".format(embeds_dir)
                # randomly sample a language embedding
                instruction = np.random.choice(all_embeds)
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            # Rescale gripper to [0, 1]
            qpos = qpos / np.array(
               [[1, 1, 1, 1, 1, 1, self.gripper_qpos_scale[0], 
                 1, 1, 1, 1, 1, 1, self.gripper_qpos_scale[1]]] 
            )
            target_qpos = f['action'][step_id:step_id+self.CHUNK_SIZE] / np.array(
               [[1, 1, 1, 1, 1, 1, self.gripper_action_scale[0], 
                 1, 1, 1, 1, 1, 1, self.gripper_action_scale[1]]] 
            )
            
            # Parse the state and action
            state = qpos[step_id:step_id+1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)

            if self.enable_eef_obs or self.enable_eef_action:
                if 'ee_pose' not in f['observations']:
                    enable_eef_obs = False
                    enable_eef_action = False
                else:
                    ee_pose = f['observations']['ee_pose'][:]
                    enable_eef_obs = self.enable_eef_obs
                    enable_eef_action = self.enable_eef_action

            if enable_eef_obs:
                ee_pose_l, ee_pose_r = ee_pose[:, :7], ee_pose[:, 7:]
                ee_pos_l, ee_quat_l = ee_pose_l[step_id:step_id+1, :3], ee_pose_l[step_id:step_id+1, 3:]
                ee_rot6d_l = quaternion_to_ortho6d(ee_quat_l, 'xyzw')
                ee_pos_r, ee_quat_r = ee_pose_r[step_id:step_id+1, :3], ee_pose_r[step_id:step_id+1, 3:]
                ee_rot6d_r = quaternion_to_ortho6d(ee_quat_r, 'xyzw')
                eef = np.concatenate([ee_pos_l, ee_rot6d_l, ee_pos_r, ee_rot6d_r], axis=-1)
                eef_std = np.std(eef, axis=0)
                eef_mean = np.mean(eef, axis=0)
                eef_norm = np.sqrt(np.mean(eef**2, axis=0))
            else:
                eef, eef_std, eef_mean, eef_norm = None, None, None, None

            if enable_eef_action:
                assert self.enable_eef_obs, "Enable eef observation first."
                cur_step_id = min(step_id+1, num_steps-1) # if step_id is the last step, use the last step's eef
                eef_actions = eef[cur_step_id:cur_step_id+self.CHUNK_SIZE] # next step's eef as current step's action
                if eef_actions.shape[0] < self.CHUNK_SIZE:
                    # Pad the actions using the last action
                    eef_actions = np.concatenate([
                        eef_actions,
                        np.tile(eef_actions[-1:], (self.CHUNK_SIZE-eef_actions.shape[0], 1))
                    ], axis=0)
            else:
                eef_actions = None
            
            # Fill the state/action into the unified vector
            def fill_in_state(qpos, eef=None):
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
                uni_vec = np.zeros(qpos.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = qpos

                if eef is not None:
                    UNI_EEF_INDICES = [
                        STATE_VEC_IDX_MAPPING[f"left_eef_pos_{i}"] for i in ['x', 'y', 'z']
                    ] + [
                        STATE_VEC_IDX_MAPPING[f"left_eef_angle_{i}"] for i in range(6)
                    ] + [
                        STATE_VEC_IDX_MAPPING[f"right_eef_pos_{i}"] for i in ['x', 'y', 'z']
                    ] + [
                        STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)
                    ]
                    uni_vec[..., UNI_EEF_INDICES] = eef

                return uni_vec
            state = fill_in_state(state, eef)
            state_indicator = fill_in_state(
                np.ones_like(state_std), 
                np.ones_like(eef_std) if eef is not None else None
            )
            state_std = fill_in_state(state_std, eef_std)
            state_mean = fill_in_state(state_mean, eef_mean)
            state_norm = fill_in_state(state_norm, eef_norm)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            raw_actions = actions.copy()
            actions = fill_in_state(actions, eef_actions)
            
            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                    img = f['observations']['images'][key][i]
                    if f.attrs.get('compress', True):
                        imgs.append(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR))
                    else:
                        imgs.append(img)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs
            # `cam_high` is the external camera image
            cam_high = parse_img('cam_high')
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist = parse_img('cam_left_wrist')
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img('cam_right_wrist')
            cam_right_wrist_mask = cam_high_mask.copy()
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "qpos": qpos[step_id:step_id+1],
                "raw_actions": raw_actions,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # Rescale gripper to [0, 1]
            qpos = qpos / np.array(
               [[1, 1, 1, 1, 1, 1, self.gripper_qpos_scale[0], 
                 1, 1, 1, 1, 1, 1, self.gripper_qpos_scale[1]]] 
            )
            target_qpos = f['action'][:] / np.array(
               [[1, 1, 1, 1, 1, 1, self.gripper_action_scale[0], 
                 1, 1, 1, 1, 1, 1, self.gripper_action_scale[1]]] 
            )
            
            # Parse the state and action
            state = qpos[first_idx-1:]
            action = target_qpos[first_idx-1:]
            
            # Fill the state/action into the unified vector
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
            action = fill_in_state(action)
            
            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }


if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
