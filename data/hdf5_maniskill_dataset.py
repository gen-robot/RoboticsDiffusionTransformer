import os
import sys

if __name__=="__main__":
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_path)

import h5py
import yaml
import numpy as np
# Assuming STATE_VEC_IDX_MAPPING is a dictionary mapping state variable names to indices
from configs.state_vec import STATE_VEC_IDX_MAPPING
import glob
from scipy.interpolate import interp1d
from PIL import Image


def interpolate_action_sequence(action_sequence, target_size):
    """
    Extend the action sequece to `target_size` by linear interpolation.
    
    Args:
        action_sequence (np.ndarray): original action sequence, shape (N, D).
        target_size (int): target sequence length.
    
    Returns:
        extended_sequence (np.ndarray): extended action sequence, shape (target_size, D).
    """
    N, D = action_sequence.shape
    indices_old = np.arange(N)
    indices_new = np.linspace(0, N - 1, target_size)

    interp_func = interp1d(indices_old, action_sequence, 
                           kind='linear', axis=0, assume_sorted=True)
    action_sequence_new = interp_func(indices_new)

    return action_sequence_new


DATASET_STATS = {'state_min': [-0.7463043928146362, -0.0801204964518547, -0.4976441562175751, -2.657780647277832, -0.5742632150650024, 1.8309762477874756, -2.2423808574676514, 0.0, 0.0], 
                 'state_max': [0.7645499110221863, 1.4967026710510254, 0.4650936424732208, -0.3866899907588959, 0.5505855679512024, 3.2900545597076416, 2.5737812519073486, 0.03999999910593033, 0.03999999910593033], 
                 'action_min': [-0.7472005486488342, -0.08631071448326111, -0.4995281398296356, -2.658363103866577, -0.5751323103904724, 1.8290787935256958, -2.245187997817993, -1.0], 
                 'action_max': [0.7654682397842407, 1.4984270334243774, 0.46786263585090637, -0.38181185722351074, 0.5517147779464722, 3.291581630706787, 2.575840711593628, 1.0], 
                 'action_std': [0.2199309915304184, 0.18780815601348877, 0.13044124841690063, 0.30669933557510376, 0.1340624988079071, 0.24968451261520386, 0.9589747190475464, 0.9827960729598999], 
                 'action_mean': [-0.00885344110429287, 0.5523102879524231, -0.007564723491668701, -2.0108158588409424, 0.004714342765510082, 2.615924596786499, 0.08461848646402359, -0.19301606714725494]}


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embodiment dataset
    stored in HDF5 files.
    """
    def __init__(self, type="all", 
                 recompute_normalization=False,
                 with_other_task=True):
        # The name of your dataset
        self.DATASET_NAME = "agilex"

        self.data_dir = "/nvme_data/liangzhi/rdt-maniskill/demo_1k"
        self.tasks = os.listdir(self.data_dir)

        # Multiple tasks
        self.type = type
        assert self.type in ["all", "original", "only_correction", "mix"]
        if with_other_task:
            self.tasks = ['PickCube-v1', 
                          'StackCube-v1', 
                          'PlugCharger-v1', 
                          'PushCube-v1', 
                          'PegInsertionSide-v1']
        else:
            self.tasks = ['StackCube-v1']
        if self.type == "all" or self.type == "mix":
            self.tasks.append('StackCube-v2')
        if self.type == "all" or self.type == "only_correction":
            self.tasks.append('StackCube-v1-correction')
        
        # Load configuration from YAML file
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        self.task_demo_num = {
            "PegInsertionSide-v1": 1000,
            "PickCube-v1": 1000,
            "StackCube-v1": 1000,
            "PlugCharger-v1": 1000,
            "PushCube-v1": 1000,
            "StackCube-v2": 500,
            "StackCube-v1-correction": 1000,
        }

        self.sum_demo_num = 0
        for task in self.tasks:
            self.sum_demo_num += self.task_demo_num[task]

        self.img = []
        self.state = []
        self.action = []

        # open the hdf5 files in memory to speed up the data loading
        for task in self.tasks:
            file_path = glob.glob(os.path.join(self.data_dir, task, 'motionplanning', '*.h5'))[0]
            with h5py.File(file_path, "r") as f:
                trajs = f.keys() #  traj_0, traj_1,
                # sort by the traj number
                trajs = sorted(trajs, key=lambda x: int(x.split('_')[-1]))
                max_index = self.task_demo_num[task]
                for traj in trajs:

                    max_index -= 1
                    if max_index < 0:
                        break

                    # images = f[traj]['obs']['sensor_data']['base_camera']['rgb'][:]
                    states = f[traj]['obs']['agent']['qpos'][:]
                    actions = f[traj]['actions'][:]

                    if task == "StackCube-v1-correction" or task == "StackCube-v2":
                        states = states[1:]
                        actions = actions[1:]

                    self.state.append(states)
                    self.action.append(actions)
                    # self.img.append(images)

        if recompute_normalization:
            self.state_min = np.concatenate(self.state).min(axis=0)
            self.state_max = np.concatenate(self.state).max(axis=0)
            self.action_min = np.concatenate(self.action).min(axis=0)
            self.action_max = np.concatenate(self.action).max(axis=0)
            self.action_std = np.concatenate(self.action).std(axis=0)
            self.action_mean = np.concatenate(self.action).mean(axis=0)
        else:
            self.state_min = np.array(DATASET_STATS['state_min'])
            self.state_max = np.array(DATASET_STATS['state_max'])
            self.action_min = np.array(DATASET_STATS['action_min'])
            self.action_max = np.array(DATASET_STATS['action_max'])
            self.action_std = np.array(DATASET_STATS['action_std'])
            self.action_mean = np.array(DATASET_STATS['action_mean'])

        self.task2lang = {
            "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
            "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
            "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
            "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
            "PushCube-v1": "Push and move a cube to a goal region in front of it.",
            "StackCube-v2":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
            "StackCube-v1-correction":  "Mission failure detected! Re-pick up the red cube first. After that continue to tack it on top of a green cube.",
        }

    def __len__(self):
        # Assume each file contains 100 episodes
        return self.sum_demo_num

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index=None):
        """
        Get a training sample at a random timestep.

        Args:
            index (int, optional): The index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
            sample (dict): A dictionary containing the training sample.
        """
        while True:
            if index is None:
                index = np.random.randint(0, self.__len__())
            valid, sample = self.parse_hdf5_file(index)
            if valid:
                return sample
            else:
                index = np.random.randint(0, self.__len__())

    def get_index(self, index):
        """
        Get the task index and the inner index of the task.

        Args:
            index (int): The index of the episode.

        Returns:
            task_index (int): The index of the task.
            task_inner_index (int): The inner index of the task.
        """
        task_index = 0
        task_inner_index = index

        for task in self.tasks:
            if task_inner_index < self.task_demo_num[task]:
                break
            task_inner_index -= self.task_demo_num[task]
            task_index += 1

        return task_index, task_inner_index

    def parse_hdf5_file(self, index):
        """
        Parse an HDF5 file to generate a training sample at a random timestep.

        Args:
            file_path (str): The path to the HDF5 file.

        Returns:
            valid (bool): Whether the episode is valid.
            dict: A dictionary containing the training sample.
        """
        num_steps = len(self.action[index])
        step_index = np.random.randint(0, num_steps)
        task_index, task_inner_index = self.get_index(index)
        language = self.task2lang[self.tasks[task_index]]
        # Skip these episodes since in the eef version dataset they are invalid.
        if self.tasks[task_index] == 'PegInsertionSide-v1' and task_inner_index > 400:
            return False, None
        proc_index = task_inner_index // 100
        episode_index = task_inner_index % 100
        # images0 = self.img[index]
        # normalize to -1, 1
        states = (self.state[index] - self.state_min) / (self.state_max - self.state_min) * 2 - 1
        states = states[:, :-1]  # remove the last state as it is replicate of the -2 state
        actions = (self.action[index] - self.action_min) / (self.action_max - self.action_min) * 2 - 1
        
        # Get image history
        start_img_idx = max(0, step_index - self.IMG_HISTORY_SIZE + 1)
        end_img_idx = step_index + 1
        img_history = []
        for i in range(start_img_idx, end_img_idx):
            img_path = os.path.join(self.data_dir, self.tasks[task_index], 'motionplanning', f'{proc_index}', f'{episode_index}', f"{i + 1}.png")
            img = np.array(Image.open(img_path))
            img_history.append(img)
        img_history = np.array(img_history)
        # img_history = images0[start_img_idx:end_img_idx]
        img_valid_len = img_history.shape[0]

        # Pad images if necessary
        if img_valid_len < self.IMG_HISTORY_SIZE:
            padding = np.tile(img_history[0:1], (self.IMG_HISTORY_SIZE - img_valid_len, 1, 1, 1))
            img_history = np.concatenate([padding, img_history], axis=0)

        img_history_mask = np.array(
            [False] * (self.IMG_HISTORY_SIZE - img_valid_len) + [True] * img_valid_len
        )

        # Compute state statistics
        state_std = np.std(states, axis=0)
        state_mean = np.mean(states, axis=0)
        state_norm = np.sqrt(np.mean(states ** 2, axis=0))

        # Get state and action at the specified timestep
        state = states[step_index: step_index + 1]
        runtime_chunksize = self.CHUNK_SIZE // 4
        action_sequence = actions[step_index: step_index + runtime_chunksize]
        # we use linear interpolation to pad the action sequence

        # Pad action sequence if necessary
        if action_sequence.shape[0] < runtime_chunksize:
            padding = np.tile(action_sequence[-1:], (runtime_chunksize - action_sequence.shape[0], 1))
            action_sequence = np.concatenate([action_sequence, padding], axis=0)

        action_sequence = interpolate_action_sequence(action_sequence, self.CHUNK_SIZE)

        # Fill state and action into unified vectors
        def fill_in_state(values):
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
            ] + [
                STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec

        state_indicator = fill_in_state(np.ones_like(state_std))
        state = fill_in_state(state)
        state_std = fill_in_state(state_std)
        state_mean = fill_in_state(state_mean)
        state_norm = fill_in_state(state_norm)
        action_sequence = fill_in_state(action_sequence)

        # Assemble the meta information
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_index,
            "instruction": language
        }

        # Return the resulting sample
        return True, {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": action_sequence,
            "state_indicator": state_indicator,
            "cam_high": img_history,  # Assuming images0 are high-level camera images
            "cam_high_mask": img_history_mask,
            "cam_left_wrist": np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0)),
            "cam_left_wrist_mask": np.zeros(self.IMG_HISTORY_SIZE, dtype=bool),
            "cam_right_wrist": np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0)),
            "cam_right_wrist_mask": np.zeros(self.IMG_HISTORY_SIZE, dtype=bool),
        }


if __name__ == "__main__":
    from PIL import Image
    
    ds = HDF5VLADataset(type="all")

    json_data = {
        'state_min': ds.state_min.tolist(),
        'state_max': ds.state_max.tolist(),
        'action_min': ds.action_min.tolist(),
        'action_max': ds.action_max.tolist(),
        'action_std': ds.action_std.tolist(),
        'action_mean': ds.action_mean.tolist(),
    }
    print(json_data)

    
