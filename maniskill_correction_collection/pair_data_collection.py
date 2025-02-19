from typing import Callable, List, Type
import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

from maniskill_correction_collection.ik_solver import IKSolver
from maniskill_correction_collection.rotation_utils import (
    quaternion_to_rot_matrix,
    get_direction_vector,
    get_direction_vector_from_quat
)

import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel
import torch
from collections import deque
from PIL import Image
import cv2
from pathlib import Path
import imageio.v2 as iio
import time

import random
import h5py

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="StackCube-v1", help=f"Environment to run motion planning solver on. ")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=25, help="Number of trajectories to test.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    parser.add_argument("--lang_embeds_path", type=str, default="./lang_embeds/", help="Path to language embedings.")
    return parser.parse_args()

def save_mp4(save_path: str, frames: list[np.ndarray], fps: int = 60):
    videoWriter = iio.get_writer(
        save_path,
        format="ffmpeg",  # type: ignore
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
    )

    for frame in frames:
        videoWriter.append_data(frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


task2lang = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}


def check_target(info, flag):
    if flag == 2:
        if "success" in info:
            return info["success"]
        return False
    else:
        return info["is_cubeA_grasped"]

def save_data(save_path, count, obs_image_array, proprio_array, action_array, low_action_array):
    file_path = os.path.join(save_path, 'motionplanning', 'data.h5')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    data_size = len(action_array)
    for i in range(data_size):
        action = action_array[i]
        proprio = proprio_array[i]
        low_action = low_action_array[i]

        if isinstance(proprio, torch.Tensor):
            proprio = proprio.cpu().numpy()
        proprio_array[i] = proprio[0]
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action_array[i] = action
        if isinstance(low_action, torch.Tensor):
            low_action = low_action.cpu().numpy()

    count -= 1

    if count == 0:
        with h5py.File(file_path, 'w') as f:
            traj = f.create_group(f'traj_{count}')

            traj.create_group("obs").create_group("agent").create_dataset('qpos', data=np.array(proprio_array))
            traj.create_dataset('actions', data=np.array(action_array))
            traj.create_dataset('low_actions', data=np.array(low_action_array))
    else:
        with h5py.File(file_path, 'a') as f:
            traj = f.create_group(f'traj_{count}')

            traj.create_group("obs").create_group("agent").create_dataset('qpos', data=np.array(proprio_array))
            traj.create_dataset('actions', data=np.array(action_array))
            traj.create_dataset('low_actions', data=np.array(low_action_array))
    
    count_head = count // 100
    count_tail = count % 100
    image_save_path = os.path.join(save_path, 'motionplanning', str(count_head), str(count_tail))

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    else:
        for f in os.listdir(image_save_path):
            file_path = os.path.join(image_save_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    for i, img in enumerate(obs_image_array):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(image_save_path, f"{i}.png"), img_rgb)

def get_policy_actions(policy, proprio, obs_window, text_embed):
    image_arrs = []
    for window_img in obs_window:
        image_arrs.append(window_img)
        image_arrs.append(None)
        image_arrs.append(None)
    images = [Image.fromarray(arr) if arr is not None else None
            for arr in image_arrs]
    actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()

    return actions

def main(args):
    env_id = args.env_id
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        reward_mode="dense" if args.reward_mode is None else args.reward_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend
    )

    config_path = 'configs/base.yaml'
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    pretrained_path = args.pretrained_path
    policy = create_model(
        args=config, 
        dtype=torch.bfloat16,
        pretrained=pretrained_path,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path
    )

    text_embed_name = os.path.join(args.lang_embeds_path, f'text_embed_{env_id}.pt')

    if os.path.exists(text_embed_name):
        text_embed = torch.load(text_embed_name)
    else:
        text_embed = policy.encode_instruction(task2lang[env_id])
        torch.save(text_embed, text_embed_name)

    render_dir = f"./outs/render/correction-{env_id}/"
    Path(render_dir).mkdir(parents=True, exist_ok=True)

    base_seed = 20241201
    total_episodes = 3000 # Number of correction we collect
    MAX_EPISODE_STEPS = 500
    success_count = 0  
    
    use_correction_count = 0
    correction_success_count = 0
    correction_process_count = 0

    ik_solver = IKSolver(env.agent.robot)
    to_base = env.agent.robot.pose.inv()

    data_root = "/nvme_data/liangzhi/rdt-maniskill/datas"
    correction_success_render_path = os.path.join(data_root, "render/correction_success_render")
    if not os.path.exists(correction_success_render_path):
        os.makedirs(correction_success_render_path)
    correction_data_save_path = os.path.join(data_root, "demo_correction/StackCube-v1-correction/")
    if not os.path.exists(correction_data_save_path):
        os.makedirs(correction_data_save_path)
    
    normal_success_demo_count = 0
    correction_success_demo_count = 0

    import tqdm
    for episode in tqdm.trange(total_episodes):
        obs_window = deque(maxlen=2)
        obs, _ = env.reset(seed = episode + base_seed)
        policy.reset()

        img = env.render().squeeze(0).detach().cpu().numpy()
        obs_window.append(None)
        obs_window.append(np.array(img))
        proprio = obs['agent']['qpos'][:, :-1]

        global_steps = 0
        video_frames = []

        is_success = 0
        is_correction = 0
        done = False

        do_ik = False
        condition_flag = 1
        condition_check_steps = 6
        last_correction_index = 0

        # For data saving
        obs_image_array = []
        proprio_array = []
        action_array = []

        while global_steps < MAX_EPISODE_STEPS and not done:

            if do_ik:
                is_correction = 1
                last_correction_index = len(action_array)
                input_cube_pose = env.cubeA.pose.raw_pose
                x_vec, y_vec, z_vec = get_direction_vector_from_quat(env.cubeA.pose.raw_pose[0, 3:])
                
                target_vec = None
                index = 0

                correction_image_array = [obs_image_array[-1]]
                correction_proprio_array = [proprio_array[-1]]
                correction_action_array = [action_array[-1]]
                correction_l_action_array = [get_policy_actions(policy, proprio, obs_window, text_embed)]

                while target_vec is None and index < 16:
                    index += 1
                    if torch.abs(x_vec[2]) < 0.01:
                        target_vec = x_vec
                    elif torch.abs(y_vec[2]) < 0.01:
                        target_vec = y_vec
                    elif torch.abs(z_vec[2]) < 0.01:
                        target_vec = z_vec
                    
                    if target_vec is not None:
                        break
                        
                    action = torch.zeros(8, dtype=torch.float32, device=policy.device)
                    action[:7] = env.agent.robot.get_qpos()[0, :7]
                    action[7] = 1.0

                    obs_image_array.append(img)
                    proprio_array.append(obs['agent']['qpos'])
                    action_array.append(action)

                    correction_image_array.append(img)
                    correction_proprio_array.append(obs['agent']['qpos'])
                    correction_action_array.append(action)
                    correction_l_action_array.append(get_policy_actions(policy, proprio, obs_window, text_embed))

                    obs, reward, terminated, truncated, info = env.step(action)
                    img = env.render().squeeze(0).detach().cpu().numpy()
                    obs_window.append(img)
                    proprio = obs['agent']['qpos'][:, :-1]
                    video_frames.append(img)
                    global_steps += 1

                    x_vec, y_vec, z_vec = get_direction_vector_from_quat(env.cubeA.pose.raw_pose[0, 3:])
                
                if target_vec is None:
                    print("Failed to find target vector")
                    break

                angle = torch.atan2(target_vec[0], target_vec[1])

                way_vector = env.cubeA.pose.raw_pose[0, :3] - env.cubeB.pose.raw_pose[0, :3]
                way_angle = torch.atan2(way_vector[0], way_vector[1])
                while way_angle < -0.001:
                    way_angle += np.pi
                while way_angle > np.pi + 0.001:
                    way_angle -= np.pi
                
                interval_min = np.pi / 4
                interval_max = 3 * np.pi / 4

                if way_angle > np.pi / 4 and way_angle < 3 * np.pi / 4:
                    interval_min = 3 * np.pi / 4
                    interval_max = 5 * np.pi / 4

                while angle < interval_min - 0.01:
                    angle += np.pi / 2
                while angle > interval_max + 0.01:
                    angle -= np.pi / 2
                input_cube_pose[0, 3:] = torch.tensor([0, 
                                                       np.sin(angle / 2), 
                                                       np.cos(angle / 2), 
                                                       0], 
                                                       dtype=torch.float32, device=policy.device)

                target_pose = ik_solver.compute_target_pose(input_cube_pose)

                if target_pose is None:
                    print("Failed to compute target pose")
                    break

                target_pose = to_base * target_pose
                
                current_qpos = env.agent.robot.get_qpos()
                target_qpos = ik_solver.compute_target_action(target_pose, current_qpos)
                current_qpos = current_qpos[:, :7]
                do_ik = False

                if target_qpos is None:
                    print("Failed to compute target action")
                    break
                
                ik_steps = 32
                actions = torch.zeros((ik_steps, 8), dtype=torch.float32, device=policy.device)

                for i in range(ik_steps - 2):
                    actions[i, :7] = (current_qpos[0] * (ik_steps - 3 - i) + target_qpos[0] * (i + 1)) / (ik_steps - 2)
                    actions[i, 7] = 1.0
                
                actions[-2, :7] = target_qpos[0]
                actions[-2, 7] = 1.0
                actions[-1] = actions[-2]

                for idx in range(actions.shape[0]):
                    action = actions[idx]
                    
                    obs_image_array.append(img)
                    proprio_array.append(obs['agent']['qpos'])
                    action_array.append(action)

                    correction_image_array.append(img)
                    correction_proprio_array.append(obs['agent']['qpos'])
                    correction_action_array.append(action)
                    correction_l_action_array.append(get_policy_actions(policy, proprio, obs_window, text_embed))

                    obs, reward, terminated, truncated, info = env.step(action)
                    img = env.render().squeeze(0).detach().cpu().numpy()
                    obs_window.append(img)
                    proprio = obs['agent']['qpos'][:, :-1]
                    video_frames.append(img)
                    global_steps += 1
                
                # Pick the cube
                target_pose = ik_solver.compute_strong_target_pose(input_cube_pose)
                target_pose = to_base * target_pose
            
                current_qpos = env.agent.robot.get_qpos()
                target_qpos = ik_solver.compute_target_action(target_pose, current_qpos)
                current_qpos = current_qpos[:, :7]

                if target_qpos is None:
                    print("Failed to compute target action")
                    break

                ik_steps = 20
                actions = torch.zeros((ik_steps, 8), dtype=torch.float32, device=policy.device)

                for i in range(ik_steps - 6):
                    actions[i, :7] = (current_qpos[0] * (ik_steps - 7 - i) + target_qpos[0] * (i + 1)) / (ik_steps - 6)
                    actions[i, 7] = 1.0
                
                actions[-6, :7] = target_qpos[0]
                actions[-6, 7] = 0.0
                actions[-1] = actions[-2] = actions[-3] = actions[-4] = actions[-5] = actions[-6]

                for idx in range(actions.shape[0]):
                    action = actions[idx]

                    obs_image_array.append(img)
                    proprio_array.append(obs['agent']['qpos'])
                    action_array.append(action)

                    correction_image_array.append(img)
                    correction_proprio_array.append(obs['agent']['qpos'])
                    correction_action_array.append(action)
                    correction_l_action_array.append(get_policy_actions(policy, proprio, obs_window, text_embed))

                    obs, reward, terminated, truncated, info = env.step(action)
                    img = env.render().squeeze(0).detach().cpu().numpy()
                    obs_window.append(img)
                    proprio = obs['agent']['qpos'][:, :-1]
                    video_frames.append(img)
                    global_steps += 1
                
                # pick up the cube
                target_cube_pose = to_base * ik_solver.compute_target_pose(input_cube_pose)
                current_qpos = env.agent.robot.get_qpos()
                target_qpos = ik_solver.compute_target_action(target_cube_pose, current_qpos)
                current_qpos = current_qpos[:, :7]

                if target_qpos is None:
                    print("Failed to compute target action")
                    break
                
                ik_steps = 12
                actions = torch.zeros((ik_steps, 8), dtype=torch.float32, device=policy.device)
                for i in range(ik_steps):
                    actions[i, :7] = (current_qpos[0] * (ik_steps - 1 - i) + target_qpos[0] * (i + 1)) / (ik_steps)
                    actions[i, 7] = 0.0

                for idx in range(actions.shape[0]):
                    action = actions[idx]

                    obs_image_array.append(img)
                    proprio_array.append(obs['agent']['qpos'])
                    action_array.append(action)

                    correction_image_array.append(img)
                    correction_proprio_array.append(obs['agent']['qpos'])
                    correction_action_array.append(action)
                    correction_l_action_array.append(get_policy_actions(policy, proprio, obs_window, text_embed))

                    obs, reward, terminated, truncated, info = env.step(action)
                    img = env.render().squeeze(0).detach().cpu().numpy()
                    obs_window.append(img)
                    proprio = obs['agent']['qpos'][:, :-1]
                    video_frames.append(img)
                    global_steps += 1

                # Put the cue into the target
                target_cube_pose = env.cubeB.pose.raw_pose
                target_cube_pose[0, 3:] = env.agent.tcp.pose.raw_pose[0, 3:]
                target_cube_pose = to_base * ik_solver.compute_release_target_pose(target_cube_pose)
                current_qpos = env.agent.robot.get_qpos()
                target_qpos = ik_solver.compute_target_action(target_cube_pose, current_qpos)
                current_qpos = current_qpos[:, :7]

                if target_qpos is None:
                    print("Failed to compute target action")
                    break
                
                ik_steps = 40
                actions = torch.zeros((ik_steps, 8), dtype=torch.float32, device=policy.device)
                for i in range(ik_steps - 2):
                    actions[i, :7] = (current_qpos[0] * (ik_steps - 3 - i) + target_qpos[0] * (i + 1)) / (ik_steps - 2)
                    actions[i, 7] = 0.0
                
                actions[-2, :7] = target_qpos[0]
                actions[-2, 7] = 1.0
                actions[-1] = actions[-2]

                for idx in range(actions.shape[0]):
                    action = actions[idx]

                    obs_image_array.append(img)
                    proprio_array.append(obs['agent']['qpos'])
                    action_array.append(action)

                    correction_image_array.append(img)
                    correction_proprio_array.append(obs['agent']['qpos'])
                    correction_action_array.append(action)
                    correction_l_action_array.append(get_policy_actions(policy, proprio, obs_window, text_embed))

                    obs, reward, terminated, truncated, info = env.step(action)
                    img = env.render().squeeze(0).detach().cpu().numpy()
                    obs_window.append(img)
                    proprio = obs['agent']['qpos'][:, :-1]
                    video_frames.append(img)
                    global_steps += 1
                
                if terminated or truncated:
                    assert "success" in info, sorted(info.keys())
                    if info['success']:
                        success_count += 1
                        is_success = 1
                        done = True

                if is_success:
                    # Save the pair data
                    correction_process_count += 1
                    save_data(correction_data_save_path, 
                              correction_process_count, 
                              correction_image_array, 
                              correction_proprio_array, 
                              correction_action_array, 
                              correction_l_action_array)
                    break

            image_arrs = []
            for window_img in obs_window:
                image_arrs.append(window_img)
                image_arrs.append(None)
                image_arrs.append(None)
            images = [Image.fromarray(arr) if arr is not None else None
                    for arr in image_arrs]
            actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
            # Take 8 steps since RDT is trained to predict interpolated 64 steps(actual 16 steps)
            actions = actions[::4, :]
            for idx in range(actions.shape[0]):
                action = actions[idx]

                obs_image_array.append(img)
                proprio_array.append(obs['agent']['qpos'])
                action_array.append(action)

                obs, reward, terminated, truncated, info = env.step(action)
                img = env.render().squeeze(0).detach().cpu().numpy()
                obs_window.append(img)
                proprio = obs['agent']['qpos'][:, :-1]
                video_frames.append(img)
                global_steps += 1
                if terminated or truncated:
                    assert "success" in info, sorted(info.keys())
                    if info['success']:
                        success_count += 1
                        is_success = 1
                        done = True
                        break 
            
            condition_check_steps -= 1
            if condition_check_steps <= 0:
                reach_target = check_target(info, condition_flag)
                if not reach_target:
                    if condition_flag == 2 and info["is_cubeA_grasped"]:
                        condition_flag = 2
                        condition_check_steps = 4
                    else:
                        condition_flag = 1
                        condition_check_steps = 2
                        do_ik = True
        
        print(f"Trial {episode+1} finished, success: {info['success']}, steps: {global_steps}, is_correction: {is_correction}, is_success: {is_success}")
        use_correction_count += is_correction
        correction_success_count += is_success * is_correction

        if is_success == 1 and is_correction == 0:
            normal_success_demo_count += 1
        elif is_success == 1 and is_correction == 1:
            correction_success_demo_count += 1
            save_mp4(
                f"{correction_success_render_path}/{correction_success_demo_count}.mp4",
                video_frames,
                fps=30,
            )
            obs_image_array = obs_image_array[last_correction_index:]
            proprio_array = proprio_array[last_correction_index:]
            action_array = action_array[last_correction_index:]

    success_rate = success_count / total_episodes * 100
    print(f"Success rate: {success_rate}%")
    print(f"Correction rate: {use_correction_count / total_episodes * 100}%")
    print(f"Correction success rate: {correction_success_count / use_correction_count * 100}%")
    env.close()



if __name__=="__main__":
    args = parse_args()
    seed = args.random_seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)

    # TODO:
    # 1. how to load and save pair data?
    # 2. compute: execute - detect - correction (plan with policy output, as pair data) to end
    # 3. save pair data

    # How to save?
    # 