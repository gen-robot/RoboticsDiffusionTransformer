import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
sys.path.append("/home/liangzhi/work-space/RoboTwin/")

from scripts.robotwin_model import create_model, RoboticDiffusionTransformerModel
from scripts.robotwin_adapter import RDTRobotwinAdapter
import importlib
import traceback
import yaml
from argparse import ArgumentParser
import torch
import math
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import imageio.v2 as iio
import transforms3d as t3d
import h5py

from robotwin_sim_test.eval_rdt_model import RDT, class_decorator
from robotwin_sim_test.outer_planner import (
    left_move_to_pose_with_screw,
    right_move_to_pose_with_screw,
    open_left_gripper,
    close_left_gripper,
    open_right_gripper,
    close_right_gripper,
    together_move_to_pose_with_screw
)

def step_action(Demo_class, model, step):
    actions = model.get_action()
    obs = model.get_last_obs()
    left_arm_actions , left_gripper , left_current_qpos, left_path = [], [], [], []
    right_arm_actions , right_gripper , right_current_qpos, right_path = [], [], [], []
    if Demo_class.dual_arm:
        left_arm_actions,left_gripper = actions[:, :6],actions[:, 6]
        right_arm_actions,right_gripper = actions[:, 7:13],actions[:, 13]
        left_current_qpos, right_current_qpos = obs['agent_pos'][:6], obs['agent_pos'][7:13]
    else:
        right_arm_actions,right_gripper = actions[:, :6],actions[:, 6]
        right_current_qpos = obs['agent_pos'][:6]
    
    if Demo_class.dual_arm:
        left_path = np.vstack((left_current_qpos, left_arm_actions))
    right_path = np.vstack((right_current_qpos, right_arm_actions))

    topp_left_flag, topp_right_flag = True, True
    try:
        times, left_pos, left_vel, acc, duration = Demo_class.left_planner.TOPP(left_path, 1/250, verbose=True)
        left_result = dict()
        left_result['position'], left_result['velocity'] = left_pos, left_vel
        left_n_step = left_result["position"].shape[0]
        left_gripper = np.linspace(left_gripper[0], left_gripper[-1], left_n_step)
    except:
        topp_left_flag = False
        left_n_step = 1
    
    if left_n_step == 0 or (not Demo_class.dual_arm):
        topp_left_flag = False
        left_n_step = 1

    try:
        times, right_pos, right_vel, acc, duration = Demo_class.right_planner.TOPP(right_path, 1/250, verbose=True)            
        right_result = dict()
        right_result['position'], right_result['velocity'] = right_pos, right_vel
        right_n_step = right_result["position"].shape[0]
        right_gripper = np.linspace(right_gripper[0], right_gripper[-1], right_n_step)
    except:
        topp_right_flag = False
        right_n_step = 1
    
    if right_n_step == 0:
        topp_right_flag = False
        right_n_step = 1
    
    n_step = max(left_n_step, right_n_step)
    obs_update_freq = n_step // actions.shape[0]

    now_left_id = 0 if topp_left_flag else 1e9
    now_right_id = 0 if topp_right_flag else 1e9
    i = 0
    success_flag = False
    
    while now_left_id < left_n_step or now_right_id < right_n_step:
        qf = Demo_class.robot.compute_passive_force(
            gravity=True, coriolis_and_centrifugal=True
        )
        Demo_class.robot.set_qf(qf)
        if topp_left_flag and now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step:
            for j in range(len(Demo_class.left_arm_joint_id)):
                left_j = Demo_class.left_arm_joint_id[j]
                Demo_class.active_joints[left_j].set_drive_target(left_result["position"][now_left_id][j])
                Demo_class.active_joints[left_j].set_drive_velocity_target(left_result["velocity"][now_left_id][j])
            if not Demo_class.fix_gripper:
                for joint in Demo_class.active_joints[34:36]:
                    # joint.set_drive_target(left_result["position"][i][6])
                    joint.set_drive_target(left_gripper[now_left_id])
                    joint.set_drive_velocity_target(0.05)
                    Demo_class.left_gripper_val = left_gripper[now_left_id]

            now_left_id +=1
            
        if topp_right_flag and now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step:
            for j in range(len(Demo_class.right_arm_joint_id)):
                right_j = Demo_class.right_arm_joint_id[j]
                Demo_class.active_joints[right_j].set_drive_target(right_result["position"][now_right_id][j])
                Demo_class.active_joints[right_j].set_drive_velocity_target(right_result["velocity"][now_right_id][j])
            if not Demo_class.fix_gripper:
                for joint in Demo_class.active_joints[36:38]:
                    # joint.set_drive_target(right_result["position"][i][6])
                    joint.set_drive_target(right_gripper[now_right_id])
                    joint.set_drive_velocity_target(0.05)
                    Demo_class.right_gripper_val = right_gripper[now_right_id]

            now_right_id +=1
        
        Demo_class.scene.step()
        Demo_class._update_render()
        Demo_class.episode_step += 1

        if i != 0 and i % obs_update_freq == 0:
            observation = Demo_class.get_obs()
            obs = Demo_class.get_cam_obs(observation)
            obs['agent_pos'] = observation['joint_action']

            model.update_obs(obs)
            Demo_class._take_picture()

        if Demo_class.episode_step % Demo_class.save_freq == 0:
            Demo_class._take_picture()
            render_observation = Demo_class.get_obs()
            render_obs = Demo_class.get_cam_obs(render_observation)
            Demo_class.render_array.append((render_obs["head_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            Demo_class.front_cam_array.append((obs["front_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            Demo_class.left_cam_array.append((obs["left_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            Demo_class.right_cam_array.append((obs["right_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            Demo_class.data_list.append(render_observation)

        if i % 5 == 0:
            Demo_class._update_render()
            if Demo_class.render_freq and i % Demo_class.render_freq == 0:
                Demo_class.viewer.render()
        
        i+=1

        if Demo_class.check_success():
            success_flag = True
            break

        if Demo_class.actor_pose == False:
            break
    
    return step + actions.shape[0], success_flag


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


def detection(Demo_class, model, step):
    ret_dict = {
        "do_correction": False,
        "correction_type": None,
    }
    if Demo_class.task_name == "shoe_place":
        pass
    elif Demo_class.task_name == "block_hammer_beat":
        pass
    elif Demo_class.task_name == "dual_bottles_pick_hard":
        pass
    elif Demo_class.task_name == "diverse_bottles_pick":
        pass
    elif Demo_class.task_name == "container_place":
        pass
    elif Demo_class.task_name == "empty_cup_place":
        if (step >= 200) and \
           (step - Demo_class.last_correction_index >= 50) and \
           (not (Demo_class.check_success() or \
                 Demo_class.cup.get_pose().p[2] > 0.82 or \
                 (abs(Demo_class.cup.get_pose().p[0] - Demo_class.coaster.get_pose().p[0]) < 0.025 and \
                  abs(Demo_class.cup.get_pose().p[1] - Demo_class.coaster.get_pose().p[1]) < 0.025))):
            print("Detected failure, do correction!")

            ret_dict["do_correction"] = True
            ret_dict["correction_type"] = "repick"

    elif Demo_class.task_name == "blocks_stack_easy":
        if (step >= 200) and \
           (step - Demo_class.last_correction_index >= 50) and \
           (not ((Demo_class.block1.get_pose().p[2] > 0.77) or \
                 ((abs(Demo_class.block1.get_pose().p[0] - 0) < 0.025) and \
                  (abs(Demo_class.block1.get_pose().p[1] + 0.1) < 0.025)))):
            print("Detected red cube failure, do correction!")
            # Meta Data
            ret_dict["do_correction"] = True
            ret_dict["correction_type"] = "repick_red"
        elif (step >= 450) and \
             (step - Demo_class.last_correction_index >= 50) and \
             (abs(Demo_class.block1.get_pose().p[0] - 0) < 0.025) and \
             (abs(Demo_class.block1.get_pose().p[1] + 0.1) < 0.025) and \
             (abs(Demo_class.block1.get_pose().p[2] - 0.765) < 0.01) and \
             (not ((Demo_class.block2.get_pose().p[2] > 0.77) or \
                   ((abs(Demo_class.block2.get_pose().p[0] - 0) < 0.025) and \
                    (abs(Demo_class.block2.get_pose().p[1] - 0.1) < 0.025)))):
            print("Detected black cube failure, do correction!")
            # Meta Data
            ret_dict["do_correction"] = True
            ret_dict["correction_type"] = "repick_black"
        
    else:
        raise NotImplementedError(f"Task {Demo_class.task_name} not implemented")

    return ret_dict


def correction_process(Demo_class, model: RDT, embeding_dict: dict, use_correction=False):

    render_dir = f"./outs/robotwin/{Demo_class.task_name}/"
    os.makedirs(render_dir, exist_ok=True)

    step = 0
    Demo_class.test_num += 1
    Demo_class.episode_step = 0
    Demo_class.last_correction_index = 0
    Demo_class.last_correction_data_index = 0
    
    success_flag = False
    do_correction = False
    correction_period = 0
    Demo_class._update_render()
    if Demo_class.render_freq:
        Demo_class.viewer.render()
    
    Demo_class.actor_pose = True

    Demo_class._take_picture()
    observation = Demo_class.get_obs()
    obs = Demo_class.get_cam_obs(observation)

    obs['agent_pos'] = observation['joint_action']
    model.update_obs(obs)

    Demo_class.render_array = []
    Demo_class.front_cam_array = []
    Demo_class.left_cam_array = []
    Demo_class.right_cam_array = []
    Demo_class.data_list = []
    # (c, h, w) -> (h, w, c)
    # print("head_cam:", obs["head_cam"], np.max(obs["head_cam"]), np.min(obs["head_cam"]))
    Demo_class.render_array.append((obs["head_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
    Demo_class.front_cam_array.append((obs["front_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
    Demo_class.left_cam_array.append((obs["left_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
    Demo_class.right_cam_array.append((obs["right_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
    Demo_class.data_list.append(observation)

    while step < Demo_class.step_lim + 600:

        if correction_period > 0:
            correction_period -= 1
        else:
            model.update_textembeding(embeding_dict["primary_instruction"])

        # Step a action chunk
        step, success_flag = step_action(Demo_class, model, step)
        correction_info = detection(Demo_class, model, step)

        if correction_info["do_correction"] and use_correction:
            do_correction = True
            model.update_textembeding(embeding_dict[correction_info["correction_type"]])
            correction_period = 8

        Demo_class._update_render()

        if Demo_class.render_freq:
            Demo_class.viewer.render()
        
        Demo_class._take_picture()

        print(f'step: {step} / {Demo_class.step_lim + 600}', end='\r')

        if success_flag:
            print("\nsuccess!")
            #self.success_record_list.append((self.test_num, 'success'))
            Demo_class.suc += 1

            save_mp4(f'{render_dir}/{save_head_index + Demo_class.test_num}.mp4', Demo_class.render_array)
            return
        
        if Demo_class.actor_pose == False:
            break
        continue

    print("\nfail!")
    save_mp4(f'{render_dir}/{save_head_index + Demo_class.test_num}.mp4', Demo_class.render_array)


def test_policy(task_name, Demo_class, args, agent: RDT, st_seed, test_num=200, 
                render=False, general=False, language_embeding_name=None):
    expert_check = True
    print('Task name: ', args['task_name'])

    Demo_class.suc = 0
    Demo_class.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    now_seed = st_seed

    suc_record_list = []
    prev_suc = 0

    text_embedding_root = '/data2/home/shanzh/RoboticsDiffusionTransformer/gpt_prompt/precomputedemb/'

    while succ_seed < test_num:
        render_freq = args['render_freq']

        if expert_check:
            try:
                Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
                Demo_class.play_once()
                Demo_class.close()
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(' -------------')
                print('Error: ', stack_trace)
                print(' -------------')
                Demo_class.close()
                now_seed += 1
                args['render_freq'] = render_freq
                print('error occurs !')
                continue
        if (not expert_check) or ( Demo_class.plan_success and Demo_class.check_success() ):
            succ_seed +=1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args['render_freq'] = render_freq
            continue

        args['render_freq'] = render_freq

        Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, **args)
        #task_name = args['task_name']
        task_detail = Demo_class.task_detail()
        if general:
            task_detail = 'general'
        num_lang_ins = len(os.listdir(os.path.join(text_embedding_root, task_name, task_detail)))
        text_embedding_dir = os.path.join(text_embedding_root, task_name, 
                                          task_detail, f'lang_embed_{np.random.randint(num_lang_ins)}.pt')
        if language_embeding_name:
            text_embedding_dir = language_embeding_name
        agent.update_textembeding(text_embedding_dir)

        embeding_dict = {
            'task_name': task_name,
            'primary_instruction': text_embedding_dir,
        }

        if task_name == "blocks_stack_easy":
            num_lang_ins_rb = len(os.listdir(os.path.join(text_embedding_root, 'blocks_stack_easy_part_repick_black', 'general')))
            text_embedding_dir_rb = os.path.join(text_embedding_root, 'blocks_stack_easy_part_repick_black',
                                                 'general', f'lang_embed_{np.random.randint(num_lang_ins_rb)}.pt')
            embeding_dict['repick_black'] = text_embedding_dir_rb

            num_lang_ins_rr = len(os.listdir(os.path.join(text_embedding_root, 'blocks_stack_easy_part_repick_red', 'general')))
            text_embedding_dir_rr = os.path.join(text_embedding_root, 'blocks_stack_easy_part_repick_red',
                                                 'general', f'lang_embed_{np.random.randint(num_lang_ins_rr)}.pt')
            embeding_dict['repick_red'] = text_embedding_dir_rr
        if task_name == "empty_cup_place":
            num_lang_ins = len(os.listdir(os.path.join(text_embedding_root, 'empty_cup_place_part_repick', 'general')))
            text_embedding_dir = os.path.join(text_embedding_root, 'empty_cup_place_part_repick',
                                              'general', f'lang_embed_{np.random.randint(num_lang_ins)}.pt')
            embeding_dict['repick'] = text_embedding_dir

        correction_process(Demo_class, agent, embeding_dict, usr_args.correction)

        now_id += 1
        Demo_class.close()
        if Demo_class.render_freq:
            Demo_class.viewer.close()
        agent.runner.reset_obs()
        print(f"{task_name} success rate: {Demo_class.suc}/{Demo_class.test_num}, current seed: {now_seed}\n")
        if Demo_class.suc > prev_suc:
            #prev_suc = Demo_class.suc
            suc_record_list.append(Demo_class.test_num)
        prev_suc = Demo_class.suc
        Demo_class._take_picture()
        now_seed += 1
    
    return now_seed, Demo_class.suc


def main(usr_args):
    task_name = usr_args.task_name
    seed = usr_args.seed

    with open(f'../RoboTwin/task_config/{task_name}.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f, Loader=yaml.FullLoader) 
    
    def get_camera_config(camera_type):
        camera_config_path = '../RoboTwin/task_config/_camera_config.yml'
        assert os.path.isfile(camera_config_path), "task config file is missing"
        with open(camera_config_path, 'r', encoding='utf-8') as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
        assert camera_type in args, f'camera {camera_type} is not defined'
        return args[camera_type]
    
    head_camera_config = get_camera_config(args['head_camera_type'])
    args['head_camera_fovy'] = head_camera_config['fovy']
    args['head_camera_w'] = head_camera_config['w']
    args['head_camera_h'] = head_camera_config['h']

    wrist_camera_config = get_camera_config(args['wrist_camera_type'])
    args['wrist_camera_fovy'] = wrist_camera_config['fovy']
    args['wrist_camera_w'] = wrist_camera_config['w']
    args['wrist_camera_h'] = wrist_camera_config['h']

    front_camera_config = get_camera_config(args['front_camera_type'])
    args['front_camera_fovy'] = front_camera_config['fovy']
    args['front_camera_w'] = front_camera_config['w']
    args['front_camera_h'] = front_camera_config['h']

    args['render_freq'] = usr_args.render_freq
    args['save_freq'] = usr_args.save_freq

    task = class_decorator(task_name)

    st_seed = 100000 * (seed + 1)
    suc_nums = []
    test_num = 1000 
    global save_head_index
    save_head_index = st_seed
    topk = 1

    rdt_args = usr_args
    #test_embeding_dir = './data/sz/' + usr_args.task_name + '/lang_embed_0.pt'
    agent = RDT(rdt_args, device=usr_args.device)

    st_seed, suc_num = test_policy(task_name, task, args, agent, st_seed, 
                                   test_num=test_num, render=usr_args.render, general=True, 
                                   language_embeding_name=usr_args.language_embeding_name)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]
    save_dir = Path(f'correction_collection/result_rdt/{task_name}_{seed}')
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f'result_ckpt.txt'
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # TODO： 整理结果
    with open(file_path, 'a') as file:
        file.write(f'\nTimestamp: {current_time}\n\n')
        file.write(f'Successful Rate of current checkpoints:\n Checkpoints: {rdt_args.pretrained_model_name_or_path}\n')
        file.write('\n'.join(map(str, np.array(suc_nums) / test_num)))

    print(f'Data has been saved to {file_path}')


if __name__ == '__main__':
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
                        default='/nvme_data/liangzhi/rdt/checkpoints/rdt-robotwin-finetune-lr1e-4-typeonly_correction-20250117-142131/checkpoint-5000', 
                        help='rdt model check points dir')
    parser.add_argument('--ctrl_freq', type=int, default=15, help='Control frequency')
    parser.add_argument('--eval_episodes', type=int, default=100, help='Number of episodes to evaluate')
    parser.add_argument('--general_ins', type=bool, default=True, help='General instruction flag')
    parser.add_argument('--language_embeding_name', type=str, default=None, help='Language embedding name')
    
    usr_args = parser.parse_args()

    main(usr_args)