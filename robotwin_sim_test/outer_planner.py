import os
import sys
import torch
import numpy as np


def left_follow_path(self, result, model, save_freq=-1): # For left arm
    save_freq = self.save_freq if save_freq == -1 else save_freq
    n_step = result["position"].shape[0]

    if n_step > 2000:
        self.plan_success = False
        return

    if save_freq != None:
        self._take_picture()

    for i in range(n_step):
        qf = self.robot.compute_passive_force(
            gravity=True, coriolis_and_centrifugal=True
        )
        self.robot.set_qf(qf)
        for j in range(len(self.left_arm_joint_id)):
            n_j = self.left_arm_joint_id[j]
            self.active_joints[n_j].set_drive_target(result["position"][i][j])
            self.active_joints[n_j].set_drive_velocity_target(
                result["velocity"][i][j]
            )
        self.scene.step()
        self.episode_step += 1
        if i%5 == 0:
            self._update_render()
            if self.render_freq and i % self.render_freq == 0:
                self.viewer.render()
        
        if self.episode_step % self.save_freq == 0:
            self._take_picture()
            render_observation = self.get_obs()
            render_obs = self.get_cam_obs(render_observation)
            self.render_array.append((render_obs["head_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.front_cam_array.append((render_obs["front_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.left_cam_array.append((render_obs["left_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.right_cam_array.append((render_obs["right_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.data_list.append(render_observation)

            render_obs['agent_pos'] = render_observation['joint_action']
            model.update_obs(render_obs)

    if save_freq != None:
        self._take_picture()

def right_follow_path(self, result, model, save_freq=-1):
    save_freq = self.save_freq if save_freq == -1 else save_freq
    n_step = result["position"].shape[0]

    if n_step > 2000:
        self.plan_success = False
        return
    
    if save_freq != None:
        self._take_picture()

    for i in range(n_step):
        qf = self.robot.compute_passive_force(
            gravity=True, coriolis_and_centrifugal=True
        )
        self.robot.set_qf(qf)
        for j in range(len(self.right_arm_joint_id)):
            n_j = self.right_arm_joint_id[j]
            self.active_joints[n_j].set_drive_target(result["position"][i][j])
            self.active_joints[n_j].set_drive_velocity_target(
                result["velocity"][i][j]
            )
        
        self.scene.step()
        self.episode_step += 1
        if i % 5 == 0:
            self._update_render()
            if self.render_freq and i % self.render_freq == 0:
                self.viewer.render()

        if self.episode_step % self.save_freq == 0:
            self._take_picture()
            render_observation = self.get_obs()
            render_obs = self.get_cam_obs(render_observation)
            self.render_array.append((render_obs["head_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.front_cam_array.append((render_obs["front_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.left_cam_array.append((render_obs["left_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.right_cam_array.append((render_obs["right_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.data_list.append(render_observation)

            render_obs['agent_pos'] = render_observation['joint_action']
            model.update_obs(render_obs)

    if save_freq != None:
        self._take_picture()

def together_follow_path(self, 
                         left_result,
                         right_result, 
                         model,
                         save_freq=-1):
        save_freq = self.save_freq if save_freq == -1 else save_freq
        left_n_step = left_result["position"].shape[0]
        right_n_step = right_result["position"].shape[0]
        n_step = max(left_n_step, right_n_step)

        if n_step > 2000:
            self.plan_success = False
            return

        if save_freq != None:
            self._take_picture()

        now_left_id = 0
        now_right_id = 0
        i = 0

        while now_left_id < left_n_step or now_right_id < right_n_step:
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            # set the joint positions and velocities for move group joints only.
            # The others are not the responsibility of the planner
            if now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step:
                for j in range(len(self.left_arm_joint_id)):
                    left_j = self.left_arm_joint_id[j]
                    self.active_joints[left_j].set_drive_target(left_result["position"][now_left_id][j])
                    self.active_joints[left_j].set_drive_velocity_target(left_result["velocity"][now_left_id][j])
                now_left_id +=1
                
            if now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step:
                for j in range(len(self.right_arm_joint_id)):
                    right_j = self.right_arm_joint_id[j]
                    self.active_joints[right_j].set_drive_target(right_result["position"][now_right_id][j])
                    self.active_joints[right_j].set_drive_velocity_target(right_result["velocity"][now_right_id][j])
                now_right_id +=1

            self.scene.step()
            self.episode_step += 1
            if i % 5==0:
                self._update_render()
                if self.render_freq and i % self.render_freq == 0:
                    self.viewer.render()

            if self.episode_step % self.save_freq == 0:
                self._take_picture()
                render_observation = self.get_obs()
                render_obs = self.get_cam_obs(render_observation)
                self.render_array.append((render_obs["head_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
                self.front_cam_array.append((render_obs["front_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
                self.left_cam_array.append((render_obs["left_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
                self.right_cam_array.append((render_obs["right_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
                self.data_list.append(render_observation)

                render_obs['agent_pos'] = render_observation['joint_action']
                model.update_obs(render_obs)
            i+=1

        if save_freq != None:
            self._take_picture()

def left_move_to_pose_with_screw(self, pose, model, use_point_cloud=False, use_attach=False, save_freq=-1):
    """
    Interpolative planning with screw motion.
    Will not avoid collision and will fail if the path contains collision.
    """
    save_freq = self.save_freq if save_freq == -1 else save_freq
    # joint_pose = self.robot.get_qpos()
    joint_position = self.get_obs()['joint_action']
    qpos=[]
    for i in range(10):
        qpos.append(0)
    for i in range(6):
        qpos.append(joint_position[i])
    for i in range(42 - 16):
        qpos.append(0)
    qpos = np.array(qpos)
    # print("Start qpos:", qpos)

    result = self.left_planner.plan_screw(
        target_pose=pose,
        qpos=qpos,
        time_step=1 / 250,
        use_point_cloud=use_point_cloud,
        use_attach=use_attach,
    )
    
    # print("result:", result["position"])

    if result["status"] == "Success":
        left_follow_path(self, result, model, save_freq=save_freq)
        return 0
    else:
        print("\n left arm palnning failed!")
        self.plan_success = False

def right_move_to_pose_with_screw(self, pose, model, use_point_cloud=False, use_attach=False, save_freq=-1):
    """
    Interpolative planning with screw motion.
    Will not avoid collision and will fail if the path contains collision.
    """
    save_freq = self.save_freq if save_freq == -1 else save_freq
    joint_position = self.get_obs()['joint_action']
    qpos=[]
    for i in range(18):
        qpos.append(0)
    for i in range(6):
        qpos.append(joint_position[i + 7])
    for i in range(42 - 24):
        qpos.append(0)
    qpos = np.array(qpos)
    result = self.right_planner.plan_screw(
        target_pose=pose,
        qpos=qpos,
        time_step=1 / 250,
        use_point_cloud=use_point_cloud,
        use_attach=use_attach,
    )
    
    if result["status"] == "Success":
        right_follow_path(self, result, model, save_freq=save_freq)
        return 0
    else:
        print("\n right arm palnning failed!")
        self.plan_success = False

def together_move_to_pose_with_screw(self, 
                                     left_target_pose,
                                     right_target_pose, 
                                     model,
                                     use_point_cloud=False, 
                                     use_attach=False,
                                     save_freq=-1):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        save_freq = self.save_freq if save_freq == -1 else save_freq
        joint_position = self.get_obs()['joint_action']

        left_qpos = []
        for i in range(10):
            left_qpos.append(0)
        for i in range(6):
            left_qpos.append(joint_position[i])
        for i in range(42 - 16):
            left_qpos.append(0)
        left_qpos = np.array(left_qpos)

        right_qpos = []
        for i in range(18):
            right_qpos.append(0)
        for i in range(6):
            right_qpos.append(joint_position[i + 7])
        for i in range(42 - 24):
            right_qpos.append(0)
        right_qpos = np.array(right_qpos)

        left_result = self.left_planner.plan_screw(
            target_pose=left_target_pose,
            qpos=left_qpos,
            time_step=1 / 250,
            use_point_cloud=use_point_cloud,
            use_attach=use_attach,
        )

        right_result = self.right_planner.plan_screw(
            target_pose=right_target_pose,
            qpos=right_qpos,
            time_step=1 / 250,
            use_point_cloud=use_point_cloud,
            use_attach=use_attach,
        )

        if left_result["status"] == "Success" and right_result["status"] == "Success":
            together_follow_path(self, left_result, right_result, model, save_freq=save_freq)
            return 0
        else:
            if left_result["status"] != "Success":
                print("\n left arm palnning failed!")
            if right_result["status"] != "Success":
                print("\n right arm palnning failed!")
            self.plan_success = False

def set_gripper(self, model, left_pos = 0.045, right_pos = 0.045, set_tag = 'together', save_freq=-1):
    '''
        Set gripper posture
        - `left_pos`: Left gripper pose
        - `right_pos`: Right gripper pose
        - `set_tag`: "left" to set the left gripper, "right" to set the right gripper, "together" to set both grippers simultaneously.
    '''
    save_freq = self.save_freq if save_freq == -1 else save_freq
    if save_freq != None:
        self._take_picture()
    
    left_gripper_step = 0
    right_gripper_step = 0
    real_left_gripper_step = 0
    real_right_gripper_step = 0

    if set_tag == 'left' or set_tag == 'together':
        left_gripper_step = (left_pos - self.left_gripper_val) / 400
        real_left_gripper_step = (left_pos - self.active_joints[34].get_drive_target()[0]) / 200

    if set_tag == 'right' or set_tag == 'together':
        right_gripper_step = (right_pos - self.right_gripper_val) / 400
        real_right_gripper_step = (right_pos - self.active_joints[36].get_drive_target()[0]) / 200
    
    for i in range(400):
        self.left_gripper_val +=  left_gripper_step
        self.right_gripper_val +=  right_gripper_step
        if i < 200:
            real_left_gripper_val = self.active_joints[34].get_drive_target()[0] + real_left_gripper_step
            real_right_gripper_val = self.active_joints[36].get_drive_target()[0] + real_right_gripper_step

        qf = self.robot.compute_passive_force(
            gravity=True, coriolis_and_centrifugal=True
        )
        self.robot.set_qf(qf)
        
        if set_tag == 'left' or set_tag == 'together':
            for joint in self.active_joints[34:36]:
                joint.set_drive_target(real_left_gripper_val)
                joint.set_drive_velocity_target(0.05)

        if set_tag == 'right' or set_tag == 'together':
            for joint in self.active_joints[36:38]:
                joint.set_drive_target(real_right_gripper_val)
                joint.set_drive_velocity_target(0.05)

        self.scene.step()
        self.episode_step += 1
        if i % 5==0:
            self._update_render()
            if self.render_freq and i % self.render_freq == 0:
                self.viewer.render()

        if self.episode_step % self.save_freq == 0:
            self._take_picture()
            render_observation = self.get_obs()
            render_obs = self.get_cam_obs(render_observation)
            self.render_array.append((render_obs["head_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.front_cam_array.append((render_obs["front_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.left_cam_array.append((render_obs["left_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.right_cam_array.append((render_obs["right_cam"].transpose(1, 2, 0) * 255).astype(np.uint8))
            self.data_list.append(render_observation)

            render_obs['agent_pos'] = render_observation['joint_action']
            model.update_obs(render_obs)

    if save_freq != None:
        self._take_picture()
    
    if set_tag == 'left' or set_tag == 'together':
        self.left_gripper_val = left_pos
    if set_tag == 'right' or set_tag == 'together':
        self.right_gripper_val = right_pos

def open_left_gripper(self, model, save_freq=-1, pos = 0.045):
    save_freq = self.save_freq if save_freq == -1 else save_freq
    set_gripper(self, model, left_pos = pos, set_tag='left', save_freq=save_freq)

def close_left_gripper(self, model, save_freq=-1, pos = 0):
    save_freq = self.save_freq if save_freq == -1 else save_freq
    set_gripper(self, model, left_pos = pos, set_tag='left',save_freq=save_freq)

def open_right_gripper(self, model, save_freq=-1,pos = 0.045):
    save_freq = self.save_freq if save_freq == -1 else save_freq
    set_gripper(self, model, right_pos=pos, set_tag='right', save_freq=save_freq)

def close_right_gripper(self, model, save_freq=-1,pos = 0):
    save_freq = self.save_freq if save_freq == -1 else save_freq
    set_gripper(self, model, right_pos=pos, set_tag='right', save_freq=save_freq)