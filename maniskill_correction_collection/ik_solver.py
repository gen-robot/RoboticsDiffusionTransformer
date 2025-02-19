import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.utils.structs import Pose
from mani_skill.agents.controllers.utils.kinematics import Kinematics

class IKSolver(object):
    def __init__(self, articulation):
        self.urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2.urdf"
        
        self.kinematics = Kinematics(
            self.urdf_path,
            "panda_hand_tcp",
            articulation,
            torch.tensor([0, 1, 2, 3, 4, 5, 6]),
        )

        self.ee_link = self.kinematics.end_link
    
    def compute_target_pose(self, cube_pose):
        target_pose = cube_pose.clone()
        target_pose[0, 2] += 0.05 # z offset
        target_pose = Pose.create(target_pose)
        return target_pose

    def compute_strong_target_pose(self, cube_pose):
        target_pose = cube_pose.clone()
        target_pose[0, 2] -= 0.004 # z offset
        target_pose = Pose.create(target_pose)
        return target_pose

    def compute_release_target_pose(self, cube_pose):
        target_pose = cube_pose.clone()
        target_pose[0, 2] += 0.04
        target_pose = Pose.create(target_pose)
        return target_pose

    def compute_target_action(self, target_pose, init_qpos):
        target_qpos = self.kinematics.compute_ik(
            target_pose,
            init_qpos,
        )
        return target_qpos
        