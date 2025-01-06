import numpy as np
import torch

def quaternion_to_rot_matrix(quat):
    w, x, y, z = quat

    rot_matrix = torch.tensor([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ], dtype=torch.float32)
    
    return rot_matrix

def get_direction_vector(rot_matrix):
    v_x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    v_y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    v_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    x_vec = torch.matmul(rot_matrix, v_x)
    y_vec = torch.matmul(rot_matrix, v_y)
    z_vec = torch.matmul(rot_matrix, v_z)

    return x_vec, y_vec, z_vec

def get_direction_vector_from_quat(quat):
    rot_matrix = quaternion_to_rot_matrix(quat)
    return get_direction_vector(rot_matrix)