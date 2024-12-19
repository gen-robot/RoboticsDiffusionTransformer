import numpy as np


# v batch*n
def normalize_vector(v):
    v_mag = np.linalg.norm(v, axis=-1)
    v_mag = np.maximum(v_mag, 1e-8)
    v_mag = np.expand_dims(v_mag, axis=-1) # batch, 1
    v = v / v_mag
    return v


def quaternion_to_rotation_matrix(quaternion, mode='wxyz'):
    batch=quaternion.shape[0]
    quat = normalize_vector(quaternion)

    if mode == 'wxyz':
        qw = np.expand_dims(quat[...,0], axis=-1) # batch*1
        qx = np.expand_dims(quat[...,1], axis=-1)
        qy = np.expand_dims(quat[...,2], axis=-1)
        qz = np.expand_dims(quat[...,3], axis=-1)
    elif mode == 'xyzw':
        qw = np.expand_dims(quat[...,3], axis=-1)
        qx = np.expand_dims(quat[...,0], axis=-1)
        qy = np.expand_dims(quat[...,1], axis=-1)
        qz = np.expand_dims(quat[...,2], axis=-1)
    else:
        raise ValueError('mode not recognized')

    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    
    row0 = np.concatenate([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], axis=1) # batch*3
    row1 = np.concatenate([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], axis=1) # batch*3
    row2 = np.concatenate([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], axis=1) # batch*3
    matrix = np.stack([row0, row1, row2], axis=1) # batch*3*3

    return matrix


def rotation_matrix_to_ortho6d(matrix):
    ortho6d = matrix[..., :, :2]
    perm = list(range(len(ortho6d.shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    ortho6d = np.transpose(ortho6d, perm)
    ortho6d = np.reshape(ortho6d, ortho6d.shape[:-2] + (6,))
    return ortho6d


def quaternion_to_ortho6d(quaternion, mode='wxyz'):
    matrix = quaternion_to_rotation_matrix(quaternion, mode)
    ortho6d = rotation_matrix_to_ortho6d(matrix)
    return ortho6d