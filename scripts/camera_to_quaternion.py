import numpy as np
from scipy.spatial.transform import Rotation as R

def camera_to_quaternion(camera_pos, target, up):
    forward = np.array(target, dtype=np.float64) - np.array(camera_pos, dtype=np.float64)
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)

    rotation_matrix = np.column_stack((right, true_up, forward))
    quat = R.from_matrix(rotation_matrix).as_quat()
    return quat
