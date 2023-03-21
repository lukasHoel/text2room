import torch
import numpy as np


def sample_noise(n, r_max, t_max):
    nr = np.random.normal(0, scale=r_max/2.0, size=(n,3))
    nr = np.clip(nr, a_min=-r_max, a_max=r_max)

    nt = np.random.normal(0, scale=t_max/2.0, size=(n,3))
    nt = np.clip(nt, a_min=-t_max, a_max=t_max)

    return nr, nt


def interpolate_noise(n, steps):
    last = np.linspace(n[-1], n[-1], num=steps)
    n = [np.linspace(n[i], n[i + 1], num=steps) for i in range(n.shape[0] - 1)]
    n.append(last)
    n = np.concatenate(n, axis=0)
    return n


def to_degrees(x):
    return x * 180.0 / np.pi


def to_radians(x):
    return x * np.pi / 180.0


# Checks if a matrix is a valid rotation matrix.
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-5


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def apply_noise(poses, chunk_size=10, r_max=5.0, t_max=0.05):
    noisy_poses = []

    # create noise vectors
    n = len(poses) // chunk_size + (len(poses) % chunk_size != 0)
    nr, nt = sample_noise(n, r_max, t_max)
    nr = interpolate_noise(nr, chunk_size)
    nt = interpolate_noise(nt, chunk_size)

    for i, p in enumerate(poses):
        if isinstance(p, torch.Tensor):
            pose_numpy = p.cpu().detach().numpy()
        else:
            pose_numpy = p

        # extract r, t
        r = pose_numpy[:3, :3]
        r = rotationMatrixToEulerAngles(r)
        r = to_degrees(r)
        t = pose_numpy[:3, 3]

        # get noise
        nr_i = nr[i // chunk_size]
        nt_i = nt[i // chunk_size]

        # apply noise
        r += nr_i
        t += nt_i

        # create pose noise
        r = to_radians(r)
        r = eulerAnglesToRotationMatrix(r)
        p_noise = np.eye(4, dtype=np.float32)
        p_noise[:3, :3] = r
        p_noise[:3, 3] = t

        if isinstance(p, torch.Tensor):
            p_noise = torch.from_numpy(p_noise).to(p)

        noisy_poses.append(p_noise)

    return noisy_poses
