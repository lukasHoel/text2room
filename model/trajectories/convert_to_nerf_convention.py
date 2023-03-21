import torch
import numpy as np
import json
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles


def parse_seen_poses(pose_file_path):
    with open(pose_file_path, "r") as f:
        poses = json.load(f)
        poses = {i: torch.from_numpy(np.array(p)).float() for i, p in poses.items()}
        return poses


def convert_pose_to_nerf_convention(world_to_cam: torch.Tensor) -> torch.Tensor:
    """
    Converts a world_to_cam matrix in pytorch3d convention to a cam_to_world matrix in the original nerf convention (e.g. as in nerf_synthetic dataset).

    We need camera orientation as in OpenGL/Blender which is:
    +X is right, +Y is up, and +Z is pointing back and away from the camera. -Z is the look-at direction.
    World Space also has a specific convetion which is:
    Our world space is oriented such that the up vector is +Z. The XY plane is parallel to the ground plane.
    See here for details: https://docs.nerf.studio/en/latest/quickstart/data_conventions.html

    We use pytorch3d convention which is:
    +X is left, +Y is up, and +Z is pointing front. +Z is the look-at direction.
    See here for details: https://pytorch3d.org/docs/cameras

    :param world_to_cam: [4,4] torch.Tensor
    :return: cam_to_world: [4,4] torch.Tensor
    """
    world_to_cam = world_to_cam.clone()

    # change rotation, XZ rotation must be flipped
    angles = matrix_to_euler_angles(world_to_cam[:3, :3], "XYZ")
    angles[0] = -angles[0]
    angles[2] = -angles[2]
    world_to_cam[:3, :3] = euler_angles_to_matrix(angles, "XYZ")

    # change translation, XZ translation must be flipped
    world_to_cam[0, 3] = -world_to_cam[0, 3]
    world_to_cam[2, 3] = -world_to_cam[2, 3]

    # have world_to_cam, need cam_to_world
    cam_to_world = world_to_cam.inverse()

    # xyz --> zxy. Want to have XY plane parallel to ground plane
    xyz_to_zxy = torch.tensor([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]).to(cam_to_world)
    
    return xyz_to_zxy @ cam_to_world


def convert_pose_from_nerf_convention(cam_to_world: torch.Tensor) -> torch.Tensor:
    """
    Converts a cam_to_world matrix in the original nerf convention (e.g. as in nerf_synthetic dataset) to a world_to_cam matrix in pytorch3d convention.

    We have camera orientation as in OpenGL/Blender which is:
    +X is right, +Y is up, and +Z is pointing back and away from the camera. -Z is the look-at direction.
    World Space also has a specific convetion which is:
    Our world space is oriented such that the up vector is +Z. The XY plane is parallel to the ground plane.
    See here for details: https://docs.nerf.studio/en/latest/quickstart/data_conventions.html

    We need pytorch3d convention which is:
    +X is left, +Y is up, and +Z is pointing front. +Z is the look-at direction.
    See here for details: https://pytorch3d.org/docs/cameras

    :param cam_to_world: [4,4] torch.Tensor or [3,4] torch.Tensor
    :return: world_to_cam: [4,4] torch.Tensor
    """
    cam_to_world = cam_to_world.clone()

    if cam_to_world.shape[0] == 3:
        cam_to_world = torch.cat([
            cam_to_world,
            torch.tensor([[0, 0, 0, 1]]).to(cam_to_world)
        ], dim=0)

    # zxy --> xyz
    zxy_to_xyz = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]).to(cam_to_world)
    cam_to_world = zxy_to_xyz @ cam_to_world

    # have cam_to_world, need world_to_cam
    world_to_cam = cam_to_world.inverse()

    # change translation, XZ translation must be flipped
    world_to_cam[0, 3] = -world_to_cam[0, 3]
    world_to_cam[2, 3] = -world_to_cam[2, 3]

    # change rotation, XZ rotation must be flipped
    angles = matrix_to_euler_angles(world_to_cam[:3, :3], "XYZ")
    angles[0] = -angles[0]
    angles[2] = -angles[2]
    world_to_cam[:3, :3] = euler_angles_to_matrix(angles, "XYZ")

    return world_to_cam
