import numpy as np
import torch
from model.mesh_fusion.util import get_extrinsics


#######################
# PRIVATE HELPERS #
#######################

def _rot_x(x):
    '''
    positive: look up, negative: look down

    :param x: rotation in degrees
    '''
    return np.array([x * np.pi / 180, 0, 0], dtype=np.float32)


def _rot_y(x):
    '''
    positive: look right, negative: look left

    :param x: rotation in degrees
    '''
    return np.array([0, x * np.pi / 180, 0], dtype=np.float32)


def _rot_z(x):
    '''
    positive: tilt left, negative: tilt right

    :param x: rotation in degrees
    '''
    return np.array([0, 0, x * np.pi / 180], dtype=np.float32)


def _trans_x(x):
    '''
    positive: right, negative: left

    :param x: translation amount
    '''
    return np.array([x, 0, 0], dtype=np.float32)


def _trans_y(x):
    '''
    positive: down, negative: up

    :param x: translation amount
    '''
    return np.array([0, x, 0], dtype=np.float32)


def _trans_z(x):
    '''
    positive: back, negative: front

    :param x: translation amount
    '''
    return np.array([0, 0, x], dtype=np.float32)


def _config_fn(fn, **kwargs):
    return lambda i, steps: fn(i, steps, **kwargs)


def _circle(i, steps=60, txmax=0, txmin=0, tymax=0, tymin=0, tzmax=0, tzmin=0, rxmax=0, rxmin=0, rymax=0, rymin=0, rzmax=0, rzmin=0):
    tx_delta = (txmax - txmin) / (steps // 2)
    ty_delta = (tymax - tymin) / (steps // 2)
    tz_delta = (tzmax - tzmin) / (steps // 2)

    rx_delta = (rxmax - rxmin) / (steps // 2)
    ry_delta = (rymax - rymin) / (steps // 2)
    rz_delta = (rzmax - rzmin) / (steps // 2)

    f = i % (steps // 2)

    tx = txmin + f * tx_delta
    ty = tymin + f * ty_delta
    tz = tzmin + f * tz_delta

    rx = rxmin + f * rx_delta
    ry = rymin + f * ry_delta
    rz = rzmin + f * rz_delta

    if i < steps // 2:
        T = _trans_x(-tx)
        T += _trans_z(tz)
        T += _trans_y(ty)
        R = _rot_y(ry)
        R += _rot_x(rx)
        R += _rot_z(rz)
    else:
        T = _trans_x(tx)
        T += _trans_z(tz)
        T += _trans_y(-ty)
        R = _rot_y(-ry)
        R += _rot_x(-rx)
        R += _rot_z(-rz)

    return get_extrinsics(R, T)


def _rot_left(i, steps=60, ty=0, rx=0):
    angle = i * 360 // steps

    T = _trans_x(0)
    T += _trans_y(ty)
    R = _rot_y(-angle)
    R += _rot_x(rx)

    return get_extrinsics(R, T)


def _back_and_forth(i, steps=20, txmax=0, txmin=0, tymax=0, tymin=0, tzmax=0, tzmin=0, rxmax=0, rxmin=0, rymax=0, rymin=0, rzmax=0, rzmin=0):
    tx_delta = (txmax - txmin) / (steps // 2)
    ty_delta = (tymax - tymin) / (steps // 2)
    tz_delta = (tzmax - tzmin) / (steps // 2)

    rx_delta = (rxmax - rxmin) / (steps // 2)
    ry_delta = (rymax - rymin) / (steps // 2)
    rz_delta = (rzmax - rzmin) / (steps // 2)

    f = i % (steps // 2)

    tx = txmin + f * tx_delta
    ty = tymin + f * ty_delta
    tz = tzmin + f * tz_delta

    rx = rxmin + f * rx_delta
    ry = rymin + f * ry_delta
    rz = rzmin + f * rz_delta

    if i < steps // 2:
        T = _trans_x(-tx)
        T += _trans_z(tz)
        T += _trans_y(ty)
        R = _rot_y(-ry)
        R += _rot_x(rx)
        R += _rot_z(rz)
    else:
        T = _trans_x(tx)
        T += _trans_z(tz)
        T += _trans_y(-ty)
        R = _rot_y(ry)
        R += _rot_x(-rx)
        R += _rot_z(-rz)

    return get_extrinsics(R, T)


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    # c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    c2w[0:3, 3] = -1 * c2w[0:3, 3]
    return c2w


# def sample_points_on_unit_sphere(n_sample, radius=4.0, height=0.0):
#     c2w = torch.stack([pose_spherical(angle, 0, radius) for angle in np.linspace(-180,180,n_sample)], dim=0)
#     c2w[:, 1, 3] = height
#     return c2w

# def spherical_trajector(steps, radius, height):
#     c2w = sample_points_on_unit_sphere(n_sample, radius, height)
#     return torch.inverse(c2w)


def _sphere_rot_xz(i, steps, radius=4.0, height=0.0, phi=20.0):
    rot_angle = i * 360 / steps
    rot_angle = rot_angle
    phi = phi if height > 0 else -phi if height < 0 else 0
    c2w = pose_spherical(rot_angle, phi, radius)
    c2w[1, 3] = height
    return torch.inverse(c2w)

def _double_sphere_rot_xz(i, steps, radius=4.0, height=0.0, phi=20.0):
    rot_angle = i * 360 / steps
    rot_angle = rot_angle
    phi = phi if height > 0 else -phi if height < 0 else 0
    c2w = pose_spherical(rot_angle, phi, radius)
    if i % 2 == 0:
        c2w[0:3, 3] = 0
    else:
        c2w[1, 3] = height
    return torch.inverse(c2w)

#######################
# PUBLIC TRAJECTORIES #
#######################


def forward(height=0, rot=0, txmax=2):
    return _config_fn(_circle, txmax=txmax, rymax=90, rymin=45, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def forward_small(height=0, rot=0):
    return _config_fn(_circle, txmax=0.5, rymax=90, rymin=45, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def left_right(height=0, rot=0):
    return _config_fn(_circle, tzmax=2, rymax=90, rymin=45, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def backward(height=0, rot=0):
    return _config_fn(_circle, txmax=-2, rymax=90, rymin=45, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def backward2(height=0, rot=0, txmax=1):
    return _config_fn(_circle, txmax=txmax, rymax=225, rymin=180, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def backward2_small(height=0, rot=0):
    return _config_fn(_circle, txmax=0.5, rymax=225, rymin=180, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def backward3(height=0, rot=0, txmax=1):
    return _config_fn(_circle, txmax=txmax, rymax=270, rymin=225, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def backward3_small(height=0, rot=0):
    return _config_fn(_circle, txmax=0.5, rymax=270, rymin=225, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def rot_left_up_down(height=0, rot=0):
    return _config_fn(_rot_left, ty=height, rx=rot)


def back_and_forth_forward(height=0, rot=0):
    return _config_fn(_back_and_forth, txmax=0, tzmax=-2, rymax=60, rymin=15, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_backward(height=0, rot=0):
    return _config_fn(_back_and_forth, txmax=0, tzmax=-2, rymax=-120, rymin=-165, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_forward_reverse(height=0, rot=0, tzmax=2):
    return _config_fn(_back_and_forth, txmax=0, tzmax=tzmax, rymax=-15, rymin=-60, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_forward_reverse_small(height=0, rot=0):
    return _config_fn(_back_and_forth, txmax=0, tzmax=0.5, rymax=-15, rymin=-60, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_backward_reverse(height=0, rot=0, tzmax=2):
    return _config_fn(_back_and_forth, txmax=0, tzmax=tzmax, rymax=165, rymin=120, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_backward_reverse_small(height=0, rot=0):
    return _config_fn(_back_and_forth, txmax=0, tzmax=0.5, rymax=165, rymin=120, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_backward_reverse2(height=0, rot=0):
    return _config_fn(_back_and_forth, txmax=0, tzmax=3, rymax=165, rymin=60, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def sphere_rot(radius=4.0, height=0.0, phi=20.0):
    return _config_fn(_sphere_rot_xz, radius=radius, height=height, phi=phi)


def double_rot(radius=4.0, height=0.0, phi=2.0):
    return _config_fn(_double_sphere_rot_xz, radius=radius, height=height, phi=phi)
