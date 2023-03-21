import numpy as np
import torch

from copy import deepcopy


def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model


# visualization functions
__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
def unnormalize(img_in):
    img_out = np.zeros(img_in.shape)
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] += __imagenet_stats['mean'][ich]
    img_out = (img_out * 255).astype(np.uint8)
    return img_out


def normal_to_rgb(norm):
    norm_rgb = ((norm + 1) * 0.5) * 255
    norm_rgb = np.clip(norm_rgb, a_min=0, a_max=255)
    norm_rgb = norm_rgb.astype(np.uint8)  # (B, H, W, 3)
    return norm_rgb


def kappa_to_alpha(pred_kappa):
    alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
            + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
    alpha_deg = np.degrees(alpha)
    return alpha, alpha_deg


def kappa_to_alpha_torch(pred_kappa):
    alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
            + ((torch.exp(- pred_kappa * torch.pi) * torch.pi) / (1 + torch.exp(- pred_kappa * torch.pi)))
    alpha_deg = torch.rad2deg(alpha)
    return alpha, alpha_deg


def make_ply_from_vertex_list(vertex_list):
    ply = ['ply', 'format ascii 1.0']
    ply += ['element vertex {}'.format(len(vertex_list))]
    ply += ['property float x', 'property float y', 'property float z',
            'property float nx', 'property float ny', 'property float nz',
            'property uchar diffuse_red', 'property uchar diffuse_green', 'property uchar diffuse_blue']
    ply += ['end_header']
    ply += vertex_list
    return '\n'.join(ply)


def save_dmap_as_ply(img, dmap, pos, target_path):
    img_2D = np.reshape(np.transpose(img, axes=[2, 0, 1]), [3, -1])
    dmap_2D = np.reshape(np.transpose(dmap, axes=[2, 0, 1]), [1, -1])

    pixel_to_ray_array_2D = np.reshape(np.transpose(pos, axes=[2, 0, 1]), [3, -1])
    pixel_to_ray_array_2D = pixel_to_ray_array_2D.astype(np.float32)

    world_coord = pixel_to_ray_array_2D * dmap_2D

    # colors
    r = deepcopy(img_2D[0, :].astype('uint8'))
    g = deepcopy(img_2D[1, :].astype('uint8'))
    b = deepcopy(img_2D[2, :].astype('uint8'))

    # coordinates
    x = deepcopy(world_coord[0, :])
    y = deepcopy(world_coord[1, :])
    z = deepcopy(world_coord[2, :])

    non_zero_idx = np.nonzero(z)
    r = r[non_zero_idx]
    g = g[non_zero_idx]
    b = b[non_zero_idx]
    x = x[non_zero_idx]
    y = y[non_zero_idx]
    z = z[non_zero_idx]

    # first ply: color-coded
    vertex_list_rgb = []
    for x_, y_, z_, r_, g_, b_ in zip(x, y, z, r, g, b):
        if z_ > 1e-3:
            vertex_list_rgb.append('{} {} {} 0 0 0 {} {} {}'.format(x_, y_, z_, r_, g_, b_))

    ply_file_rgb = open(target_path, 'w')
    ply_file_rgb.write(make_ply_from_vertex_list(vertex_list_rgb))
    ply_file_rgb.close()
