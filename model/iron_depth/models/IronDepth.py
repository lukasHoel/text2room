import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from model.iron_depth.models.submodules.DNET import DNET
from model.iron_depth.models.submodules.Dr_submodules import LSPN


# downsample the predicted norm, predicted kappa (surface normal confidence), and pos (ray with unit depth)
def downsample(input_dict, size):
    for k in ['pred_norm', 'pred_kappa', 'pos']:
        input_dict[k+'_down'] = F.interpolate(input_dict[k], size=size, mode='bilinear', align_corners=False)
        # make sure to normalize
        if k == 'pred_norm':
            norm = torch.sqrt(torch.sum(torch.square(input_dict[k+'_down']), dim=1, keepdim=True))
            norm[norm < 1e-10] = 1e-10
            input_dict[k+'_down'] = input_dict[k+'_down'] / norm
    return input_dict


class IronDepth(nn.Module):
    def __init__(self, args):
        super(IronDepth, self).__init__()
        self.args = args
        self.downsample_ratio = 8

        # define D-Net
        self.output_dim = output_dim = 1
        self.feature_dim = feature_dim = 64
        self.hidden_dim = hidden_dim = 64

        #print('defining D-Net...')

        self.d_net = DNET(output_dims=[output_dim, feature_dim, hidden_dim])

        #print('defining Dr-Net...')
        self.dr_net = LSPN(args)

        self.ps = 5
        self.center_idx = (self.ps * self.ps - 1) // 2

        self.pad = (self.ps - 1) // 2
        self.irm_train_iter = args.train_iter             # 3
        self.irm_test_iter = args.test_iter               # 20

    def forward(self, input_dict, mode='train', ini_depth=None):
        pred_dmap, input_dict['feat'], h = self.d_net(input_dict['img'])

        down_size = [pred_dmap.size(2), pred_dmap.size(3)]
        input_dict = downsample(input_dict, down_size)

        if ini_depth is not None:
            pred_dmap = F.interpolate(ini_depth, size=down_size, mode='bilinear', align_corners=False)

        # depth_weights (B, ps*ps, H, W)
        input_dict['depth_candidate_weights'] = self.get_depth_candidate_weights(input_dict)

        # weights for upsampling
        input_dict['upsampling_weights'] = self.get_upsampling_weights(input_dict)

        # upsample first prediction
        up_pred_dmap = self.dr_net(h, pred_dmap, input_dict, upsample_only=True)

        if 'input_depth' in input_dict:
            # replace prediction from D-Net with other depth map, but use the same resolution as pred_dmap for consistency
            # only replace those pixels that are not masked out in input_mask
            mask = input_dict['input_mask'] > 0.5
            mask = mask + (input_dict['input_depth'].isinf()) + (input_dict['input_depth'] == 0)
            input_dict['input_mask'] = mask
            up_pred_dmap_merged = torch.where(mask, up_pred_dmap, input_dict['input_depth'])
            pred_dmap = F.interpolate(up_pred_dmap_merged, size=down_size, mode='bilinear', align_corners=False)

        # iterative refinement
        pred_list = [up_pred_dmap]
        N = self.irm_train_iter if mode == 'train' else self.irm_test_iter
        for i in range(N):
            h, pred_dmap, up_pred_dmap = self.dr_net(h, pred_dmap.detach(), input_dict)
            pred_list.append(up_pred_dmap)
            if 'input_depth' in input_dict and input_dict.get('fix_input_depth', False):
                # from section 5.2 in the paper: fix 'anchor points' from the input depth
                up_pred_dmap_merged = torch.where(input_dict['input_mask'], up_pred_dmap, input_dict['input_depth'])
                pred_dmap = F.interpolate(up_pred_dmap_merged, size=down_size, mode='bilinear', align_corners=False)

        return pred_list

    def get_depth_candidate_weights(self, input_dict):
        with torch.no_grad():
            B, _, H, W = input_dict['pred_norm_down'].shape

            # pred norm down - nghbr
            pred_norm_down = F.pad(input_dict['pred_norm_down'], pad=(self.pad, self.pad, self.pad, self.pad), mode='replicate')
            pred_norm_down_unfold = F.unfold(pred_norm_down, [self.ps, self.ps], padding=0)                                     # (B, 3*ps*ps, H*W)
            pred_norm_down_unfold = pred_norm_down_unfold.view(B, 3, self.ps*self.ps, H, W)                                     # (B, 3, ps*ps, H, W)

            # pos down - nghbr
            pos_down_nghbr = F.pad(input_dict['pos_down'], pad=(self.pad, self.pad, self.pad, self.pad), mode='replicate')
            pos_down_nghbr_unfold = F.unfold(pos_down_nghbr, [self.ps, self.ps], padding=0)                                         # (B, 2*ps*ps, H*W)
            pos_down_nghbr_unfold = pos_down_nghbr_unfold.view(B, 2, self.ps*self.ps, H, W)                                         # (B, 2, ps*ps, H, W)

            # norm and pos - nghbr
            nx, ny, nz = pred_norm_down_unfold[:, 0, ...], pred_norm_down_unfold[:, 1, ...], pred_norm_down_unfold[:, 2, ...]       # (B, ps*ps, H, W) or (B, 1, H, W)
            pos_u, pos_v = pos_down_nghbr_unfold[:, 0, ...], pos_down_nghbr_unfold[:, 1, ...]                                       # (B, ps*ps, H, W)

            # pos - center
            pos_u_center = pos_u[:, self.center_idx, :, :].unsqueeze(1)                                                             # (B, 1, H, W)
            pos_v_center = pos_v[:, self.center_idx, :, :].unsqueeze(1)                                                             # (B, 1, H, W)

            ddw_num = nx * pos_u + ny * pos_v + nz
            ddw_denom = nx * pos_u_center + ny * pos_v_center + nz
            ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

            ddw_weights = ddw_num / ddw_denom                                                                                       # (B, ps*ps, H, W)
            ddw_weights[ddw_weights != ddw_weights] = 1.0               # nan
            ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0   # inf
        return ddw_weights

    def get_upsampling_weights(self, input_dict):
        with torch.no_grad():
            B, _, H, W = input_dict['pos_down'].shape
            k = self.downsample_ratio

            # norm nghbr
            pred_norm_down = F.pad(input_dict['pred_norm_down'], pad=(1,1,1,1), mode='replicate')
            up_norm = F.unfold(pred_norm_down, [3, 3], padding=0)   # (B, 3, H, W) -> (B, 3 X 3*3, H*W)
            up_norm = up_norm.view(B, 3, 9, 1, 1, H, W)             # (B, 3, 3*3, 1, 1, H, W)

            # pos nghbr
            pos_down = F.pad(input_dict['pos_down'], pad=(1,1,1,1), mode='replicate')
            up_pos_nghbr = F.unfold(pos_down, [3, 3], padding=0)        # (B, 2, H, W) -> (B, 2 X 3*3, H*W)
            up_pos_nghbr = up_pos_nghbr.view(B, 2, 9, 1, 1, H, W)       # (B, 2, 3*3, 1, 1, H, W)

            # pos ref
            pos = input_dict['pos']
            up_pos_ref = pos.reshape(B, 2, H, k, W, k)                  # (B, 2, H, k, W, k)
            up_pos_ref = up_pos_ref.permute(0, 1, 3, 5, 2, 4)           # (B, 2, k, k, H, W)
            up_pos_ref = up_pos_ref.unsqueeze(2)                        # (B, 2, 1, k, k, H, W)

            # compute new depth
            new_depth_num = (up_norm[:, 0:1, ...] * up_pos_nghbr[:, 0:1, ...]) + \
                            (up_norm[:, 1:2, ...] * up_pos_nghbr[:, 1:2, ...]) + \
                            (up_norm[:, 2:3, ...])                      # (B, 1, 3*3, 1, 1, H, W)

            new_depth_denom = (up_norm[:, 0:1, ...] * up_pos_ref[:, 0:1, ...]) + \
                              (up_norm[:, 1:2, ...] * up_pos_ref[:, 1:2, ...]) + \
                              (up_norm[:, 2:3, ...])                    # (B, 1, 3*3, k, k, H, W)

            new_depth_denom[torch.abs(new_depth_denom) < 1e-8] = torch.sign(new_depth_denom[torch.abs(new_depth_denom) < 1e-8]) * 1e-8
            new_depth = new_depth_num / new_depth_denom                 # (B, 1, 3*3, k, k, H, W)

            # check for nan, inf
            new_depth[new_depth != new_depth] = 1.0  # nan
            new_depth[torch.abs(new_depth) == float("Inf")] = 1.0  # inf        
        return new_depth
