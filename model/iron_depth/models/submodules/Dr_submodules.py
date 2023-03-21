import torch
import torch.nn as nn
import torch.nn.functional as F


class LSPN(nn.Module):
    def __init__(self, args):
        super(LSPN, self).__init__()
        self.args = args
        self.downsample_ratio = 8

        self.output_dim = output_dim = 1
        self.feature_dim = feature_dim = 64
        self.hidden_dim = hidden_dim = 64

        # define ConvGRU cell
        self.input_dim = output_dim     # use current depth
        self.input_dim += feature_dim   # use context feature
        self.input_dim += 1             # use surface normal uncertainty (kappa)

        conv_gru_ks=5
        self.gru = ConvGRU(hidden_dim=self.hidden_dim, input_dim=self.input_dim, ks=conv_gru_ks)

        # Pr-Net
        self.ps = 5                     # propagation patch size
        self.pad = (self.ps - 1) // 2

        d_dim = self.ps*self.ps
        h_dim = 128
        self.depth_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, h_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, d_dim, 1),
        )

        # Up-Net
        h_dim = 128
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, h_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, 9 * self.downsample_ratio * self.downsample_ratio, 1)
        )

    def forward(self, h, pred_depth, input_dict, upsample_only=False):
        if upsample_only:
            return self.upsample_depth(pred_depth, h, input_dict)
        else:
            B, _, H, W = pred_depth.shape
            x = self.prepare_input(pred_depth, input_dict)
            h_new = self.gru(h, x)

            # get new depth
            depth_prob = self.depth_head(h_new)                                                                         # (B, ps*ps, H, W)
            depth_prob = torch.softmax(depth_prob, dim=1)                                                               # (B, ps*ps, H, W)

            # get depth candidates
            depth_candidate_weights = input_dict['depth_candidate_weights']                                             # (B, ps*ps, H, W)

            pred_depth_pad = F.pad(pred_depth[:, 0:1, :, :], pad=(self.pad, self.pad, self.pad, self.pad), mode='replicate')
            pred_depth_unfold = F.unfold(pred_depth_pad, [self.ps, self.ps], dilation=1, padding=0)                     # (B, ps*ps, H*W)
            pred_depth_unfold = pred_depth_unfold.view(B, self.ps*self.ps, H, W)                                        # (B, ps*ps, H, W)
            new_depth_candidates = depth_candidate_weights * pred_depth_unfold                                          # (B, ps*ps, H, W)

            pred_depth = torch.sum(depth_prob * new_depth_candidates, dim=1, keepdim=True)                              # (B, 1, H, W)

            # upsample
            up_pred_depth = self.upsample_depth(pred_depth, h_new, input_dict)

        return h_new, pred_depth, up_pred_depth


    def upsample_depth(self, pred_depth, h, input_dict):
        up_mask = self.mask_head(h)
        return upsample_depth_via_normal(pred_depth, up_mask, self.downsample_ratio, input_dict)

    def prepare_input(self, pred_depth, input_dict):
        inputs = [pred_depth]
        inputs.append(input_dict['feat'])
        inputs.append(input_dict['pred_kappa_down'])
        return torch.cat(inputs, dim=1)


# ConvGRU cell
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, ks=3):
        super(ConvGRU, self).__init__()
        p = (ks - 1) // 2
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h


# normal-guided upsampling 
def upsample_depth_via_normal(depth, up_mask, k, input_dict):
    B, o_dim, H, W = depth.shape
    up_mask = up_mask.view(B, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)             # (B, 1, 9, k, k, H, W)

    # split depth
    depth_mu = depth

    # upsample depth mu
    depth_mu = F.pad(depth_mu, pad=(1,1,1,1), mode='replicate')
    up_depth_mu = F.unfold(depth_mu, [3, 3], padding=0)  # (B, 1, H, W) -> (B, 1 X 3*3, H*W)
    up_depth_mu = up_depth_mu.view(B, 1, 9, 1, 1, H, W)  # (B, 1, 3*3, 1, 1, H, W)

    # multiply nghbr depth
    new_depth = input_dict['upsampling_weights'] * up_depth_mu

    up_depth_mu = torch.sum(up_mask * new_depth, dim=2)  # (B, 1, k, k, H, W)
    up_depth_mu = up_depth_mu.permute(0, 1, 4, 2, 5, 3)  # (B, 1, H, k, W, k)
    up_depth_mu = up_depth_mu.reshape(B, 1, k * H, k * W)  # (B, 1, kH, kW)

    return up_depth_mu



