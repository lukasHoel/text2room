import torch.nn as nn
from model.iron_depth.models.submodules.D_submodules import Encoder, UpSampleGN


class DNET(nn.Module):
    def __init__(self, output_dims):
        super(DNET, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(output_dims)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Decoder(nn.Module):
    def __init__(self, output_dims):
        super(Decoder, self).__init__()
        output_dim, feature_dim, hidden_dim = output_dims
        features = 2048
        bottleneck_features = 2048

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleGN(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleGN(skip_input=features // 2 + 40 + 24, output_features=features // 4)

        # three prediction heads
        i_dim = features // 4
        h_dim = 128

        self.depth_head = nn.Sequential(
                    nn.Conv2d(i_dim, h_dim, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(h_dim, h_dim, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(h_dim, output_dim, 1),
        )

        self.feature_head = nn.Sequential(
                    nn.Conv2d(i_dim, h_dim, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(h_dim, h_dim, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(h_dim, feature_dim, 1),
        )

        self.hidden_head = nn.Sequential(
                    nn.Conv2d(i_dim, h_dim, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(h_dim, h_dim, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(h_dim, hidden_dim, 1),
        )

    def forward(self, features):
        _, _, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_feat = self.up2(x_d1, x_block2)
        
        d = self.depth_head(x_feat)
        f = self.feature_head(x_feat)
        h = self.hidden_head(x_feat)
        return d, f, h

