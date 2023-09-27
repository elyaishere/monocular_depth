import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpSample, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(out_features),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(out_features),
                                 nn.LeakyReLU(inplace=True))

    def forward(self, x, residual):
        x = F.interpolate(x, size=[residual.size(2), residual.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([x, residual], dim=1)
        return self.net(f)


class Decoder(nn.Module):
    def __init__(self, features, num_classes=1, bottleneck_features=2048):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[0], bottleneck_features, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(bottleneck_features, bottleneck_features, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        features = [bottleneck_features] + features[1:]
        self.num_layers = len(features) - 1
        self.up = nn.ModuleList([
            UpSample(in_features=features[i]+features[i+1], out_features=features[i+1])
            for i in range(self.num_layers)])

        self.conv = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[-1], num_classes, kernel_size=3, stride=1, padding=1)

        )

    def forward(self, features):
        out = self.bottleneck(features[f'layer{self.num_layers}'])
        for i, m in enumerate(self.up, start=1):
            out = m(out, features[f'layer{self.num_layers - i}'])
        out = self.conv(out)
        return out
