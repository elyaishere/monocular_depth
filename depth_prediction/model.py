from torchvision.models.feature_extraction import create_feature_extractor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from AdaBins.models.miniViT import mViT
import clip

from modules import Decoder


clip_vis = "RN50"

class Model(nn.Module):
    def __init__(
            self, min_val=0.1, max_val=10,
            embedding_dim=128, K=10, norm='linear',
            n_query_channels=48, feature_extraction_layer='encoder0',
            softmax_first=True):
        super().__init__()


        self.min_val = min_val
        self.max_val = max_val
        self.norm = norm
        self.softmax_first = softmax_first

        model, _ = clip.load(clip_vis, device='cpu')
        self.clip_visual_encoder = model.visual
        for p in self.clip_visual_encoder.parameters():
            p.requires_grad = False

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = create_feature_extractor(
            resnet,
            return_nodes={
                'relu': 'layer0',
                'layer1': 'layer1',
                'layer2': 'layer2',
                'layer3': 'layer3',
                'layer4': 'layer4',
                'avgpool': 'layer5'
            }
        )
        # test run to save the dimentions
        with torch.no_grad():
            t = torch.randn(1,3,224,224)
            out = self.encoder(t)
            c2 = self.clip_visual_encoder(t).shape[1]
        features = [v.shape[1] for _, v in out.items()]
        self.decoder = Decoder(features=features[::-1], num_classes=c2)
        self.proj = nn.Parameter(torch.randn(1, K))
        if feature_extraction_layer == 'decoder':
            self.layer = 'decoder'
        else:
            layer = int(feature_extraction_layer[-1])
            self.layer = f'layer{layer}'
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2**layer),
                nn.Conv2d(out[self.layer].shape[1], c2, kernel_size=1, stride=1, padding=0)
            )
        self.bins = K

        # w and h must be divisible by patch_size
        self.adaptive_bins_layer = mViT(c2, n_query_channels=n_query_channels, patch_size=16,
                                        dim_out=self.bins,
                                        embedding_dim=embedding_dim, norm=norm)
        self.conv_out = nn.Conv2d(n_query_channels, self.bins, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # shape of x: B 3 H W 
        with torch.no_grad():
            clip_f = self.clip_visual_encoder(x).float().unsqueeze(-1) # B C2 1

        clip_f = (clip_f @ self.proj).permute(0, 2, 1) # B K C2
        clip_f = clip_f / clip_f.norm(dim=-1, keepdim=True) # normalize clip_f

        e = self.encoder(x)
        img_f = self.decoder(e) # B C2 H/2 W/2
        h, w = img_f.shape[-2:] # h=H/2, w=W/2
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(img_f)
        out = self.conv_out(range_attention_maps) # B K h w

        if self.layer == 'decoder':
            e = (img_f / img_f.norm(dim=1, keepdim=True)).flatten(start_dim=2) # normalize e: B C2 (hw)

        else:
            e = self.conv(e[self.layer]) # B C2 H/2 W/2
            e = (e / e.norm(dim=1, keepdim=True)).flatten(start_dim=2) # normalize e: B C2 (hw)
        similarity = torch.bmm(clip_f, e).view(-1, self.bins, h, w) # B K h w

        if self.softmax_first:
            similarity = F.softmax(similarity, dim=1)
            out = F.softmax(out, dim=1)
            weights = 0.5 * (similarity + out)
        else:
            weights = F.softmax(similarity + out, dim=1)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        centers = centers.view(-1, self.bins, 1, 1)

        pred = torch.sum(weights * centers, dim=1, keepdim=True) # B 1 H/2 W/2
        return bin_edges, pred
