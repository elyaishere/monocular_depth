from torchvision.utils import make_grid
from munch import Munch
import torch
import wandb
import torch.nn as nn
import numpy as np

def denormalize(img, mean, std):
    if len(img.shape) == 3:
        img = img.clone()[None, ...]
    std_t = torch.zeros_like(img)
    mean_t = torch.zeros_like(img)
    for i in range(3):
        std_t[:,i,:,:] = std[i]
        mean_t[:,i,:,:] = mean[i]
    return (img * std_t + mean_t).clamp_(0, 1)

@torch.no_grad()
def sample_depth(model, batch, depth=None, device="cpu"):
    img, gt_depth = batch
    N, _, _, _ = img.size()
    if depth is None:
        _, depth = model(img.to(device))
    depth = depth.cpu()
    if gt_depth.shape != depth.shape:
        depth = nn.functional.interpolate(depth, gt_depth.shape[-2:], mode='bilinear', align_corners=True)

    d_concat = torch.cat([gt_depth, depth], dim=0)
    img = denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    depth_grid = make_grid(d_concat, nrow=N, padding=0)
    img_grid = make_grid(img, nrow=N, padding=0)
    return wandb.Image(depth_grid), wandb.Image(img_grid)


def compute_metrics(gt, pred):
    metrics = Munch()
    metrics.rel = np.mean(np.abs(gt - pred) / gt)
    metrics.rms = np.mean((gt - pred) ** 2) ** 0.5
    metrics.log_10 = np.mean(np.abs(np.log10(gt) - np.log10(pred)))
    metrics.sq_rel = np.mean(((gt - pred) ** 2) / gt)
    metrics.rmse_log = np.mean((np.log(gt) - np.log(pred)) ** 2) ** 0.5
    delta = np.maximum((gt / pred), (pred / gt))
    metrics.threshold_acc_1 = np.mean((delta < 1.25))
    metrics.threshold_acc_2 = np.mean((delta < 1.25 ** 2))
    metrics.threshold_acc_3 = np.mean((delta < 1.25 ** 3))
    return metrics
