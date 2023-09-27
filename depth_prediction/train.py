import sys

sys.path.append('AdaBins')
sys.path.append('CLIP')

from torch.utils.data import DataLoader
from munch import Munch
from tqdm.auto import tqdm
import os
import gc
import torch
import wandb
import hydra
import omegaconf

from AdaBins.loss import SILogLoss, BinsChamferLoss

from datasets import *
from model import Model
from transforms import *
from utils import sample_depth

os.makedirs('checkpoint', exist_ok=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    torch.manual_seed(1)
    wandb.init(
        project=cfg.setup.project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
    )
    os.makedirs('checkpoint', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0 if device.type == 'cuda' else 2
    pin_memory = True if device.type == 'cuda' else False

    model_cfg = wandb.config.model
    model = Model(
        min_val=model_cfg['min_depth'], max_val=model_cfg['max_depth'],
        K=model_cfg['n_bins'], norm=model_cfg['norm'],
        feature_extraction_layer=model_cfg['feature_extraction_layer'],
        softmax_first=model_cfg['softmax_first']
    )

    data_cfg = wandb.config.data
    transform = [RandomRotate(data_cfg['rotate_angle']), RandomCrop(data_cfg['crop_size'])]
    if not data_cfg['no_hflip']:
        transform += [RandomHFlip()]
    if data_cfg['color_augment']:
        transform += [ColorJitter()]

    dataset = COCO('train2017', 'COCO_midas_depth', resize=data_cfg['resize_size'], transform=transform, normalize=True)

    train_cfg = wandb.config.train
    train_loader = DataLoader(dataset, train_cfg['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    optim_cfg =  wandb.config.optim
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=optim_cfg['weight_decay'], lr=optim_cfg['lr'])
    scheduler_cfg =  wandb.config.scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, scheduler_cfg['lr'], epochs=train_cfg['epochs'], steps_per_epoch=len(train_loader),
        cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
        div_factor=scheduler_cfg['div_factor'], final_div_factor=scheduler_cfg['final_div_factor'])

    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss()

    model = model.to(device)
    step = 0

    log_info = Munch()

    for epoch in range(train_cfg['epochs']):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{train_cfg['epochs']}. Loop: Train", total=len(train_loader)):

            optimizer.zero_grad()
            img, depth = [b.to(device) for b in batch]

            bin_edges, pred = model(img)

            mask = (depth > model_cfg['min_depth']) * (depth < model_cfg['max_depth'])
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=pred.shape!=depth.shape)

            if train_cfg['w_chamfer'] > 0:
                l_chamfer = criterion_bins(bin_edges.cpu(), depth.cpu())
            else:
                l_chamfer = torch.Tensor([0]).to(img.device)

            loss = l_dense + train_cfg['w_chamfer'] * l_chamfer
            loss.backward()

            log_info['SILogLoss'] = l_dense.item()
            if train_cfg['w_chamfer'] > 0:
                log_info['BinsChamferLoss'] = l_chamfer.item()
            log_info['Loss'] = loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()

            step += 1
            scheduler.step()

            if step % train_cfg['save_every'] == 0:
                torch.save(model.state_dict(), f'checkpoint/model_{step}.ckpt')
                torch.save(optimizer.state_dict(), f'checkpoint/optimizer_{step}.ckpt')
            
            if step % train_cfg['sample_every'] == 0:
                batch = (img[::2, ...].cpu(), depth[::2, ...].cpu())
                depth, img = sample_depth(model, batch, depth=pred[::2, ...], device=device)
                log_info.depth_samples = depth
                log_info.input = img
            wandb.log(log_info)

        torch.cuda.empty_cache()
        gc.collect()
    wandb.finish()

if __name__ == "__main__":
    main()
