from munch import Munch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import os
import sys
import torch
import random
import logging


logging.basicConfig(level=logging.INFO)
sys.path.append('DRIT/src')


class UnpairedDataset(Dataset):
    def __init__(self, pathA, pathB, resize_size=256, phase='train', crop_size=216, no_flip=True):

        # A
        images_A = os.listdir(pathA)
        self.A = [os.path.join(pathA, x) for x in images_A]

        # B
        images_B = os.listdir(pathB)
        self.B = [os.path.join(pathB, x) for x in images_B]

        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = max(self.A_size, self.B_size)

        # setup image transformation
        transforms = [Resize((resize_size, resize_size), Image.BICUBIC)]
        if phase == 'train':
            transforms.append(RandomCrop(crop_size))
        else:
            transforms.append(CenterCrop(crop_size))
        if not no_flip:
            transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        logging.info('A: %d, B: %d images'%(self.A_size, self.B_size))

    def __getitem__(self, index):
        if self.dataset_size == self.A_size:
            data_A = self.load_img(self.A[index])
            data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)])
        else:
            data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)])
            data_B = self.load_img(self.B[index])
        return data_A, data_B

    def load_img(self, img_name):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        return img

    def __len__(self):
        return self.dataset_size


from model import DRIT
from saver import Saver


def main():
    # parameters
    args = Munch()
    args.resize_size = 256
    args.crop_size = 216
    args.no_flip = False #RandomHorizontalFlip
    args.batch_size = 2
    args.concat = 0
    args.no_ms = False #enabled mode seeking regularization
    args.dis_scale = 3
    args.input_dim_a = 3
    args.input_dim_b = 3
    args.dis_norm = 'None'
    args.dis_spectral_norm = True
    args.lr_policy = 'lambda'
    args.n_ep = 1200
    args.n_ep_decay = 600
    args.display_dir = 'logs'
    args.name = 'panel-coco'
    args.result_dir = 'results'
    args.display_freq = 1
    args.img_save_freq = 5
    args.model_save_freq = 3
    args.d_iter = 3
    args.no_display_img = False
    
    dataset = UnpairedDataset('dcm-dataset/panels', 'val2017', resize_size=args.resize_size, crop_size=args.crop_size, no_flip=args.no_flip)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    model = DRIT(args)
    model.setgpu(0)

    model.initialize()
    ep0 = -1
    total_it = 0
    #ep0, total_it = model.resume(os.path.join(args.result_dir, args.name, '00572.pth'))
    model.set_scheduler(args, last_ep=ep0)
    ep0 += 1
    saver = Saver(args)
    max_it = 500000
    for ep in range(ep0, args.n_ep):
        for it, (images_a, images_b) in enumerate(train_loader):
            if images_a.size(0) != args.batch_size or images_b.size(0) != args.batch_size:
                continue

            # input data
            images_a = images_a.cuda(0).detach()
            images_b = images_b.cuda(0).detach()

            # update model
            if (it + 1) % args.d_iter != 0 and it < len(train_loader) - 2:
                model.update_D_content(images_a, images_b)
                continue
            else:
                model.update_D(images_a, images_b)
                model.update_EG()

            # save to display file
            if not args.no_display_img:
                saver.write_display(total_it, model)

            logging.info('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(-1, model)
                saver.write_model(-1, model)
                break

        # decay learning rate
        if args.n_ep_decay > -1:
            model.update_lr()

        # save result image
        saver.write_img(ep, model)

        # Save network weights
        saver.write_model(ep, total_it, model)

    return

if __name__ == '__main__':
    main()
