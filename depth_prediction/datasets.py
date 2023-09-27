from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import torch
import os
import numpy as np

class DIODE(Dataset):
    def __init__(self, data, resize=None, transform=None, test=False, normalize=None):
        self.imgs = data['image']
        self.depths = data['depth']
        self.masks = data['mask']
        
        self.transform = transform
        self.img_resize = transforms.Resize(resize, interpolation=Image.BILINEAR) if resize is not None else None
        self.depth_resize = transforms.Resize(resize, interpolation=Image.NEAREST) if resize is not None and not test else None
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize is not None else None
        self.to_tensor = transforms.ToTensor()
        self.min_depth = 0.6
        self.max_depth = 350

    def __len__(self):
        return len(self.imgs)                 

    def __getitem__(self, idx):
        image = cv2.imread(self.imgs[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.to_tensor(image) / 255
        if self.img_resize is not None:
            image = self.img_resize(image)
        
        depth_map = np.load(self.depths[idx]).squeeze()

        mask = np.load(self.masks[idx])
        mask = mask > 0
        depth_map = np.ma.masked_where(~mask, depth_map)
        depth_map = np.clip(depth_map, self.min_depth, self.max_depth)
        depth_map = self.to_tensor(depth_map) / self.max_depth
        if self.depth_resize is not None:
            depth_map = self.depth_resize(depth_map)

        if self.transform:
            for t in self.transform:
                image, depth_map = t(image, depth_map)

        if self.normalize is not None:
            image = self.normalize(image)

        return image.float(), depth_map.float()


class NYU(Dataset):
    def __init__(self, img_path, gt_path, file=None, resize=None, transform=None, test=False, normalize=None):
        self.img_path = img_path
        self.gt_path = gt_path
        self.filenames = [i.split('.')[0] for i in os.listdir(self.img_path)]
        if file is not None:
            with open(file, 'r') as f:
                self.filenames = [l.strip() for l in f.readlines()]
        
        self.transform = transform
        self.img_resize = transforms.Resize(resize, interpolation=Image.BILINEAR) if resize is not None else None
        self.depth_resize = transforms.Resize(resize, interpolation=Image.NEAREST) if resize is not None and not test else None
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize is not None else None
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)                 

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_path, self.filenames[idx] + '.jpg')
        depth_path = os.path.join(self.gt_path, self.filenames[idx] + '.npy')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.to_tensor(image)

        depth_gt = np.flip(np.load(depth_path).transpose(1,0), 1)[np.newaxis, ...]
        depth_gt = torch.from_numpy(depth_gt.copy())

        # To avoid blank boundaries due to pixel registration
        image = image[:, 45:473, 43:609]
        depth_gt = depth_gt[:, 45:473, 43:609]

        if self.img_resize is not None:
            image = self.img_resize(image)

        if self.depth_resize is not None:
            depth_gt = self.depth_resize(depth_gt)

        if self.transform:
            for t in self.transform:
                image, depth_gt = t(image, depth_gt)

        if self.normalize is not None:
            image = self.normalize(image)

        return image.float(), depth_gt.float()


class COCO(Dataset):
    def __init__(self, img_path, gt_path, file=None, resize=None, transform=None, test=False, normalize=None):
        self.img_path = img_path
        self.gt_path = gt_path
        self.filenames = [i.split('.')[0] for i in os.listdir(self.img_path)]
        if file is not None:
            with open(file, 'r') as f:
                self.filenames = [l.strip() for l in f.readlines()]
        
        self.transform = transform
        self.img_resize = transforms.Resize(resize, interpolation=Image.BILINEAR) if resize is not None else None
        self.depth_resize = transforms.Resize(resize, interpolation=Image.NEAREST) if resize is not None and not test else None
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize is not None else None
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)                 

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_path, self.filenames[idx] + '.jpg')
        depth_path = os.path.join(self.gt_path, self.filenames[idx] + '.npy')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.to_tensor(image)

        depth_gt = np.load(depth_path)[np.newaxis, ...]
        depth_gt = torch.from_numpy(depth_gt.copy())

        if self.img_resize is not None:
            image = self.img_resize(image)

        if self.depth_resize is not None:
            depth_gt = self.depth_resize(depth_gt)

        if self.transform:
            for t in self.transform:
                image, depth_gt = t(image, depth_gt)

        if self.normalize is not None:
            image = self.normalize(image)

        return image.float(), depth_gt.float()
