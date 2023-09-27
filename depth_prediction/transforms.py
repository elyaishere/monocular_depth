from torchvision.transforms.functional import crop, get_dimensions, rotate, hflip, adjust_brightness, adjust_saturation, adjust_contrast
from PIL import Image
import numbers
import torch
import torch.nn as nn


class RandomCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif len(size) == 1:
            self.size = (int(size[0]), int(size[0]))
        else:
            self.size = (int(size[0]), int(size[1]))
    
    def forward(self, img, depth):
        _, h, w = get_dimensions(img)
        assert h >= self.size[0] and w >= self.size[1]
        assert h, w == get_dimensions(depth)[1:]
        x = torch.randint(0, w - self.size[1] + 1, size=(1,)).item()
        y = torch.randint(0, h - self.size[0] + 1, size=(1,)).item()
        return crop(img, y, x, self.size[0], self.size[1]), crop(depth, y, x, self.size[0], self.size[1])


class RandomRotate(nn.Module):
    def __init__(self, angle):
        super().__init__()
        if isinstance(angle, numbers.Number):
            self.angle = (float(-abs(angle)), float(abs(angle)))
        elif len(angle) == 1:
            self.angle = (float(-abs(angle[0])), float(-abs(angle[0])))
        else:
            self.angle = (float(angle[0]), float(angle[1]))
    
    def forward(self, img, depth):
        random_angle = float(torch.empty(1).uniform_(self.angle[0], self.angle[1]).item())
        return rotate(img, random_angle, interpolation=Image.BILINEAR), rotate(depth, random_angle, interpolation=Image.NEAREST)


class RandomHFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, img, depth):
        if torch.rand(1) < self.p:
            return hflip(img), hflip(depth)
        return img, depth


class ColorJitter(nn.Module):
    def __init__(self, brightness=(0.9,1.1), saturation=(0.9,1.1), contrast=(0.9,1.1)):
        super().__init__()
        self.brightness = brightness
        self.saturation = saturation
        self.contrast = contrast
    
    def forward(self, img, depth):
        b = float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
        s = float(torch.empty(1).uniform_(self.saturation[0], self.saturation[1]))
        c = float(torch.empty(1).uniform_(self.contrast[0], self.contrast[1]))
        img = adjust_brightness(img, b)
        img = adjust_saturation(img, s)
        img = adjust_contrast(img, c)
        return img, depth
