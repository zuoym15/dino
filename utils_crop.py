import math

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms, datasets
import torchvision.transforms.functional as F
from os.path import dirname
import os

from PIL import Image

def load_kitti_boxlist(filename, frame_id):
    file1 = open(filename, 'r')
    Lines = file1.readlines()

    res = []

    for line in Lines:
        info = line.split(' ')
        if int(info[0]) == frame_id and info[2] != 'DontCare':
            res.append([float(info[6]), float(info[7]), float(info[8])-float(info[6]), float(info[9])-float(info[7])])  # [x1, y1, x2, y2] -> j, i, w, h

    return res # list of N elements, each is a list of 4 elements

class KittiDataset(datasets.ImageFolder):
    def __init__(self, data_path, transform):
        super().__init__(data_path, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        base_dir = dirname(dirname(dirname(path))) # /projects/katefgroup/datasets/kitti_tracking/training/image_02/0009/000698.png -> /projects/katefgroup/datasets/kitti_tracking/training
        video_id = os.path.basename(dirname(path)) # '0009'
        frame_id = int(os.path.splitext(os.path.basename(path))[0]) # 000698 -> 698

        label_file = os.path.join(base_dir, 'label_02', video_id+'.txt')

        bboxes = load_kitti_boxlist(label_file, frame_id) # list of N elements, each is a list of 4 elements

        if self.transform is not None:
            sample = self.transform(sample, bboxes)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class RandomResizedCropAroundLocations(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)

    def get_params(self, img, scale, ratio, bbox=None):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
            bbox (list): bbox has format (i, j, h, w), corresponding to the feasible area of that the centroid of the crop can lie.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))

        if bbox is not None:
            box_j, box_i, box_w, box_h = bbox
            bbox_area = box_w * box_h
            
            # for _ in range(10):
            # target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            target_area = bbox_area * torch.empty(1).uniform_(scale[0], scale[1]).item() # based on bbox area, rather than image area
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            # if 0 < w <= width and 0 < h <= height:
            ci = torch.randint(round(box_i), round(box_i + box_h + 1), size=(1,)).item() # random a center location inside the box
            cj = torch.randint(round(box_j), round(box_j + box_w + 1), size=(1,)).item()
            i = int(round(ci - h/2))
            j = int(round(cj - w/2))

            return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


    def forward(self, img, bbox):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
            bbox (list): each bbox has format (i, j, h, w), corresponding to the feasible area of that the centroid of the crop can lie.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio, bbox)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)