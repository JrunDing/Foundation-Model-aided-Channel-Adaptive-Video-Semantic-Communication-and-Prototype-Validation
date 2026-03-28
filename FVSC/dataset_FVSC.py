"""
@Brief: Video dataset.
If you are using a video dataset, you must first extract and save the video frames using FFmpeg, a
nd then read them using the dataset class below.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import random
import warnings
import torchvision
from subnet.basics import *
from PIL import Image

warnings.filterwarnings("ignore")


def random_crop_and_pad_image_and_labels(image, labels, size):
    """
    :brief: Data augmentation
    """
    combined = torch.cat([image, labels], 0)
    last_image_dim = image.size()[0]
    image_shape = image.size()
    combined_pad = F.pad(combined, (
    0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0, max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :]


def random_flip(images, labels):
    """
    :brief: Data augmentation
    """
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1

    if transforms and vertical_flip and random.randint(0, 1) == 1:  
        images = torch.flip(images, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:  
        images = torch.flip(images, [2])
        labels = torch.flip(labels, [2])

    return images, labels


class UCF101(data.Dataset):
    """
    :brief: Dataset
    :param path: Data path
    :param im_height: video/image height
    :param im_width: video/image width
    :return: Dataset class
    """
    def __init__(self, path="../data/UCF101/train/", im_height=240, im_width=320):
        self.parent_path = path
        self.im_height = im_height
        self.im_width = im_width
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.image_path = os.listdir(self.parent_path)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        img_path_ = self.image_path[index]
        img_path = os.path.join(self.parent_path, img_path_)
        ref_frame_path = os.path.join(img_path, "1.png")  # 1 is the previous frame, i.e., the reference frame.
        frame_path = os.path.join(img_path, "2.png")  # current frame

        frame = self.transform(Image.open(frame_path).convert("RGB"))
        ref_frame = self.transform(Image.open(ref_frame_path).convert("RGB"))

        frame, ref_frame = random_crop_and_pad_image_and_labels(frame, ref_frame, [self.im_height, self.im_width])
        frame, ref_frame = random_flip(frame, ref_frame)
        # First return to the previous reference frame, then return to the current frame behind it.
        return (ref_frame * 255).clone().requires_grad_(True), (frame * 255).clone().requires_grad_(True)


if __name__ == "__main__":
    # Example
    data_set = UCF101(path="../data/UCF101/train/")





