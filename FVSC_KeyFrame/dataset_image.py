"""
@Brief: Key frame image dataset
"""


import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class UCF101_Image(Dataset):
    def __init__(self, path):
        self.data_path = path
        # transforms
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize([240, 320]),
                                                        torchvision.transforms.ToTensor()])
        self.image_path = os.listdir(self.data_path)

    def __getitem__(self, idx):
        img_name = self.image_path[idx]
        img_item_path = os.path.join(self.data_path, img_name)
        img = Image.open(img_item_path)
        img = img.convert("RGB")
        img = self.transform(img)
        return (img*255).clone().requires_grad_(True)

    def __len__(self):
        return len(self.image_path)


if __name__ == '__main__':
    img = UCF101_Image('../data/UCF101/UCF-101-TRA-IMAGE')
    print(len(img))

