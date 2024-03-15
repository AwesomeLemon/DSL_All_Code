import numpy as np
import random

import torch
from torchvision.transforms import transforms
import PIL.Image


class RandomColorJitter:
    def __init__(self, training=True):
        self.training = training
        self.torch_transform = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        if not self.training:
            return {'image': image, 'mask': mask}

        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)

        if random.random() < 0.8:
            image = PIL.Image.fromarray(image.transpose(1, 2, 0))
            image = self.torch_transform(image)
            image = np.array(image).transpose(2, 0, 1)

        return {'image': image, 'mask': mask, 'misc': misc}
