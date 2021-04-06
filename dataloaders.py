#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms

from torch.utils.data import Dataset

from PIL import Image


class Left2RightDataset(Dataset):

    """ Insert documentation for class """

    def __init__(self, name: str, mode: str, transforms: transforms, group: str = "l2r", root: str = f"./dataset"):

        # Arguments
        self.name = name
        self.mode = mode
        self.group = group
        self.transforms = transforms

        # Sort stereo left/right and corresponding disparity
        self.stereo_l = sorted(glob.glob(os.path.join(root, group, name, f"{mode}/left") + "/*.*"))
        self.stereo_r = sorted(glob.glob(os.path.join(root, group, name, f"{mode}/right") + "/*.*"))

    def __getitem__(self, index):

        item_left = self.get_transforms(Image.open(self.stereo_l[index % len(self.stereo_l)]))
        item_right = self.get_transforms(Image.open(self.stereo_r[index % len(self.stereo_r)]))

        return {"left": item_left, "right": item_right}

    def __len__(self):
        return max(len(self.stereo_l), len(self.stereo_r))


class Stereo2DisparityDataset(Dataset):

    """ Insert documentation for class """

    def __init__(self, name: str, mode: str, transforms: transforms, group: str = "s2d", root: str = f"./dataset"):

        # Arguments
        self.name = name
        self.mode = mode
        self.group = group
        self.transforms = transforms

        # Sort stereo left/right and corresponding disparity
        self.stereo_l = sorted(glob.glob(os.path.join(root, group, name, f"{mode}/A/left") + "/*.*"))
        self.stereo_r = sorted(glob.glob(os.path.join(root, group, name, f"{mode}/A/right") + "/*.*"))
        self.disparity = sorted(glob.glob(os.path.join(root, group, name, f"{mode}/B") + "/*.*"))

    def __getitem__(self, index):

        item_A_l = self.transforms(Image.open(self.stereo_l[index % len(self.stereo_l)]))
        item_A_r = self.transforms(Image.open(self.stereo_r[index % len(self.stereo_r)]))
        disparity = self.transforms(Image.open(self.disparity[index % len(self.disparity)]))

        return {"A_left": item_A_l, "A_right": item_A_r, "B": disparity}

    def __len__(self):
        return max(len(self.stereo_l), len(self.disparity))


class MyDataLoader:

    """ Insert documentation for class """

    def __init__(self, verbose: str = True):

        """ Insert documentation """

        self.verbose = verbose

    def get_dataset(
        self, dataset_group: str, dataset_name: str, dataset_mode: str, image_size: tuple, channels: int,
    ):

        """ Insert documentation """

        # Summary
        summary = f"group='{dataset_group}'; name='{dataset_name}', mode='{dataset_mode}', channels='{channels}'; image_size='{image_size}'"
        print(f"- [MyDataLoader] | Added a dataset; {summary}")

        # Return corresponding dataset class object
        if dataset_group == "l2r":
            return Left2RightDataset(dataset_name, dataset_mode, self.__get_transforms(channels, image_size))
        elif dataset_group == "s2d":
            return Stereo2DisparityDataset(dataset_name, dataset_mode, self.__get_transforms(channels, image_size))
        else:
            raise Exception(f"Can not get dataset for given: {summary}")

    def __get_transforms(self, channels, image_size, crop_ratio: float = 0.82) -> transforms:

        RANDOM_CROP = (int(image_size[0] * crop_ratio), int(image_size[1] * crop_ratio))

        transforms_1d: transforms = transforms.Compose(
            [
                transforms.Resize(size=image_size, interpolation=Image.BICUBIC),
                transforms.RandomCrop(size=RANDOM_CROP),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5)),
            ]
        )

        transforms_3d: transforms = transforms.Compose(
            [
                transforms.Resize(size=image_size, interpolation=Image.BICUBIC),
                transforms.RandomCrop(size=RANDOM_CROP),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        if channels == 1:
            return transforms_1d
        elif channels == 3:
            return transforms_3d
        else:
            raise Exception(f"Can not return transformations; for channels={channels}")
