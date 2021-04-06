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

    def __init__(
        self, root: str = f"./dataset", mode: str = "train", group: str = "l2r", transforms: transforms = None,
    ):

        # Dataset group
        self.group = group

        # Get transformation and alignment
        self.transforms = transforms

        # Sort stereo left/right and corresponding disparity
        self.stereo_l = sorted(glob.glob(os.path.join(root, group, f"{mode}/left") + "/*.*"))
        self.stereo_r = sorted(glob.glob(os.path.join(root, group, f"{mode}/right") + "/*.*"))

    def __getitem__(self, index):

        item_left = self.transforms(Image.open(self.stereo_l[index % len(self.stereo_l)]))
        item_right = self.transforms(Image.open(self.stereo_r[index % len(self.stereo_r)]))

        return {"left": item_left, "right": item_right}

    def __len__(self):
        return max(len(self.stereo_l), len(self.stereo_r))

    def __repr__(self) -> str:
        return f"<class> Left2RightDataset, len={max(len(self.stereo_l), len(self.stereo_r))}"


class Stereo2DisparityDataset(Dataset):

    """ Insert documentation for class """

    def __init__(
        self, root: str = f"./dataset", mode: str = "train", group: str = "s2d", transforms: transforms = None,
    ):
        # Dataset group
        self.group = group

        # Get transformation and alignment
        self.transforms = transforms

        # Sort stereo left/right and corresponding disparity
        self.stereo_l = sorted(glob.glob(os.path.join(root, group, f"{mode}/A/left") + "/*.*"))
        self.stereo_r = sorted(glob.glob(os.path.join(root, group, f"{mode}/A/right") + "/*.*"))
        self.disparity = sorted(glob.glob(os.path.join(root, group, f"{mode}/B") + "/*.*"))

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

        # Constants
        self.verbose = verbose

        # Dataset class objects dict
        self.__list: dict = {
            "train": {},
            "test": {},
        }

    def add_dataset(
        self,
        dataset_class_object: object,
        dataset_group: str,
        dataset_name: str,
        mode: str,
        image_size: tuple,
        channels: int,
    ) -> object:

        """ Insert documentation """

        # Add the dataset object
        self.__list[mode][dataset_group] = MyDataWrapper(
            dataset_class_object, f"./dataset/{dataset_group}", dataset_name, mode, image_size, channels,
        )

        # Print for debug
        self.__print(
            f"Added a dataset; group='{dataset_group}'; name='{dataset_name}', mode='{mode}', channels='{channels}'; image_size='{image_size}'"
        )

        return self.__list[mode][dataset_group]

    def __print(self, message, prefix: str = "- [MyDataLoader] | ") -> None:

        """ Insert documentation """

        if self.verbose is True:
            print(f"{prefix}{message}")

    @property
    def get_dataset(self, tag) -> object:
        return self.__list[tag]

    @property
    def get_list(self) -> dict:
        return self.__list


class MyDataWrapper:

    """ Insert documentation for class """

    def __init__(
        self,
        dataset_class_object: object,
        dataset_dir: str,
        dataset_name: str,
        mode: str,
        image_size: tuple,
        channels: int,
        crop_ratio: float = 0.82,
    ) -> None:

        # Image size
        self.image_size: tuple = image_size
        self.crop_ratio: float = crop_ratio
        self.randm_crop = (int(self.image_size[0] * self.crop_ratio), int(self.image_size[1] * self.crop_ratio))

        # Channels
        self.channels: int = channels

        # Dataset
        self.dataset_class_object = dataset_class_object
        self.dataset_dir: str = dataset_dir
        self.dataset_name: str = dataset_name
        self.dataset_root: str = f"./{dataset_dir}/{dataset_name}"

        # Load actual dataset into object
        self.dataset = self.load_dataset(mode)

    def load_dataset(self, mode):

        """ Insert documentation """

        self.dataset = self.dataset_class_object(self.dataset_root, mode, self.get_transforms)

        return self.dataset

    @property
    def get_transforms(self) -> transforms:

        """ Insert documentation """

        if self.channels == 1:
            return self.__transforms_1d

        elif self.channels == 3:
            return self.__transforms_3d
        else:
            raise Exception(f"Can not return transformations; for channels={self.channels}")

    @property
    def __transforms_1d(self) -> transforms:
        return transforms.Compose(
            [
                transforms.Resize(size=self.image_size, interpolation=Image.BICUBIC),
                transforms.RandomCrop(size=self.randm_crop),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5)),
            ]
        )

    @property
    def __transforms_3d(self) -> transforms:
        return transforms.Compose(
            [
                transforms.Resize(size=self.image_size, interpolation=Image.BICUBIC),
                transforms.RandomCrop(size=self.randm_crop),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

