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
        self,
        name: str,
        mode: str,
        channels: int,
        check_channels: bool,
        transforms: transforms,
        group: str = "l2r",
        root: str = f"./dataset",
    ):

        # Arguments
        self.name = name
        self.mode = mode
        self.group = group
        self.transforms = transforms
        self.channels = channels
        self.transforms = transforms
        self.check_channels = check_channels

        self.prefix = f"- [L2R][{name}][{mode}]"

        # print(f"{self.prefix} run_checks:", check_channels)
        # print(f"{self.prefix} transforms:", transforms, "\n")

        self.stereo_l = self.get_images(
            path=os.path.join(root, group, name, f"{mode}/left") + "/*.*",
            channels=channels,
            check_channels=check_channels,
            prefix=self.prefix + "[left]",
        )
        self.stereo_r = self.get_images(
            path=os.path.join(root, group, name, f"{mode}/right") + "/*.*",
            channels=channels,
            check_channels=check_channels,
            prefix=self.prefix + "[right]",
        )

    def get_images(self, path: str, channels: int, check_channels: bool, prefix: str):

        images = sorted(glob.glob(path))

        if not check_channels:
            return images
        else:
            return self.run_checks(images, channels, prefix)

    @property
    def dataset_name(self) -> str:
        return self.name

    @property
    def dataset_mode(self) -> str:
        return self.mode

    @property
    def dataset_group(self) -> str:
        return self.group

    @staticmethod
    def run_checks(images: list, channels: int, prefix: str):

        digit_count = len(str(len(images)))

        for i, path in enumerate(images):

            # Open image, transform to tensor, read its channels
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(Image.open(path))
            image_channels = image_tensor.shape[0]

            # Equality check
            equality = int(image_channels) == int(channels)

            # Print progress
            print(
                f"{prefix} | [{str(i + 1).zfill(digit_count)}/{str(len(images)).zfill(digit_count)}] Image channels: {image_channels}, must be: {channels} | Equality: {equality} "
            )

            if not equality:

                print(f"\n\n {path} \n\n {image_tensor} \n")
                print("- Manually convert images using the convert functions found in '~./tools/convert.py'\n")

                break

                # image_grayscale = Image.open(path).convert("L")
                # image_grayscale.save(path)

        return images

    def __getitem__(self, index):

        item_left = self.transforms(Image.open(self.stereo_l[index % len(self.stereo_l)]))
        item_right = self.transforms(Image.open(self.stereo_r[index % len(self.stereo_r)]))

        return {"left": item_left, "right": item_right}

    def __len__(self):
        return max(len(self.stereo_l), len(self.stereo_r))


class Stereo2DisparityDataset(Dataset):

    """ Insert documentation for class """

    def __init__(
        self,
        name: str,
        mode: str,
        channels: int,
        check_channels: bool,
        transforms: transforms,
        group: str = "s2d",
        root: str = f"./dataset",
    ):

        # Arguments
        self.name = name
        self.mode = mode
        self.group = group
        self.channels = channels
        self.transforms = transforms
        self.check_channels = check_channels

        self.prefix = f"- [S2D][{name}][{mode}]"

        # print(f"{self.prefix} run_checks:", check_channels)
        # print(f"{self.prefix} transforms:", transforms, "\n")

        self.stereo_l = self.get_images(
            path=os.path.join(root, group, name, f"{mode}/A/left") + "/*.*",
            channels=channels,
            check_channels=check_channels,
            prefix=self.prefix + "[A/left]",
        )
        self.stereo_r = self.get_images(
            path=os.path.join(root, group, name, f"{mode}/A/right") + "/*.*",
            channels=channels,
            check_channels=check_channels,
            prefix=self.prefix + "[A/right]",
        )
        self.disparity = self.get_images(
            path=os.path.join(root, group, name, f"{mode}/B") + "/*.*",
            channels=channels,
            check_channels=check_channels,
            prefix=self.prefix + "[B]",
        )

    def get_images(self, path: str, channels: int, check_channels: bool, prefix: str):

        images = sorted(glob.glob(path))

        if not check_channels:
            return images
        else:
            return self.run_checks(images, channels, prefix)

    @property
    def dataset_name(self) -> str:
        return self.name

    @property
    def dataset_mode(self) -> str:
        return self.mode

    @property
    def dataset_group(self) -> str:
        return self.group

    @staticmethod
    def run_checks(images: list, channels: int, prefix: str):

        digit_count = len(str(len(images)))

        for i, path in enumerate(images):

            # Open image, transform to tensor, read its channels
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(Image.open(path))
            image_channels = image_tensor.shape[0]

            # Equality check
            equality = int(image_channels) == int(channels)

            # Print progress
            print(
                f"{prefix} | [{str(i + 1).zfill(digit_count)}/{str(len(images)).zfill(digit_count)}] Image channels: {image_channels}, must be: {channels} | Equality: {equality} "
            )

            if not equality:

                print(f"\n\n {path} \n\n {image_tensor} \n")
                print("- Manually convert images using the convert functions found in '~./tools/convert.py'\n")

                break

                # image_grayscale = Image.open(path).convert("L")
                # image_grayscale.save(path)

        return images

    def __getitem__(self, index):

        item_A_l = self.transforms(Image.open(self.stereo_l[index % len(self.stereo_l)]))
        item_A_r = self.transforms(Image.open(self.stereo_r[index % len(self.stereo_r)]))
        disparity = self.transforms(Image.open(self.disparity[index % len(self.disparity)]))

        return {"A_left": item_A_l, "A_right": item_A_r, "B": disparity}

    def __len__(self):
        return max(len(self.stereo_l), len(self.disparity))


class MyDataLoader:

    """ Insert documentation for class """

    def __init__(self):

        """ Insert documentation """

        pass

    def get_dataset(
        self,
        dataset_group: str,
        dataset_name: str,
        dataset_mode: str,
        image_size: tuple,
        channels: int,
        check_channels: bool = True,
    ):

        """ Insert documentation """

        # Summary
        summary = f"group='{dataset_group}'; name='{dataset_name}', mode='{dataset_mode}', channels='{channels}'; image_size='{image_size}'"
        print(f"- [MyDataLoader] | Added a dataset; {summary}")

        # Return corresponding dataset class object
        if dataset_group == "l2r":
            return Left2RightDataset(
                group=dataset_group,
                mode=dataset_mode,
                name=dataset_name,
                tFsforms=self.__get_transforms(channels, image_size),
                channels=channels,
                check_channels=check_channels,
            )
        elif dataset_group == "s2d":
            return Stereo2DisparityDataset(
                mode=dataset_mode,
                name=dataset_name,
                group=dataset_group,
                transforms=self.__get_transforms(channels, image_size),
                channels=channels,
                check_channels=check_channels,
            )
        else:
            raise Exception(f"Can not get dataset for given: {summary}")

    @staticmethod
    def __get_transforms(channels, image_size, crop_ratio: float = 1.0) -> transforms:

        RANDOM_CROP = (int(image_size[0] * crop_ratio), int(image_size[1] * crop_ratio))

        transforms_1d: transforms = transforms.Compose(
            [
                transforms.Resize(size=image_size, interpolation=Image.BICUBIC),
                # transforms.RandomCrop(size=RANDOM_CROP),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5)),
            ]
        )

        transforms_3d: transforms = transforms.Compose(
            [
                transforms.Resize(size=image_size, interpolation=Image.BICUBIC),
                # transforms.RandomCrop(size=RANDOM_CROP),
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
