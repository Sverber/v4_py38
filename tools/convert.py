#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import torchvision.transforms as transforms

from PIL import Image, ImageOps
from torchvision import transforms


def transform_CHANNELS2xCHANNELS(path, to_channels: int = 3) -> None:

    images = sorted(glob.glob(path))
    digits = len(str(len(images)))

    print("path:", path)
    print("len(images):", len(images))

    for i, path in enumerate(images):

        # Open image, transform to tensor, read its channels
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(Image.open(path))
        image_channels = image_tensor.shape[0]

        # Equality check
        equality = int(image_channels) == int(to_channels)

        # Verbose
        prefix = f"{path} | [{str(i + 1).zfill(digits)} / {str(len(images)).zfill(digits)}]"

        if not equality:
            image_rgb = Image.open(path).convert("RGB")
            image_rgb.save(path)
            print(
                f"{prefix} Image channels: {image_channels}, must be: {to_channels} | Equality: {equality} | Converted image ({i + 1}) to {to_channels} channels."
            )

        else:
            print(f"{prefix} Image channels: {image_channels}, must be: {to_channels} | Equality: {equality} ")


def invert_RGB_COLOURS(path) -> None:

    images = sorted(glob.glob(path))
    digits = len(str(len(images)))

    to_channels = 3

    for i, path in enumerate(images):

        # Invert colours and save
        inverted_image = ImageOps.invert(Image.open(path))
        inverted_image.save(path)

        # Verbose
        print(f"{path} | [{str(i + 1).zfill(digits)}/{str(len(images)).zfill(digits)}] Inverted image colours")


if __name__ == "__main__":

    try:

        # Transform a dataset
        transform_CHANNELS2xCHANNELS(
            path=os.path.join("dataset/s2d/Test_Set_RGB_Original/test/B" + "/*.*"), to_channels=3
        )

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

