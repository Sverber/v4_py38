import os
import cv2
import glob
import random
import numpy as np


from typing import List
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


class StereoDisparityDataset(Dataset):

    """ Insert documentation for ImageDataset class """

    def __init__(self, root, mode: str = "train", transforms_GRAY: transforms = None, transforms_RGB: transforms = None):

        # Get transformation and alignment
        self.transform_GRAY = transforms_GRAY
        self.transform_RGB = transforms_RGB

        # Sort stereo left/right and corresponding disparity
        self.raw_stereo_l = sorted(glob.glob(os.path.join(root, f"{mode}/A/left") + "/*.*"))
        self.raw_stereo_r = sorted(glob.glob(os.path.join(root, f"{mode}/A/right") + "/*.*"))
        self.raw_disparity = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))

        # Transform grayscale images to grayscale RGB
        self.__transform_CHANNELS2RGB(files=self.raw_stereo_l, image_type="left")
        self.__transform_CHANNELS2RGB(files=self.raw_stereo_r, image_type="right")
        self.__transform_CHANNELS2RGB(files=self.raw_disparity, image_type="depth")

        # Import files again, some may have been transformed in the prior step
        self.stereo_l = sorted(glob.glob(os.path.join(root, f"{mode}/A/left") + "/*.*"))
        self.stereo_r = sorted(glob.glob(os.path.join(root, f"{mode}/A/right") + "/*.*"))
        self.disparity = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))

    def __transform_CHANNELS2RGB(self, files, image_type) -> List:

        # For every not-RGB image, convert to RGB and save again
        for i, filepath in enumerate(files):

            try:
                # Open image, transform to tensor
                transform_tensor = transforms.Compose([transforms.ToTensor()])
                image_rgb_tensor = transform_tensor(Image.open(filepath))

                # Check channels and if not 3 (RGB), convert to RGB and save
                if image_rgb_tensor.shape[0] != 3:
                    print(f"- Found a non-RGB '{image_type}'' image, converting it to a 3 CHANNEL RGB image.")
                    image_rgb = Image.open(filepath).convert("RGB")
                    image_rgb.save(filepath)

            except Exception as e:
                raise Exception(e)

    def __transform_CHANNELS2GRAY(self, files, image_type) -> List:

        # For every not-RGB image, convert to RGB and save again
        for i, filepath in enumerate(files):

            try:
                # Open image, transform to tensor
                transform_tensor = transforms.Compose([transforms.ToTensor()])
                grayscale_tensor = transform_tensor(Image.open(filepath))

                # Check channels and if not 3 (RGB), convert to RGB and save
                if grayscale_tensor.shape[0] != 1:
                    print(
                        f"- Found a non-GRAYSCALE '{image_type}' image, converting it to a 1 CHANNEL GRAYSCALE image."
                    )
                    image_grayscale = Image.open(filepath).convert("L")
                    image_grayscale.save(filepath)

            except Exception as e:
                raise Exception(e)

    def __getitem__(self, index):

        item_A_l = self.transform_RGB(Image.open(self.stereo_l[index % len(self.stereo_l)]))
        item_A_r = self.transform_RGB(Image.open(self.stereo_r[index % len(self.stereo_r)]))
        disparity = self.transform_GRAY(Image.open(self.disparity[index % len(self.disparity)]))

        return {"A_left": item_A_l, "A_right": item_A_r, "B": disparity}

    def __len__(self):
        return max(len(self.stereo_l), len(self.disparity))

