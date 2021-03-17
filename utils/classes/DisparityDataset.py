import os
import cv2
import glob
import random
import numpy as np


from PIL import Image
from typing import List
from torch.utils.data import Dataset
from torchvision import transforms


class DisparityDataset(Dataset):

    """ Insert documentation for ImageDataset class """

    def __init__(self, root, mode="train", transform=None):

        # Get transformation and alignment
        self.transform = transform

        # Sort stereo left/right and corresponding disparity
        self.stereo_l = sorted(glob.glob(os.path.join(root, f"{mode}/A/left") + "/*.*"))
        self.stereo_r = sorted(glob.glob(os.path.join(root, f"{mode}/A/right") + "/*.*"))
        self.disparity = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))

        # Transform grayscale images to grayscale RGB
        self.__transform_GRAY2RGB(self.stereo_l)
        self.__transform_GRAY2RGB(self.stereo_r)
        self.__transform_GRAY2RGB(self.disparity)

        # Import files again, some may have been transformed in the prior step
        self.stereo_l = sorted(glob.glob(os.path.join(root, f"{mode}/A/left") + "/*.*"))
        self.stereo_r = sorted(glob.glob(os.path.join(root, f"{mode}/A/right") + "/*.*"))
        self.disparity = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))

    def __transform_GRAY2RGB(self, files) -> List:

        # For every not-RGB image, convert to RGB and save again
        for i, filepath in enumerate(files):

            try:
                # Open image, transform to tensor
                transform_tensor = transforms.Compose([transforms.ToTensor()])
                image_rgb_tensor = transform_tensor(Image.open(filepath))

                # Check channels and if not 3 (RGB), convert to RGB and save
                if image_rgb_tensor.shape[0] != 3:
                    image_rgb = Image.open(filepath).convert("RGB")
                    image_rgb.save(filepath)

            except Exception as e:
                raise Exception(e)

    def __getitem__(self, index):

        item_A_l = self.transform(Image.open(self.stereo_l[index % len(self.stereo_l)]))
        item_A_r = self.transform(Image.open(self.stereo_r[index % len(self.stereo_r)]))
        disparity = self.transform(Image.open(self.disparity[index % len(self.disparity)]))

        return {"A_left": item_A_l, "A_right": item_A_r, "B": disparity}

    def __len__(self):
        return max(len(self.stereo_l), len(self.disparity))

