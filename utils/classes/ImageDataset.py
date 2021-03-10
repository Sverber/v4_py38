import os
import cv2
import glob
import random
import numpy as np


from PIL import Image
from typing import List
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):

    """ Insert documentation for ImageDataset class """

    def __init__(self, root, mode="train", transform=None, unaligned=False):
        self.transform = transform
        self.unaligned = unaligned

        # Sort files from folder A and B
        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}/A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))

        # Transform grayscale images to grayscale RGB
        self.files_A = self._transform_GRAY2RGB(self.files_A)
        self.files_B = self._transform_GRAY2RGB(self.files_B)

        # Import files again, some may have been transformed in the prior step
        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}/A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))

    def _transform_GRAY2RGB(self, files) -> List:

        # For every not-RGB image, convert to RGB and save again
        for i, filepath in enumerate(files):

            # Open image, transform to tensor
            transform_tensor = transforms.Compose([transforms.ToTensor()])
            image_rgb_tensor = transform_tensor(Image.open(filepath))

            # Check channels and if not 3 (RGB), convert to RGB and save
            if image_rgb_tensor.shape[0] is not 3:
                image_rgb = Image.open(filepath).convert("RGB")
                image_rgb.save(filepath)

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
