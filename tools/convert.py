#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import sys
import cv2
import glob
import random
import numpy as np
import torchvision.transforms as transforms

import PIL.Image as PIL
import matplotlib as mpl
import matplotlib.cm as cm

from PIL import Image, ImageOps
from torchvision import transforms


def transform_RGB2INVERTED_RGB(path) -> None:

    images = sorted(glob.glob(path))
    digits = len(str(len(images)))

    for i, path in enumerate(images):

        image = Image.open(path)

        # Open image, transform to tensor, read its channels
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image)
        image_channels = image_tensor.shape[0]

        # Equality check
        equality = int(image_channels) == 3

        if equality:
            # Invert colours and save
            inverted_image = ImageOps.invert(Image.open(path))
            inverted_image.save(path)

            # Verbose
            print(f"{path} | [{str(i + 1).zfill(digits)}/{str(len(images)).zfill(digits)}] Inverted image colours")

        else:
            print(f"{path} | [{str(i + 1).zfill(digits)}/{str(len(images)).zfill(digits)}] Is not a RGB image")


def transform_CHANNELS2xCHANNELS(path, to_channels: int = 3) -> None:

    images = sorted(glob.glob(path))
    digits = len(str(len(images)))

    print("- Path:", path)
    print("- Length(images):", len(images))

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
            if to_channels == 1:
                image_target = Image.open(path).convert("L")
            elif to_channels == 3:
                image_target = Image.open(path).convert("RGB")
            else:
                raise Exception(f"- Can not convert the image to given channels: {to_channels}")

            image_target.save(path)

            print(
                f"{prefix} Image channels: {image_channels}, must be: {to_channels} | Equality: {equality} | Converted image ({i + 1}) to {to_channels} channels."
            )

        else:
            print(f"{prefix} Image channels: {image_channels}, must be: {to_channels} | Equality: {equality} ")


def transform_GRAYSCALE2DISPARITY(
    path, save: bool = True, cmap: str = "magma", show_samples: bool = True, num_samples: int = 1
) -> None:

    images = sorted(glob.glob(path))
    digits = len(str(len(images)))

    print("- Path:", path)
    print("- Length(images):", len(images))

    for i, path in enumerate(images):

        image = Image.open(path)

        # Open image, transform to tensor, read its channels
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image)
        image_channels = image_tensor.shape[0]

        # Equality check
        equality = int(image_channels) == 1

        # Verbose
        prefix = f"{path} | [{str(i + 1).zfill(digits)} / {str(len(images)).zfill(digits)}]"

        if equality:

            image_np = image_tensor.squeeze().cpu().numpy()

            vmin = image_np.min()
            vmax = np.percentile(image_np, 100)

            # cmaps = [ 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

            # Normalize using the provided cmap
            normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
            colormapped_im = (mapper.to_rgba(image_np)[:, :, :3] * 255).astype(np.uint8)

            # Convert numpy array to PIL
            image_target = PIL.fromarray(colormapped_im)

            print(
                f"{prefix} Image channels: {image_channels}, thus grayscale | Equality: {equality} | Converted image to a disparity map."
            )

            if save:
                image_target.save(path)

            continue

            # if show_samples and i == 1:
            #     image_target.show()

            """ Alpha matting (infoFlow) """

            # https://stackoverflow.com/questions/55353735/how-to-do-alpha-matting-in-python

            """ Bilateral filter """

            # def __bilateral_filter(image):

            #     open_cv_image = np.array(image_target)
            #     open_cv_image = open_cv_image[:, :, ::-1].copy()

            #     return cv2.bilateralFilter(open_cv_image, 9, 75, 75)

            # image_target_bilateral = __bilateral_filter(image_target)

            # image_target_bilateral_PIL = PIL.fromarray(image_target_bilateral)

            # # image_target_bf_pil.show()

            # image_target_bilateral = __bilateral_filter(image_target)

            # # Show images
            # image_target.show()
            # image_target_bilateral.show()

            # cv2.imshow("image_target_bf_pil", image_target_bf_pil)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            def __concatenate(images):

                # images = [Image.open(x) for x in ['Test1.jpg', 'Test2.jpg', 'Test3.jpg']]
                widths, heights = zip(*(i.size for i in images))

                total_width = sum(widths)
                max_height = max(heights)

                new_image = Image.new("RGB", (total_width, max_height))

                x_offset = 0

                for im in images:
                    new_image.paste(im, (x_offset, 0))

                x_offset += im.size[0]

                new_image.show()

            # __concatenate([image_target, image_target_bilateral])

        else:
            print(f"{prefix} Image channels: {image_channels}, thus already in RGB format | Equality: {equality}")


# Execute main code
if __name__ == "__main__":

    try:

        """ Transform grayscale to a RGB disparity map """

        # transform_GRAYSCALE2DISPARITY(os.path.join("dataset/s2d/Test_Set_RGB_DISPARITY/train/B" + "/*.*"), save=False)
        transform_GRAYSCALE2DISPARITY(os.path.join("dataset/s2d/Club_I/train/B" + "/*.*"), save=True)

        """ Transform RGB colours to their inverse """

        # transform_RGB2INVERTED_RGB(os.path.join("dataset/s2d/Test_Set_RGB_DISPARITY/train/B" + "/*.*"))

        """ Transform channels to {x} channels """

        # transform_CHANNELS2xCHANNELS(os.path.join("dataset/l2r/Test_Set_GRAYSCALE/train/left" + "/*.*"), 1)
        # transform_CHANNELS2xCHANNELS(os.path.join("dataset/l2r/Test_Set_GRAYSCALE/train/right" + "/*.*"), 1)
        # transform_CHANNELS2xCHANNELS(os.path.join("dataset/l2r/Test_Set_GRAYSCALE/test/left" + "/*.*"), 1)
        # transform_CHANNELS2xCHANNELS(os.path.join("dataset/l2r/Test_Set_GRAYSCALE/test/right" + "/*.*"), 1)

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
