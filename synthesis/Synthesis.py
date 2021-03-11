#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import torch
import hashlib
import zipfile
import argparse
import numpy as np

import PIL.Image as PIL
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms, datasets

from six.moves import urllib

from synthesis_v2.layers import disp_to_depth
from synthesis_v2.utils import download_model_if_doesnt_exist
from synthesis_v2.networks.depth_decoder import DepthDecoder
from synthesis_v2.networks.resnet_encoder import ResnetEncoder


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls["resnet{}".format(num_layers)])
        loaded["conv1.weight"] = torch.cat([loaded["conv1.weight"]] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


"""
    [TO-DO] The code will get a different location. Underneath perhaps too.

"""


class Image:

    """ Insert class documentation """

    def __init__(
        self,
        imdir_source: str = f"mono_640x192",
        imdir_target: str = f"mono_640x192",
        image_ftype: str = f"synthesis_v2/models",
        path: str = None,
        feed_height: int = None,
        feed_width: int = None,
    ):

        self.SOURCE_DIR: str = imdir_source
        self.TARGET_DIR: str = imdir_target
        self.FILE_TYPE: str = image_ftype

        pass


class Model:

    """ Insert class documentation """

    def __init__(
        self,
        name: str = f"mono_640x192",
        root: str = f"synthesis_v2/models",
        path: str = f"synthesis_v2/models/mono_640x192",
        feed_height: int = None,
        feed_width: int = None,
    ):

        self.NAME = name
        self.ROOT = root
        self.PATH = path
        self.FEED_HEIGHT = feed_height
        self.FEED_WIDTH = feed_width


class Synthesis:

    """ Insert class documentation """

    def __init__(
        self,
        model_name: str = f"mono_640x192",
        model_root: str = f"synthesis_v2/models",
        imdir_source: str = f"synthesis_v2/assets/A",
        imdir_target: str = f"synthesis_v2/assets/B",
        image_ftype: str = f"png",
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):

        """ [TO-DO] Fix those strings with str.format(x) """

        # Constants: system
        self.DEVICE = device

        # Model class (network)
        self.model = Model(
            name=model_name,
            root=model_root,
            path=os.path.join(model_root, model_name),
            feed_height=None,
            feed_width=None,
        )

        # Image class (data)
        self.image = Image(imdir_source=imdir_source, imdir_target=imdir_target, image_ftype=image_ftype,)

        # Networks
        self.encoder = None
        self.decoder = None

        # Variables
        self.paths = None
        self.output_directory = None

        # Make required directories for the source and target images
        self._makedirs()
        self._download_models_if_none()

    """ Public methods """

    def load_models(self) -> None:

        """ Insert documentation """

        # Get encoder- and decoder paths
        image_encoder_path = os.path.join(self.model.PATH, "encoder.pth")
        depth_decoder_path = os.path.join(self.model.PATH, "depth.pth")

        # Instantiate a encoder class and a decoder class
        self.encoder = ResnetEncoder(18, False)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        # Load them into a torch pickle (yuck!)
        loaded_dict_enc = torch.load(image_encoder_path, map_location=self.DEVICE)
        loaded_dict_dec = torch.load(depth_decoder_path, map_location=self.DEVICE)

        # Extract the height and width of image that this model was trained with
        self.model.FEED_HEIGHT = loaded_dict_enc["height"]
        self.model.FEED_WIDTH = loaded_dict_enc["width"]

        # Make a dictionary of the parameters
        filter_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        loaded_dict_dec = torch.load(depth_decoder_path, map_location=self.DEVICE)

        # Load both dictionaries
        self.encoder.load_state_dict(filter_dict_enc)
        self.decoder.load_state_dict(loaded_dict_dec)

        # Fit both the encoder- and decoder to our device
        self.encoder.to(self.DEVICE).eval()
        self.decoder.to(self.DEVICE).eval()

        # When the source is a single image, set path and output directory
        if os.path.isfile(self.image.SOURCE_DIR):
            self.paths = [self.image.SOURCE_DIR]
            self.output_directory = os.path.dirname(self.image.TARGET_DIR)

        # When the source is an image folder, set path and output directory
        elif os.path.isdir(self.image.SOURCE_DIR):
            self.paths = glob.glob(os.path.join(self.image.SOURCE_DIR, f"*.{self.image.FILE_TYPE}"))
            self.output_directory = self.image.TARGET_DIR

        else:
            raise Exception("Can't find image source path: {self.image.SOURCE_DIR}")

        print(f"Estimating the depth of {len(self.paths)} images")
        pass

    def estimate_depth_v2(self) -> None:

        """ Insert documentation """

        with torch.no_grad():
            for idx, image_path in enumerate(self.paths):

                """ [TO-DO] Check this """
                # Make sure that no disparity predictions are made on disparity images
                # Obsolute now, since there are A and B folders.
                if image_path.endswith("_disp.jpg"):
                    continue

                # Load image and preprocess
                input_image = PIL.open(image_path).convert("RGB")
                original_width, original_height = input_image.size
                input_image = input_image.resize((self.model.FEED_WIDTH, self.model.FEED_HEIGHT), PIL.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                # Make the prediction
                input_image = input_image.to(self.DEVICE)
                features = self.encoder(input_image)
                outputs = self.decoder(features)

                # Extract the disparity values from the depth encoder output
                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False
                )

                # Transform the disparity tensor over the input image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = PIL.fromarray(colormapped_im)

                # Save the predicted disparity map
                OUTPUT_NAME = os.path.splitext(os.path.basename(image_path))[0]
                name_dest_im = os.path.join(self.output_directory, f"{OUTPUT_NAME}_depth.{self.image.FILE_TYPE}")
                im.save(name_dest_im)

                print(f"Processed {idx + 1} of {len(self.paths)} images - saved prediction to: {name_dest_im}")

        pass

    """ Private methods """

    def _makedirs(self) -> None:

        """ Create required source- and target folders, if none exist """

        try:
            os.makedirs(os.path.join(self.image.SOURCE_DIR))
            os.makedirs(os.path.join(self.image.TARGET_DIR))
        except OSError:
            pass

    def _download_models_if_none(self) -> None:

        """ Insert documentation """

        # If pretrained kitti model doesn't exist, download and unzip it.
        # Values are tuples of (<google cloud URL>, <md5 checksum>)
        download_paths = {
            "mono_640x192": (
                "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
                "a964b8356e08a02d009609d9e3928f7c",
            ),
            "stereo_640x192": (
                "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
                "3dfb76bcff0786e4ec07ac00f658dd07",
            ),
            "mono+stereo_640x192": (
                "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
                "c024d69012485ed05d7eaa9617a96b81",
            ),
            "mono_no_pt_640x192": (
                "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
                "9c2f071e35027c895a4728358ffc913a",
            ),
            "stereo_no_pt_640x192": (
                "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
                "41ec2de112905f85541ac33a854742d1",
            ),
            "mono+stereo_no_pt_640x192": (
                "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
                "46c3b824f541d143a45c37df65fbab0a",
            ),
            "mono_1024x320": (
                "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
                "0ab0766efdfeea89a0d9ea8ba90e1e63",
            ),
            "stereo_1024x320": (
                "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
                "afc2f2126d70cf3fdf26b550898b501a",
            ),
            "mono+stereo_1024x320": (
                "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
                "cdc5fc9b23513c07d5b19235d9ef08f7",
            ),
        }

        # Check whether the path exists, makedir if not
        if not os.path.exists(self.model.ROOT):
            os.makedirs(self.model.ROOT)

        # Check whether the file matches MD5
        def check_file_matches_md5(checksum, fpath):
            if not os.path.exists(fpath):
                return False
            with open(fpath, "rb") as f:
                current_md5checksum = hashlib.md5(f.read()).hexdigest()
            return current_md5checksum == checksum

        print(self.model.PATH)
        return
        # Check whether the model has already been downloaded
        if not os.path.exists(os.path.join(self.model.PATH, "encoder.pth")):

            model_url, required_md5checksum = download_paths[self.model.NAME]

            if not check_file_matches_md5(required_md5checksum, self.model.PATH + ".zip"):
                print("Downloading pretrained model to {}".format(self.model.PATH + ".zip"))
                urllib.request.urlretrieve(model_url, self.model.PATH + ".zip")

            if not check_file_matches_md5(required_md5checksum, self.model.PATH + ".zip"):
                print("Failed to download a file which matches the checksum - quitting")
                quit()

            print("Unzipping model...")
            with zipfile.ZipFile(self.model.PATH + ".zip", "r") as f:
                f.extractall(self.model.PATH)

            print("Model unzipped to {}".format(self.model.PATH))

        pass

    def __sizeof__(self) -> int:
        pass

    def __repr__(self) -> str:
        return f"Synthesis(model.NAME={self.model.NAME})"
