#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import glob
import torch
import hashlib
import zipfile
import numpy as np

import PIL.Image as PIL
import matplotlib as mpl
import matplotlib.cm as cm

from six.moves import urllib

from torchvision import transforms, datasets

# from .layers import disp_to_depth
from .networks.depth_decoder import DepthDecoder
from .networks.resnet_encoder import ResnetEncoder


class Synthesis:

    """ Insert class documentation """

    def __init__(
        self,
        mode: str = f"train",
        file_type: str = f"png",
        model_root: str = f"synthesis/models",
        model_name: str = f"mono_640x192",
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):

        # Constants: system and model
        self.DEVICE = device
        self.MODE = mode
        self.MODEL_ROOT = model_root
        self.MODEL_NAME = model_name

        # Constants: assets dirs and file type
        self.SOURCE_DIR = f"synthesis/assets/{mode}/A"
        self.TARGET_DIR = f"synthesis/assets/{mode}/B"
        self.FILE_TYPE = file_type

        # Networks
        self.encoder = None
        self.decoder = None

        # Variables
        self.feed_height = None
        self.feed_width = None
        self.paths = None
        self.output_directory = None

        # Make the required directories for the source and target images
        self.__makedirs()

        # Download the models
        self.__download_models_if_none_exist(model_root, model_name)

        # Load the models
        self.__load_models(model_root, model_name)

    """ Public methods """

    def predict_depth(self) -> None:

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
                input_image = input_image.resize((self.feed_width, self.feed_height), PIL.LANCZOS)
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
                name_dest_im = os.path.join(self.output_directory, f"{OUTPUT_NAME}_depth.{self.FILE_TYPE}")
                im.save(name_dest_im)

                print(f"Processed {idx + 1} of {len(self.paths)} images - saved prediction to: {name_dest_im}")

    """ Private methods """

    def __makedirs(self) -> None:

        """ Insert documentation """

        try:
            os.makedirs(self.SOURCE_DIR)
            os.makedirs(self.TARGET_DIR)
        except OSError:
            pass

    def __download_models_if_none_exist(self, MODEL_ROOT, MODEL_NAME):

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

        # Check whether the path exists
        if not os.path.exists(MODEL_ROOT):
            os.makedirs(MODEL_ROOT)

        # Join path with the specified model name
        MODEL_PATH = os.path.join(MODEL_ROOT, MODEL_NAME)

        def check_file_matches_md5(checksum, fpath):
            if not os.path.exists(fpath):
                return False
            with open(fpath, "rb") as f:
                current_md5checksum = hashlib.md5(f.read()).hexdigest()
            return current_md5checksum == checksum

        # Check whether the model has already been downloaded
        if not os.path.exists(os.path.join(MODEL_PATH, "encoder.pth")):

            model_url, required_md5checksum = download_paths[MODEL_NAME]

            if not check_file_matches_md5(required_md5checksum, MODEL_PATH + ".zip"):
                print("Downloading pretrained model to {}".format(MODEL_PATH + ".zip"))
                urllib.request.urlretrieve(model_url, MODEL_PATH + ".zip")

            if not check_file_matches_md5(required_md5checksum, MODEL_PATH + ".zip"):
                print("Failed to download a file which matches the checksum - quitting")
                quit()

            print("Unzipping model...")
            with zipfile.ZipFile(MODEL_PATH + ".zip", "r") as f:
                f.extractall(MODEL_PATH)

            print("Model unzipped to {}".format(MODEL_PATH))

    def __load_models(self, MODEL_ROOT, MODEL_NAME) -> None:

        """ Insert documentation """

        # Make models directory
        if not os.path.exists(MODEL_ROOT):
            os.makedirs(MODEL_ROOT)

        # Define models path
        MODEL_PATH = os.path.join(MODEL_ROOT, MODEL_NAME)

        # Get encoder- and decoder paths
        image_encoder_path = os.path.join(MODEL_PATH, "encoder.pth")
        depth_decoder_path = os.path.join(MODEL_PATH, "depth.pth")

        # Instantiate a encoder class and a decoder class
        self.encoder = ResnetEncoder(18, False)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        # Load them into a torch pickle (yuck!)
        loaded_dict_enc = torch.load(image_encoder_path, map_location=self.DEVICE)
        loaded_dict_dec = torch.load(depth_decoder_path, map_location=self.DEVICE)

        # Extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc["height"]
        self.feed_width = loaded_dict_enc["width"]

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
        if os.path.isfile(self.SOURCE_DIR):
            self.paths = [self.SOURCE_DIR]
            self.output_directory = os.path.dirname(self.TARGET_DIR)

        # When the source is an image folder, set path and output directory
        elif os.path.isdir(self.SOURCE_DIR):
            self.paths = glob.glob(os.path.join(self.SOURCE_DIR, f"*.{self.FILE_TYPE}"))
            self.output_directory = self.TARGET_DIR

        # Raise an Exception if all else fails
        else:
            raise Exception("Can't find image source path")

        print(f"Estimating the depth of {len(self.paths)} images")

    def __repr__(self) -> str:

        """ Insert documentation """

        return f"<class>\n\
        Synthesis(\n\
            model_name = {self.MODEL_NAME},\n\
            model_root = {self.MODEL_ROOT},\n\
            num_assets = {len(self.paths)},\n\
        )\n"

