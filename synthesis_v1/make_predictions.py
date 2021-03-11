#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import torch
import argparse
import numpy as np
import PIL.Image as PIL
import matplotlib as mpl
import matplotlib.cm as cm

from torchvision import transforms, datasets

from synthesis_v1.layers import disp_to_depth
from synthesis_v1.utils import download_model_if_doesnt_exist
from synthesis_v1.networks.depth_decoder import DepthDecoder
from synthesis_v1.networks.resnet_encoder import ResnetEncoder

# Test function that produces a depth map
def estimate_depth_v1():

    """ Insert documentation && rewrite function into cleaner code """

    # Clear terminal
    os.system("cls")

    # Constants: list of all model names, just for quick-access
    MODEL_LIST = [
        "mono_640x192",
        "stereo_640x192",
        "mono+stereo_640x192",
        "mono_no_pt_640x192",
        "stereo_no_pt_640x192",
        "mono+stereo_no_pt_640x192",
        "mono_1024x320",
        "stereo_1024x320",
        "mono+stereo_1024x320",
    ]

    # Constants: network model
    MODEL_NAME = f"mono_640x192"
    MODEL_ROOT = f"synthesis/models"

    # Constants: directories and image file type
    IMAGE_SOURCE = "A"
    IMAGE_TARGET = "B"
    IMDIR_SOURCE = f"synthesis_v1/assets/{IMAGE_SOURCE}"
    IMDIR_TARGET = f"synthesis_v1/assets/{IMAGE_TARGET}"
    IMAGE_FTYPE = f"png"

    # Make required directories for the source and target images
    try:
        os.makedirs(os.path.join(IMDIR_SOURCE))
        os.makedirs(os.path.join(IMDIR_TARGET))
    except OSError:
        pass

    # Constants: system
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Function to predict for a single image or folder of images
    """

    download_model_if_doesnt_exist(MODEL_ROOT=MODEL_ROOT, MODEL_NAME=MODEL_NAME)
    MODEL_PATH = os.path.join(MODEL_ROOT, MODEL_NAME)
    print(f"Loading model from: {MODEL_PATH}")
    encoder_path = os.path.join(MODEL_PATH, "encoder.pth")
    depth_decoder_path = os.path.join(MODEL_PATH, "depth.pth")

    # Load the selected pre-trained model
    print(f"Loading pre-trained encoder: {encoder_path}")
    encoder = ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=DEVICE)

    # Extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    # Fit the encoder to our device
    encoder.to(DEVICE)
    encoder.eval()

    # Load the decoder with corresponding data
    print(f"Loading pre-trained decoder: {depth_decoder_path}")
    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=DEVICE)
    depth_decoder.load_state_dict(loaded_dict)

    # Fit the decoder to our device
    depth_decoder.to(DEVICE)
    depth_decoder.eval()

    # Check whether IMDIR_SOURCE is a single image or a folder and set paths and output_dir accordingly
    if os.path.isfile(IMDIR_SOURCE):
        paths = [IMDIR_SOURCE]
        output_directory = os.path.dirname(IMDIR_TARGET)
    elif os.path.isdir(IMDIR_SOURCE):
        paths = glob.glob(os.path.join(IMDIR_SOURCE, "*.{}".format(IMAGE_FTYPE)))
        output_directory = IMDIR_TARGET
    else:
        raise Exception("Can not find IMDIR_SOURCE: {}".format(IMDIR_SOURCE))

    print("Predicting on {:d} test images".format(len(paths)))

    # Iterate over the images and make a prediction
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            # Make sure that no disparity predictions are made on disparity images
            # Obsolute now, since there are A and B folders.
            if image_path.endswith("_disp.jpg"):
                continue

            # Load image and preprocess
            input_image = PIL.open(image_path).convert("RGB")
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), PIL.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # Make the prediction
            input_image = input_image.to(DEVICE)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            # Extract the disparity values from the depth encoder output
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False
            )

            """
            # # Save the predicted disparity map to a numpy array
            # output_name = os.path.splitext(os.path.basename(image_path))[0]
            # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            # scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            """

            # Transform the disparity tensor over the input image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = PIL.fromarray(colormapped_im)

            # Save the predicted disparity map
            OUTPUT_NAME = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_im = os.path.join(output_directory, f"{OUTPUT_NAME}_depth.{IMAGE_FTYPE}")
            im.save(name_dest_im)

            print("Processed {:d} of {:d} images - saved prediction to {}".format(idx + 1, len(paths), name_dest_im))

    pass
