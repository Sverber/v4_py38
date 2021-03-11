# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib


def download_model_if_doesnt_exist(MODEL_ROOT, MODEL_NAME):
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


# def readlines(filename):
#     """Read all the lines in a text file and return as a list
#     """
#     with open(filename, "r") as f:
#         lines = f.read().splitlines()
#     return lines


# def normalize_image(x):
#     """Rescale image pixels to span range [0, 1]
#     """
#     ma = float(x.max().cpu().data)
#     mi = float(x.min().cpu().data)
#     d = ma - mi if ma != mi else 1e5
#     return (x - mi) / d


# def sec_to_hm(t):
#     """Convert time in seconds to time in hours, minutes and seconds
#     e.g. 10239 -> (2, 50, 39)
#     """
#     t = int(t)
#     s = t % 60
#     t //= 60
#     m = t % 60
#     t //= 60
#     return t, m, s


# def sec_to_hm_str(t):
#     """Convert time in seconds to a nice string
#     e.g. 10239 -> '02h50m39s'
#     """
#     h, m, s = sec_to_hm(t)
#     return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

