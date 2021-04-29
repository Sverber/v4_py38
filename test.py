#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import cv2
import sys
import numpy as np
import PIL.Image as PIL
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

from torch.utils.data.dataloader import DataLoader

from utils.classes.dataloaders import MyDataLoader
from utils.classes.RunCycleBuilder import RunCycleBuilder
from utils.classes.RunCycleManager import RunCycleManager
from utils.classes.StereoDisparityDataset import StereoDisparityDataset
from utils.models.cycle.Generators import Generator


# Constants: required directories
DIR_DATASET = f"./dataset"
DIR_OUTPUTS = f"./outputs"
DIR_RESULTS = f"./results"
DIR_WEIGHTS = f"./weights"


PARAMETERS: OrderedDict = OrderedDict(
    device=[torch.device("cuda" if torch.cuda.is_available() else "cpu")],
    shuffle=[True],
    num_workers=[8],
    manualSeed=[999],
    learning_rate=[0.0002],
    batch_size=[1],
    num_epochs=[100],
    decay_epochs=[50],
)


# Initialize weights
def initialize_weights(m):

    """ Custom weights initialization called on net_G and net_D """

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

    pass


# Testing function
def test(
    dataset: StereoDisparityDataset,
    parameters: OrderedDict,
    channels: int,
    #
    dataset_group: str,
    dataset_name: str,
    model_name: str,
    model_date: str,
    #
    model_netG_A2B: str,
    model_netG_B2A: str,
) -> None:

    """ Insert documentation """

    path_to_folder = f"weights/{dataset_group}/{dataset_name}/{model_date}/{model_name}"

    # Iterate over every run, based on the configurated params
    for run in RunCycleBuilder.get_runs(parameters):

        # Clear occupied CUDA memory
        torch.cuda.empty_cache()

        # Store today's date in string format
        TODAY_DATE = datetime.today().strftime("%Y-%m-%d")
        TODAY_TIME = datetime.today().strftime("%H.%M.%S")

        # Create a unique name for this run
        RUN_NAME = f"{TODAY_TIME}___EP{run.num_epochs}_DE{run.decay_epochs}_LR{run.learning_rate}_BS{run.batch_size}"
        RUN_PATH = f"{dataset_group}/{dataset_name}/{TODAY_DATE}/{RUN_NAME}"

        # Make required directories for testing
        try:
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "A"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "B"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "D"))
        except OSError:
            pass

        # Allow cuddn to look for the optimal set of algorithms to improve runtime speed
        cudnn.benchmark = True

        # Dataloader
        loader = DataLoader(
            dataset=dataset, batch_size=run.batch_size, num_workers=run.num_workers, shuffle=run.shuffle
        )

        # Create model
        netG_A2B = Generator(in_channels=channels, out_channels=channels).to(run.device)
        netG_B2A = Generator(in_channels=channels, out_channels=channels).to(run.device)

        # Load state dicts
        netG_A2B.load_state_dict(torch.load(os.path.join(str(path_to_folder), "net_G_A2B", model_netG_A2B)))
        netG_B2A.load_state_dict(torch.load(os.path.join(str(path_to_folder), "net_G_B2A", model_netG_B2A)))

        # Set model mode
        netG_A2B.eval()
        netG_B2A.eval()

        # Create progress bar
        progress_bar = tqdm(enumerate(loader), total=len(loader))

        # Initiate a mean square error (MSE) loss function
        mse_loss = nn.MSELoss()

        # Initiate mean squared error (MSE) losses variables
        avg_mse_loss_A, avg_mse_loss_B, avg_mse_loss_f_or_A, avg_mse_loss_f_or_B = 0, 0, 0, 0
        cum_mse_loss_A, cum_mse_loss_B, cum_mse_loss_f_or_A, cum_mse_loss_f_or_B = 0, 0, 0, 0

        # Iterate over the data
        for i, data in progress_bar:

            """ (1) Read input data """

            # Get image A and image B
            real_image_A_left = data["A_left"].to(run.device)
            real_image_A_right = data["A_right"].to(run.device)
            real_image_B = data["B"].to(run.device)

            # Concatenate left- and right view into one stereo image
            real_image_A = torch.cat((real_image_A_left, real_image_A_right), dim=-1)
            real_image_B = real_image_B

            """ (2) Generate output """

            # Generate output
            _fake_image_A = netG_B2A(real_image_B)
            _fake_image_B = netG_A2B(real_image_A)

            # Generate original image from generated (fake) output
            _fake_original_image_A = netG_B2A(_fake_image_B)
            _fake_original_image_B = netG_A2B(_fake_image_A)

            """ (1) Convert to usable images """

            # Convert to usable images
            fake_image_A = 0.5 * (_fake_image_A.data + 1.0)
            fake_image_B = 0.5 * (_fake_image_B.data + 1.0)

            # Convert to usable images
            fake_original_image_A = 0.5 * (_fake_original_image_A.data + 1.0)
            fake_original_image_B = 0.5 * (_fake_original_image_B.data + 1.0)

            """ (1) Calculate losses for the generated (fake) output """

            # Calculate the mean square error (MSE) loss
            mse_loss_A = mse_loss(fake_image_A, real_image_B)
            mse_loss_B = mse_loss(fake_image_B, real_image_A)

            # Calculate the sum of all mean square error (MSE) losses
            cum_mse_loss_A += mse_loss_A
            cum_mse_loss_B += mse_loss_B

            # Calculate the average mean square error (MSE) loss
            avg_mse_loss_A = cum_mse_loss_A / (i + 1)
            avg_mse_loss_B = cum_mse_loss_B / (i + 1)

            """ (1) Calculate losses for the generated (fake) original images """

            # Calculate the mean square error (MSE) for the generated (fake) originals A and B
            mse_loss_f_or_A = mse_loss(fake_original_image_A, real_image_A)
            mse_loss_f_or_B = mse_loss(fake_original_image_B, real_image_B)

            # Calculate the average mean square error (MSE) for the fake originals A and B
            cum_mse_loss_f_or_A += mse_loss_f_or_A
            cum_mse_loss_f_or_B += mse_loss_f_or_B

            # Calculate the average mean square error (MSE) for the fake originals A and B
            avg_mse_loss_f_or_A = cum_mse_loss_f_or_A / (i + 1)
            avg_mse_loss_f_or_B = cum_mse_loss_f_or_B / (i + 1)

            """ (1) Define filepaths, save generated output and print a progress bar """

            # Filepath and filename for the real and generated output images
            (
                filepath_disparity_real,
                filepath_disparity_fake,
                filepath_real_A,
                filepath_real_B,
                filepath_fake_A,
                filepath_fake_B,
                filepath_f_or_A,
                filepath_f_or_B,
            ) = (
                f"{DIR_RESULTS}/{RUN_PATH}/D/{i + 1:04d}___disparity_real.png",
                f"{DIR_RESULTS}/{RUN_PATH}/D/{i + 1:04d}___disparity_fake.png",
                #
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___fake_sample_MSE{mse_loss_A:.3f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___fake_sample_MSE{mse_loss_B:.3f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___fake_original_MSE{mse_loss_f_or_A:.3f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___fake_original_MSE{mse_loss_f_or_B:.3f}.png",
            )

            # Save real input images
            vutils.save_image(real_image_A.detach(), filepath_real_A, normalize=True)
            vutils.save_image(real_image_B.detach(), filepath_real_B, normalize=True)

            # Save generated (fake) output images
            vutils.save_image(fake_image_A.detach(), filepath_fake_A, normalize=True)
            vutils.save_image(fake_image_B.detach(), filepath_fake_B, normalize=True)

            # Save generated (fake) original images
            vutils.save_image(fake_original_image_A.detach(), filepath_f_or_A, normalize=True)
            vutils.save_image(fake_original_image_B.detach(), filepath_f_or_B, normalize=True)

            # Save generated (fake) original images
            vutils.save_image(fake_original_image_A.detach(), filepath_f_or_A, normalize=True)
            vutils.save_image(fake_original_image_B.detach(), filepath_f_or_B, normalize=True)

            """ Convert to disparity maps """

            def __convert_disparty(image_np, path) -> np.ndarray:

                vmin = image_np.min()
                vmax = np.percentile(image_np, 100)

                # cmaps = [ 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

                # Normalize using the provided cmap
                normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
                colormapped_im = (mapper.to_rgba(image_np)[:, :, :3] * 255).astype(np.uint8)

                # Convert numpy array to PIL
                image_target = PIL.fromarray(colormapped_im)

                # Save
                image_target.save(path)

                return image_target

            # Convert images to numpy array
            np_image_real_A = real_image_B.squeeze().cpu().numpy()
            np_image_fake_B = real_image_B.squeeze().cpu().numpy()

            # Convert to disparity and save to path
            disparity_image_real: np.ndarray = __convert_disparty(np_image_real_A, filepath_disparity_real)
            disparity_image_fake: np.ndarray = __convert_disparty(np_image_fake_B, filepath_disparity_fake)

            disparity_image_real.show()
            disparity_image_fake.show()

            """ Bilateral filter """

            def __bilateral_filter(image) -> np.ndarray:

                open_cv_image = np.array(image)
                open_cv_image = open_cv_image[:, :, ::-1].copy()

                bilateral_image = cv2.bilateralFilter(open_cv_image, 9, 75, 75)

                return PIL.fromarray(bilateral_image)

            # # Run a bilateral filter over the images
            # bilateral_image_real: np.ndarray = __bilateral_filter(np_image_real_A)
            # bilateral_image_fake: np.ndarray = __bilateral_filter(np_image_fake_B)

            # # Show
            # bilateral_image_real.show()
            # bilateral_image_fake.show()

            """ Progress bar """

            # Print a progress bar in terminal
            progress_bar.set_description(f"Process images {i + 1} of {len(loader)}")

        # Calculate average mean squared error (MSE)
        avg_mse_loss_A = cum_mse_loss_A / len(loader)
        avg_mse_loss_B = cum_mse_loss_B / len(loader)

        # Calculate average mean squared error (MSE)
        avg_mse_loss_f_or_A = cum_mse_loss_f_or_A / len(loader)
        avg_mse_loss_f_or_B = cum_mse_loss_f_or_B / len(loader)

        # Print
        print("MSE(avg) fake_image_A:", avg_mse_loss_A)
        print("MSE(avg) fake_image_B:", avg_mse_loss_B)
        print("MSE(avg) fake_original_image_A:", avg_mse_loss_f_or_A)
        print("MSE(avg) fake_original_image_B:", avg_mse_loss_f_or_B)

    # </end> def test():
    pass


# Execute main code
if __name__ == "__main__":

    try:

        mydataloader = MyDataLoader()

        s2d_dataset_test_RGB_DISPARITY = mydataloader.get_dataset(
            "s2d", "Test_Set_RGB_DISPARITY", "test", (164, 276), 1, False
        )

        test(
            parameters=PARAMETERS,
            dataset=s2d_dataset_test_RGB_DISPARITY,
            channels=1,
            #
            dataset_group="s2d",
            dataset_name="Test_Set_RGB_DISPARITY",
            model_date="2021-04-28",
            model_name="11.51.48___EP100_DE050_LR0.0002_CH1",
            #
            model_netG_A2B=f"net_G_A2B.pth",
            model_netG_B2A=f"net_G_B2A.pth",
        )

        pass

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
