#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from torchvision.transforms.transforms import Grayscale
import torchvision.utils as vutils
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim

from torch.utils.data.dataloader import DataLoader

from synthesis.Synthesis import Synthesis

from utils.functions.initialize_weights import initialize_weights

from utils.classes.DecayLR import DecayLR
from utils.classes.ReplayBuffer import ReplayBuffer
from utils.classes.RunCycleBuilder import RunCycleBuilder
from utils.classes.RunCycleManager import RunCycleManager
from utils.classes.StereoDisparityDataset import StereoDisparityDataset

from utils.models.cycle.Discriminator import Discriminator
from utils.models.cycle.Generators import Generator


# Constants: required directories
DIR_DATASET = f"./dataset"
DIR_OUTPUTS = f"./outputs"
DIR_RESULTS = f"./results"
DIR_WEIGHTS = f"./weights"


# Testing function
def test(
    PARAMETERS: OrderedDict,
    NAME_S2D_DATASET: str,
    SHOW_IMG_FREQ: int,
    dataset: StereoDisparityDataset,
    path_to_folder: str,
    model_netG_A2B: str,
    model_netG_B2A: str,
) -> None:

    """ Insert documentation """

    # Iterate over every run, based on the configurated params
    for run in RunCycleBuilder.get_runs(PARAMETERS):

        # Store today's date in string format
        TODAY_DATE = datetime.today().strftime("%Y-%m-%d")
        TODAY_TIME = datetime.today().strftime("%H.%M.%S")

        # Create a unique name for this run
        RUN_NAME = f"{TODAY_TIME}___EP{run.num_epochs}_DE{run.decay_epochs}_LR{run.learning_rate}_BS{run.batch_size}"
        RUN_PATH = f"{NAME_S2D_DATASET}/{TODAY_DATE}/{RUN_NAME}"

        # Make required directories for testing
        try:
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "A"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "B"))
        except OSError:
            pass

        # Allow cuddn to look for the optimal set of algorithms to improve runtime speed
        cudnn.benchmark = True

        # Dataloader
        loader = DataLoader(
            dataset=dataset, batch_size=run.batch_size, num_workers=run.num_workers, shuffle=run.shuffle
        )

        # Create model
        netG_A2B = Generator(in_channels=3, out_channels=3).to(run.device)
        netG_B2A = Generator(in_channels=3, out_channels=3).to(run.device)

        # Load state dicts
        netG_A2B.load_state_dict(torch.load(os.path.join(str(path_to_folder), model_netG_A2B)))
        netG_B2A.load_state_dict(torch.load(os.path.join(str(path_to_folder), model_netG_B2A)))

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
            filepath_real_A, filepath_real_B, filepath_fake_A, filepath_fake_B, filepath_f_or_A, filepath_f_or_B = (
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___fake_sample_MSE{mse_loss_A:.3f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___fake_sample_MSE{mse_loss_B:.3f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___fake_original_MSE{mse_loss_f_or_A:.3f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___fake_original_MSE{mse_loss_f_or_B:.3f}.png",
            )

            # Save images
            if (i + 1) % SHOW_IMG_FREQ == 0:
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

