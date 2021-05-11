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
from torchvision import transforms
from collections import OrderedDict

from torch.utils.data.dataloader import DataLoader

from utils.classes.dataloaders import MyDataLoader
from utils.classes.RunCycleBuilder import RunCycleBuilder
from utils.classes.RunCycleManager import RunCycleManager
from utils.classes.StereoDisparityDataset import StereoDisparityDataset
from utils.models.cycle.Generators import Generator


# Clear the terminal
os.system("cls")


# Constants: required directories
DIR_DATASET = f"./dataset"
DIR_OUTPUTS = f"./outputs"
DIR_RESULTS = f"./results"
DIR_WEIGHTS = f"./weights"


PARAMETERS: OrderedDict = OrderedDict(
    device=[torch.device("cuda" if torch.cuda.is_available() else "cpu")],
    shuffle=[False],
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
    extra_note: str,
    #
    model_group: str,
    model_folder: str,
    model_name: str,
    model_date: str,
    #
    model_netG_A2B: str,
    model_netG_B2A: str,
) -> None:

    """ Insert documentation """

    path_to_folder = f"weights/{model_group}/{model_folder}/{model_date}/{model_name}"

    # Iterate over every run, based on the configurated params
    for run in RunCycleBuilder.get_runs(parameters):

        # Clear occupied CUDA memory
        torch.cuda.empty_cache()

        # Store today's date in string format
        TODAY_DATE = datetime.today().strftime("%Y-%m-%d")
        TODAY_TIME = datetime.today().strftime("%H.%M.%S")

        # Create a unique name for this run
        RUN_NAME = f"{TODAY_TIME}___EP{run.num_epochs}_DE{run.decay_epochs}_LR{run.learning_rate}_BS{run.batch_size}_{extra_note}"
        RUN_PATH = f"{dataset_group}/{dataset_name}/{TODAY_DATE}/{RUN_NAME}"

        # Make required directories for testing
        try:
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "A"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "B"))
            if channels == 1:
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
            if dataset_group == "l2r":
                real_image_A = data["left"].to(run.device)
                real_image_B = data["right"].to(run.device)

            elif dataset_group == "s2d":
                real_image_A_left = data["A_left"].to(run.device)
                real_image_A_right = data["A_right"].to(run.device)
                real_image_B = data["B"].to(run.device)

                real_image_A_left = real_image_A_left
                real_image_A_right = real_image_A_right
                real_image_B = real_image_B

                # Concatenate left- and right view into one stereo image
                # self.real_image_A = torch.cat((real_image_A_left, real_image_A_right), dim=-1)

                # Add the left- and right into one image and normalize again
                real_image_A = torch.add(real_image_A_left, real_image_A_right)

                if channels == 1:
                    transforms.Normalize(mean=(0.5), std=(0.5))(real_image_A)
                elif channels == 3:
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(real_image_A)

            """ (2) Generate output """

            # Generate output
            _fake_image_A = netG_B2A(real_image_B)
            _fake_image_B = netG_A2B(real_image_A)

            # Generate original image from generated (fake) output
            _fake_original_image_A = netG_B2A(_fake_image_B)
            _fake_original_image_B = netG_A2B(_fake_image_A)

            """ (1) Convert to usable images """

            # Convert to usable images, this makes sure they are in a [0, 1] range instead of [-1, 1]
            fake_image_A = 0.5 * (_fake_image_A.data + 1.0)
            fake_image_B = 0.5 * (_fake_image_B.data + 1.0)

            # Convert to usable images, this makes sure they are in a [0, 1] range instead of [-1, 1]
            fake_original_image_A = 0.5 * (_fake_original_image_A.data + 1.0)
            fake_original_image_B = 0.5 * (_fake_original_image_B.data + 1.0)

            """ (1) Calculate losses for the generated (fake) output """

            """ Calculate MSE and SSIM """

            def mse(image_A, image_B):

                # the 'Mean Squared Error' between the two images is the
                # sum of the squared difference between the two images;
                # NOTE: the two images must have the same dimension

                error = np.sum((image_A.astype("float") - image_B.astype("float")) ** 2)
                error /= float(image_A.shape[0] * image_A.shape[1])

                # return the MSE, the lower the error, the more "similar"
                # the two images are

                return error

            np_real_image_A, np_real_image_B = (
                real_image_A.squeeze().cpu().numpy(),
                real_image_B.squeeze().cpu().numpy(),
            )
            np_fake_image_A, np_fake_image_B = (
                fake_image_A.squeeze().cpu().numpy(),
                fake_image_B.squeeze().cpu().numpy(),
            )

            # Calculate the mean square error (MSE) loss
            mse_loss_A = mse(np_fake_image_A, np_real_image_B)
            mse_loss_B = mse(np_fake_image_B, np_real_image_A)

            # print(mse_loss_A, mse_loss_B)
            # print(mse_loss(fake_image_A, real_image_B), mse_loss(fake_image_B, real_image_A))

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

            # std, mean = 0.5, 0.5

            # random_noise_image_A = (torch.randn(torch.tensor(fake_image_A).size()) * std + mean).to(run.device)
            # random_noise_image_B = (torch.randn(torch.tensor(fake_image_B).size()) * std + mean).to(run.device)

            # # Calculate the mean square error (MSE) loss compared to random noise
            # mse_loss_A_noise = mse_loss(fake_image_A, random_noise_image_A)
            # mse_loss_B_noise = mse_loss(fake_image_B, random_noise_image_B)

            """ (1) Define filepaths, save generated output and print a progress bar """

            # Filepath and filename for the real and generated output images
            (
                filepath_real_A,
                filepath_real_B,
                filepath_fake_A,
                filepath_fake_B,
                filepath_f_or_A,
                filepath_f_or_B,
                # filepath_random_noise_A,
                # filepath_random_noise_B,
            ) = (
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___fake_sample_MSE_{mse_loss_A:.5f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___fake_sample_MSE_{mse_loss_B:.5f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___fake_original_MSE_{mse_loss_f_or_A:.5f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___fake_original_MSE_{mse_loss_f_or_B:.5f}.png",
                # f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___random_noise_MSE_{mse_loss_A_noise:.5f}.png",
                # f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___random_noise_MSE_{mse_loss_B_noise:.5f}.png",
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

            # # Save generated (fake) original images
            # vutils.save_image(random_noise_image_A.detach(), filepath_random_noise_A, normalize=True)
            # vutils.save_image(random_noise_image_B.detach(), filepath_random_noise_B, normalize=True)

            """ Convert to disparity maps """

            def __convert_disparty(image_np) -> np.ndarray:

                vmin = image_np.min()
                vmax = np.percentile(image_np, 100)

                # cmaps = [ 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

                # Normalize using the provided cmap
                normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
                colormapped_im = (mapper.to_rgba(image_np)[:, :, :3] * 255).astype(np.uint8)

                # Convert to PIL and return np.ndarray
                image_target = PIL.fromarray(colormapped_im)

                return np.array(image_target)

            if channels == 1:

                # Convert grayscale to disparity map
                np_real_image_B: np.ndarray = __convert_disparty(real_image_B.squeeze().cpu().numpy())
                np_fake_image_A: np.ndarray = __convert_disparty(fake_image_A.squeeze().cpu().numpy())

                mse_disparity_real = mse(np_real_image_B, np_real_image_B)

                # Filepaths
                filepath_disparity_real, filepath_disparity_fake, filepath_input_sample = (
                    f"{DIR_RESULTS}/{RUN_PATH}/D/{i + 1:04d}___disparity_real_MSE_{mse_disparity_real:.3f}.png",
                    f"{DIR_RESULTS}/{RUN_PATH}/D/{i + 1:04d}___disparity_fake_MSE_{mse_loss_A:.3f}.png",
                    f"{DIR_RESULTS}/{RUN_PATH}/D/{i + 1:04d}___input_sample.png",
                )

                # Save disparity maps
                Image.fromarray(np_real_image_B).save(filepath_disparity_real)
                Image.fromarray(np_fake_image_A).save(filepath_disparity_fake)
                vutils.save_image(real_image_A.detach(), filepath_input_sample, normalize=True)

                # Convert to disparity tensor into the range [0,1]
                disparity_image_real = 0.5 * (torch.tensor(np_real_image_B).data + 1.0)
                disparity_image_fake = 0.5 * (torch.tensor(np_fake_image_A).data + 1.0)

                # Calculate the mean square error (MSE) loss
                mse_loss_disparity = mse_loss(disparity_image_fake, disparity_image_real)

                # print(mse_loss_disparity)

                # # Transform to PIL
                # disparity_image_real = transforms.ToPILImage()(disparity_image_real)
                # disparity_image_fake = transforms.ToPILImage()(disparity_image_fake)

                # vutils.save_image(torch.tensor(disparity_image_fake).detach(), filepath_disparity_fake, normalize=True)

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
            progress_bar.set_description(
                f"Process images {i + 1} of {len(loader)}  ||  "
                f"avg_mse_loss_A: {avg_mse_loss_A:.3f} ; avg_mse_loss_B: {avg_mse_loss_B:.3f} ||  "
                f"avg_mse_loss_f_or_A: {avg_mse_loss_f_or_A:.3f} ; avg_mse_loss_f_or_B: {avg_mse_loss_f_or_B:.3f} ||  "
            )

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

    pass


# Testing function
def test_A2B(
    dataset: StereoDisparityDataset,
    parameters: OrderedDict,
    channels: int,
    #
    dataset_group: str,
    dataset_name: str,
    extra_note: str,
    #
    model_group: str,
    model_folder: str,
    model_name: str,
    model_date: str,
    #
    model_netG_A2B: str,
    model_netG_B2A: str,
) -> None:

    """ Insert documentation """

    path_to_folder = f"weights/{model_group}/{model_folder}/{model_date}/{model_name}"

    # Iterate over every run, based on the configurated params
    for run in RunCycleBuilder.get_runs(parameters):

        # Clear occupied CUDA memory
        torch.cuda.empty_cache()

        # Store today's date in string format
        TODAY_DATE = datetime.today().strftime("%Y-%m-%d")
        TODAY_TIME = datetime.today().strftime("%H.%M.%S")

        # Create a unique name for this run
        RUN_NAME = f"{TODAY_TIME}___EP{run.num_epochs}_DE{run.decay_epochs}_LR{run.learning_rate}_BS{run.batch_size}_{extra_note}"
        RUN_PATH = f"{dataset_group}/{dataset_name}/{TODAY_DATE}/{RUN_NAME}"

        # Make required directories for testing
        try:
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "A"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "B"))
            if channels == 1:
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
        avg_mse_loss_A2B, cum_mse_loss_A2B = 0, 0

        # Iterate over the data
        for i, data in progress_bar:

            """ (1) Read input data """

            # Get image A and image B
            if dataset_group == "l2r":
                real_image_A = data["left"].to(run.device)
                real_image_B = data["right"].to(run.device)

            elif dataset_group == "s2d":
                real_image_A_left = data["A_left"].to(run.device)
                real_image_A_right = data["A_right"].to(run.device)
                real_image_B = data["B"].to(run.device)

                real_image_A_left = real_image_A_left
                real_image_A_right = real_image_A_right
                real_image_B = real_image_B

                # Concatenate left- and right view into one stereo image
                # self.real_image_A = torch.cat((real_image_A_left, real_image_A_right), dim=-1)

                # Add the left- and right into one image and normalize again
                real_image_A = torch.add(real_image_A_left, real_image_A_right)

                if channels == 1:
                    transforms.Normalize(mean=(0.5), std=(0.5))(real_image_A)
                elif channels == 3:
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(real_image_A)

            """ (2) Generate output """

            # Generate output
            generated_image_A2B = netG_A2B(real_image_A)

            # vutils.save_image(generated_image_A2B.detach(), f"{DIR_RESULTS}/{RUN_PATH}/{i + 1:04d}out_a2b(a).png", normalize=True)
            # vutils.save_image(real_image_A.detach(), f"{DIR_RESULTS}/{RUN_PATH}/{i + 1:04d}in_a2b(a).png", normalize=True)

            """ (1) Convert to usable images """

            # Convert to usable images, this makes sure they are in a [0, 1] range instead of [-1, 1]
            generated_image_A2B = 0.5 * (generated_image_A2B.data + 1.0)

            """ (1) Calculate losses for the generated (fake) output """

            """ Calculate MSE and SSIM """

            def mse(image_A, image_B):

                # the 'Mean Squared Error' between the two images is the
                # sum of the squared difference between the two images;
                # NOTE: the two images must have the same dimension

                error = np.sum((image_A.astype("float") - image_B.astype("float")) ** 2)
                error /= float(image_A.shape[0] * image_A.shape[1])

                # return the MSE, the lower the error, the more "similar"
                # the two images are

                return error

            np_real_image_A = real_image_A.squeeze().cpu().numpy()
            np_real_image_B = real_image_B.squeeze().cpu().numpy()

            """ (1) Define filepaths, save generated output and print a progress bar """

            # Filepath and filename for the real and generated output images
            (filepath_real_A, filepath_real_B, filepath_fake_B,) = (
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___fake_sample.png",
            )

            # Initiate a mean square error (MSE) loss function
            mse_loss = nn.MSELoss()

            # Calculate the mean square error (MSE) loss
            # mse_loss_generated_A = mse_loss(generated_image_B2A.detach(), real_image_A)
            mse_loss_A2B = mse_loss(generated_image_A2B.detach(), real_image_B)

            # Calculate the sum of all mean square error (MSE) losses
            cum_mse_loss_A2B += mse_loss_A2B

            # Calculate the average mean square error (MSE) loss
            avg_mse_loss_A2B = cum_mse_loss_A2B / (i + 1)

            """ (1) Define filepaths, save generated output and print a progress bar """

            # Filepath and filename for the real and generated output images
            (filepath_real_A, filepath_real_B, filepath_fake_B,) = (
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___fake_sample_MSE_{mse_loss_A2B:.5f}.png",
            )

            # Save real input images
            vutils.save_image(real_image_A.detach(), filepath_real_A, normalize=True)
            vutils.save_image(real_image_B.detach(), filepath_real_B, normalize=True)

            # Save generated (fake) output images
            vutils.save_image(generated_image_A2B.detach(), filepath_fake_B, normalize=True)

            """ Convert to disparity maps """

            def __convert_disparty(image_np) -> np.ndarray:

                vmin = image_np.min()
                vmax = np.percentile(image_np, 100)

                # cmaps = [ 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

                # Normalize using the provided cmap
                normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
                colormapped_im = (mapper.to_rgba(image_np)[:, :, :3] * 255).astype(np.uint8)

                # Convert to PIL and return np.ndarray
                image_target = PIL.fromarray(colormapped_im)

                return np.array(image_target)

            if channels == 1:

                # Save generated (fake) output images
                vutils.save_image(generated_image_A2B.detach(), filepath_fake_B, normalize=True)

                # Put on cpu and convert to numpy
                np_generated_image_A2B = generated_image_A2B.squeeze().cpu().numpy()

                # Convert grayscale to disparity map
                np_real_image_B: np.ndarray = __convert_disparty(np_real_image_B)
                np_fake_image_A2B: np.ndarray = __convert_disparty(np_generated_image_A2B)

                mse_disparity_real = mse(np_real_image_B, np_real_image_B)

                # Filepaths
                filepath_disparity_real, filepath_disparity_fake, filepath_input_sample = (
                    f"{DIR_RESULTS}/{RUN_PATH}/D/{i + 1:04d}___disparity_real_MSE_{mse_disparity_real:.3f}.png",
                    f"{DIR_RESULTS}/{RUN_PATH}/D/{i + 1:04d}___disparity_fake_MSE_{mse_loss_A2B:.3f}.png",
                    f"{DIR_RESULTS}/{RUN_PATH}/D/{i + 1:04d}___input_sample.png",
                )

                # Save disparity maps
                Image.fromarray(np_real_image_B).save(filepath_disparity_real)
                Image.fromarray(np_fake_image_A2B).save(filepath_disparity_fake)
                vutils.save_image(real_image_A.detach(), filepath_input_sample, normalize=True)

                # Convert to disparity tensor into the range [0,1]
                disparity_image_real = 0.5 * (torch.tensor(np_real_image_B).data + 1.0)
                disparity_image_fake = 0.5 * (torch.tensor(np_fake_image_A2B).data + 1.0)

                # Calculate the mean square error (MSE) loss
                mse_loss_disparity = mse_loss(disparity_image_fake, disparity_image_real)

            """ Progress bar """

            # Print a progress bar in terminal
            progress_bar.set_description(
                f"Process images {i + 1} of {len(loader)}  ||  " f"avg_mse_loss_B: {avg_mse_loss_A2B:.3f} ||  "
            )

        # Calculate average mean squared error (MSE)
        avg_mse_loss_A2B = cum_mse_loss_A2B / len(loader)

        # Print
        print("avg_mse_loss_A2B:", avg_mse_loss_A2B)

    pass


# Execute main code
if __name__ == "__main__":

    # weights\l2r\Test_Set_ORIGINAL\2021-05-01\14.22.13___EP100_DE050_LR0.0002_CH3
    # weights\s2d\Test_Set_RGB_DISPARITY\2021-04-28\11.51.48___EP100_DE050_LR0.0002_CH1

    try:

        mydataloader = MyDataLoader()

        testset_foldername = "test"

        # test(
        #     parameters=PARAMETERS,
        #     dataset=mydataloader.get_dataset("l2r", "DrivingStereoDemo", "test", (176, 80), 3, False),
        #     channels=3,
        #     #
        #     dataset_group=f"l2r",
        #     dataset_name=f"DrivingStereoDemo",
        #     extra_note=testset_foldername,
        #     #
        #     model_group="l2r",
        #     model_folder="Test_Set_RGB_DISPARITY",
        #     model_date=f"2021-05-01",
        #     model_name=f"14.22.13___EP100_DE050_LR0.0002_CH3",
        #     #
        #     model_netG_A2B=f"net_G_A2B.pth",
        #     model_netG_B2A=f"net_G_B2A.pth",
        # )

        test_A2B(
            parameters=PARAMETERS,
            dataset=mydataloader.get_dataset("s2d", "Test_Set_RGB_DISPARITY", testset_foldername, (68, 120), 1, False),
            channels=1,
            #
            dataset_group=f"s2d",
            dataset_name=f"Test_Set_RGB_DISPARITY",
            extra_note=testset_foldername,
            #
            model_group="s2d",
            model_folder="Test_Set_RGB_DISPARITY",
            model_date=f"2021-05-10",
            model_name=f"15.23.54___EP100_DE050_LR0.0002_CH1",
            #
            model_netG_A2B=f"net_G_A2B_epoch_90.pth",
            model_netG_B2A=f"net_G_B2A_epoch_90.pth",
        )

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
