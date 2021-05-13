#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
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
from utils.models.cycle.Generators import Generator


# Clear the terminal
os.system("cls")


# Constants: required directories
DIR_DATASET = f"./dataset"
DIR_OUTPUTS = f"./outputs"
DIR_RESULTS = f"./results"
DIR_WEIGHTS = f"./weights"


# Parameters
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

    """ Custom weights initialization called on a Generator or Discriminator network """

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

    pass


# Testing function
def test(
    dataset,
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

    """ Test the generators models A2B and B2A on a given dataset """

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

            # Get image A and image B from a l2r dataset
            if dataset_group == "l2r":
                real_image_A = data["left"].to(run.device)
                real_image_B = data["right"].to(run.device)

            # Get image A and B from a s2d dataset
            elif dataset_group == "s2d":
                real_image_A_left = data["A_left"].to(run.device)
                real_image_A_right = data["A_right"].to(run.device)
                real_image_B = data["B"].to(run.device)

                # Add the left- and right into one image (image A)
                real_image_A = torch.add(real_image_A_left, real_image_A_right)

                # Normalize again
                if channels == 1:
                    transforms.Normalize(mean=(0.5), std=(0.5))(real_image_A)
                elif channels == 3:
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(real_image_A)

            """ Generate output """

            # Generate output
            _fake_image_A = netG_B2A(real_image_B)
            _fake_image_B = netG_A2B(real_image_A)

            # Generate original image from generated (fake) output
            _fake_original_image_A = netG_B2A(_fake_image_B)
            _fake_original_image_B = netG_A2B(_fake_image_A)

            """ Convert to usable images """

            # Convert to usable images, this makes sure they are in a [0, 1] range instead of [-1, 1]
            fake_image_A = 0.5 * (_fake_image_A.data + 1.0)
            fake_image_B = 0.5 * (_fake_image_B.data + 1.0)

            # Convert to usable images, this makes sure they are in a [0, 1] range instead of [-1, 1]
            fake_original_image_A = 0.5 * (_fake_original_image_A.data + 1.0)
            fake_original_image_B = 0.5 * (_fake_original_image_B.data + 1.0)

            """ Calculate MSE losses """

            def mse(image_A, image_B):

                # the 'Mean Squared Error' between the two images is the
                # sum of the squared difference between the two images;
                # note: the two images must have the same dimension

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

            # Calculate the sum of all mean square error (MSE) losses
            cum_mse_loss_A += mse_loss_A
            cum_mse_loss_B += mse_loss_B

            # Calculate the average mean square error (MSE) loss
            avg_mse_loss_A = cum_mse_loss_A / (i + 1)
            avg_mse_loss_B = cum_mse_loss_B / (i + 1)

            """ Calculate losses for the generated (fake) original images """

            # Calculate the mean square error (MSE) for the generated (fake) originals A and B
            mse_loss_f_or_A = mse_loss(fake_original_image_A, real_image_A)
            mse_loss_f_or_B = mse_loss(fake_original_image_B, real_image_B)

            # Calculate the average mean square error (MSE) for the fake originals A and B
            cum_mse_loss_f_or_A += mse_loss_f_or_A
            cum_mse_loss_f_or_B += mse_loss_f_or_B

            # Calculate the average mean square error (MSE) for the fake originals A and B
            avg_mse_loss_f_or_A = cum_mse_loss_f_or_A / (i + 1)
            avg_mse_loss_f_or_B = cum_mse_loss_f_or_B / (i + 1)

            """ Define filepaths, save generated output and print a progress bar """

            # Filepath and filename for the real and generated output images
            (
                filepath_real_A,
                filepath_real_B,
                filepath_fake_A,
                filepath_fake_B,
                filepath_f_or_A,
                filepath_f_or_B,
            ) = (
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___fake_sample_MSE_{mse_loss_A:.5f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___fake_sample_MSE_{mse_loss_B:.5f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}___fake_original_MSE_{mse_loss_f_or_A:.5f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}___fake_original_MSE_{mse_loss_f_or_B:.5f}.png",
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


""" [TO-DO]

    - Rewrite the variable names so that they correspond (again) with the ones I use in train.py
    - Clean up the code and remove unnecessary code
    
""" 


# Execute main code
if __name__ == "__main__":

    try:

        mydataloader = MyDataLoader()

        test(
            parameters=PARAMETERS,
            dataset=mydataloader.get_dataset("s2d", "Test_Set_RGB_DISPARITY", "test", (68, 120), 1, False),
            channels=1,
            #
            dataset_group=f"s2d",
            dataset_name=f"Test_Set_RGB_DISPARITY",
            extra_note="test",
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
