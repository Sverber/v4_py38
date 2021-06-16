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

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

# Clear the terminal
os.system("cls")


# Constants: required directories
DIR_DATASET = f"./dataset"
DIR_OUTPUTS = f"./outputs"
DIR_RESULTS = f"./results"
DIR_WEIGHTS = f"./weights"

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
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "B2A"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "A2B"))
            #
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "FID", "A2B"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "FID", "A2B", "real"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "FID", "A2B", "fake"))
            #
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "FID", "B2A"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "FID", "B2A", "real"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "FID", "B2A", "fake"))
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
        netG_B2A.load_state_dict(torch.load(os.path.join(str(path_to_folder), "net_G_B2A", model_netG_B2A)))
        netG_A2B.load_state_dict(torch.load(os.path.join(str(path_to_folder), "net_G_A2B", model_netG_A2B)))

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

        # Initiate FID score variables
        avg_fid_score_A, avg_fid_score_B = 0, 0
        cum_fid_score_A, cum_fid_score_B = 0, 0

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
            mse_loss_A = mse(np_fake_image_A, np_real_image_A)
            mse_loss_B = mse(np_fake_image_B, np_real_image_B)

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
            
            """ Calculate FID scores """
            
            def calculate_fid(act1, act2):
                # calculate mean and covariance statistics
                mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
                mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
                # calculate sum squared difference between means
                ssdiff = np.sum((mu1 - mu2)**2.0)
                # calculate sqrt of product between cov
                covmean = sqrtm(sigma1.dot(sigma2))
                # check and correct imaginary numbers from sqrt
                if iscomplexobj(covmean):
                    covmean = covmean.real
                # calculate score
                fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
                return fid

            # # define two collections of activations
            # act1 = random(10*2048)
            # act1 = act1.reshape((10,2048))
            # act2 = random(10*2048)
            # act2 = act2.reshape((10,2048))

            # Calculate the mean square error (MSE) loss
            fid_score_A = calculate_fid(np_fake_image_A, np_real_image_A)
            fid_score_B = calculate_fid(np_fake_image_B, np_real_image_B)

            # Calculate the cumulative FID score 
            cum_fid_score_A += fid_score_A
            cum_fid_score_B += fid_score_B

            # Calculate the average FID score 
            avg_fid_score_A = cum_fid_score_A / (i + 1)
            avg_fid_score_B = cum_fid_score_B / (i + 1)

            """ Define filepaths, save generated output and print a progress bar """

            # Filepath and filename for the real and generated output images
            (filepath_real_A, filepath_real_B, filepath_fake_A, filepath_fake_B, filepath_f_or_A, filepath_f_or_B,) = (
                f"{DIR_RESULTS}/{RUN_PATH}/B2A/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A2B/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B2A/{i + 1:04d}___fake_sample_MSE_{mse_loss_A:.5f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A2B/{i + 1:04d}___fake_sample_MSE_{mse_loss_B:.5f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B2A/{i + 1:04d}___fake_original_MSE_{mse_loss_f_or_A:.5f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A2B/{i + 1:04d}___fake_original_MSE_{mse_loss_f_or_B:.5f}.png",
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
            
            """ Save data for FID evaluation """

            # Filepath for A2B real and fake images
            filepath_FID_A2B_REAL, filepath_FID_A2B_FAKE = (
                f"{DIR_RESULTS}/{RUN_PATH}/FID/A2B/real/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/FID/B2A/fake/{i + 1:04d}___fake_sample_MSE_{mse_loss_A:.5f}.png",
            )

            # Filepath for B2A real and fake images
            filepath_FID_B2A_REAL, filepath_FID_B2A_FAKE = (
                f"{DIR_RESULTS}/{RUN_PATH}/FID/B2A/real/{i + 1:04d}___real_sample.png",
                f"{DIR_RESULTS}/{RUN_PATH}/FID/A2B/fake/{i + 1:04d}___fake_sample_MSE_{mse_loss_B:.5f}.png",
            )

            # Save real images of domain A
            vutils.save_image(real_image_A.detach(), filepath_FID_B2A_REAL, normalize=True)
            vutils.save_image(fake_image_A.detach(), filepath_FID_B2A_FAKE, normalize=True)

            # Save real images of domain B
            vutils.save_image(real_image_B.detach(), filepath_FID_A2B_REAL, normalize=True)
            vutils.save_image(fake_image_B.detach(), filepath_FID_A2B_FAKE, normalize=True)

            """ Convert to disparity maps if only 1 CHANNEL and datagroup is S2D """

            if channels == 1 and dataset_group == "s2d":

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

                # Convert grayscale to disparity map
                np_real_image_B: np.ndarray = __convert_disparty(real_image_B.squeeze().cpu().numpy())
                np_fake_image_B: np.ndarray = __convert_disparty(fake_image_B.squeeze().cpu().numpy())

                mse_disparity_real = mse(np_real_image_B, np_real_image_B)

                # Filepaths
                filepath_disparity_real, filepath_disparity_fake, filepath_input_sample = (
                    f"{DIR_RESULTS}/{RUN_PATH}/D/{i + 1:04d}___disparity_real_MSE_{mse_disparity_real:.3f}.png",
                    f"{DIR_RESULTS}/{RUN_PATH}/D/{i + 1:04d}___disparity_fake_MSE_{mse_loss_A:.3f}.png",
                    f"{DIR_RESULTS}/{RUN_PATH}/D/{i + 1:04d}___input_sample.png",
                )

                # Save disparity maps
                Image.fromarray(np_real_image_B).save(filepath_disparity_real)
                Image.fromarray(np_fake_image_B).save(filepath_disparity_fake)
                vutils.save_image(real_image_A.detach(), filepath_input_sample, normalize=True)

                # Convert to disparity tensor into the range [0,1]
                disparity_image_real = 0.5 * (torch.tensor(np_real_image_B).data + 1.0)
                disparity_image_fake = 0.5 * (torch.tensor(np_fake_image_B).data + 1.0)

                # Calculate the mean square error (MSE) loss
                mse_loss_disparity = mse_loss(disparity_image_fake, disparity_image_real)

            """ Progress bar """

            # Print a progress bar in terminal
            progress_bar.set_description(
                f"Process images {i + 1} of {len(loader)}  ||  "
                f"avg_mse_loss_A: {avg_mse_loss_A:.3f} ; avg_mse_loss_B: {avg_mse_loss_B:.3f} ||  "
                # f"avg_mse_loss_f_or_A: {avg_mse_loss_f_or_A:.3f} ; avg_mse_loss_f_or_B: {avg_mse_loss_f_or_B:.3f} ||  "
                f"avg_fid_score_A: {avg_fid_score_A:.3f} ; avg_fid_score_B: {avg_fid_score_B:.3f}   "
            )

        """ Store evaluation in a .txt file """

        # Create text file
        f = open(f"{DIR_RESULTS}/{RUN_PATH}/evaluation.txt", "w+")

        # Add line to text file
        f.write(f"MSE score fake_images_A: {avg_mse_loss_A:.3f}")
        f.write(f"MSE score fake_images_B: {avg_mse_loss_B:.3f}")
        f.write(f"FID score fake_images_A: {avg_fid_score_A:.3f}")
        f.write(f"FID score fake_images_B: {avg_fid_score_B:.3f}")

        # Close file
        f.close()

        """ Print evaluation """

        # Print
        print("MSE score fake_images_A:", avg_mse_loss_A)
        print("MSE score fake_images_B:", avg_mse_loss_B)
        print("FID score fake_images_A:", avg_fid_score_A)
        print("FID score fake_images_B:", avg_fid_score_B)

    pass


# Parameters
PARAMETERS: OrderedDict = OrderedDict(
    device=[torch.device("cuda" if torch.cuda.is_available() else "cpu")],
    shuffle=[False],
    num_workers=[8],
    manualSeed=[999],
    learning_rate=[0.0002],
    batch_size=[1],
    num_epochs=[20],
    decay_epochs=[10],
)


# Execute main code
if __name__ == "__main__":

    try:

        mydataloader = MyDataLoader()

        channels = 1
        group = "s2d"

        test(
            parameters=PARAMETERS,
            dataset=mydataloader.get_dataset(group, "Test_Set_RGB_DISPARITY", "test", (68, 120), channels, False),
            channels=channels,
            #
            dataset_group=group,
            dataset_name=f"Test_Set_RGB_DISPARITY",
            extra_note="test",
            #
            model_group=group,
            model_folder="Test_Set_RGB_DISPARITY",
            model_date=f"2021-06-16",
            model_name=f"19.09.05___EP20_DE10_LRG0.0002_CH1",
            #
            model_netG_A2B=f"net_G_A2B.pth",
            model_netG_B2A=f"net_G_B2A.pth",
        )

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
