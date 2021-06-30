#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import sys
import yaml
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
from utils.tools.yaml_reader import read_yaml
from utils.tools.pytorch_fid.src.pytorch_fid.fid_score import calculate_fid_given_paths

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm


# Clear the terminal
os.system("cls")


# Constants: required directories
DIR_DATASET = f"./dataset"
DIR_OUTPUTS = f"./outputs"
DIR_RESULTS = f"./results"
DIR_WEIGHTS = f"./weights"


# Initialize_weights
def initialize_weights(m):

    """ Custom weights initialization called on a Generator or Discriminator network """

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

    pass


# Convert disparity
def convert_disparty(image_np) -> np.ndarray:

    """ Converts a grayscale disparity map to coloured disparity map """

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


# Calculate MSE
def calculate_mse(image_A, image_B):

    """ Calclate MSE loss for a source- and target image """

    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # note: the two images must have the same dimension

    error = np.sum((image_A.astype("float") - image_B.astype("float")) ** 2)
    error /= float(image_A.shape[0] * image_A.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are

    return error


# Calculate FID
def calculate_fid(act1, act2):

    """ Calclate FID score for a source- and target image, according: 
    
        d^2 = ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2))

    """

    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID score
    FID = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)

    return FID


# Testing function
def test(
    dataset,
    parameters: OrderedDict,
    channels: int,
    #
    dataset_group: str,
    dataset_name: str,
    #
    model_group: str,
    model_folder: str,
    model_name: str,
    model_date: str,
    #
    model_netG_A2B: str,
    model_netG_B2A: str,
    #
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

        # Determine how many digits need to be used
        digits = len(str(run.num_epochs))

        # Create a unique name for this run
        RUN_NAME = f"{TODAY_TIME}___EP{str(run.num_epochs).zfill(digits)}_DE{str(run.decay_epochs).zfill(digits)}_LRG{run.learning_rate_gen}_CH{channels}"
        RUN_PATH = f"{dataset_group}/{dataset_name}/{TODAY_DATE}/{RUN_NAME}"

        # Make required directories for testing
        try:
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "B2A"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "A2B"))
            #
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "FID", "A2B", "real"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "FID", "A2B", "fake"))
            #
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "FID", "B2A", "real"))
            os.makedirs(os.path.join(DIR_RESULTS, RUN_PATH, "FID", "B2A", "fake"))

        except OSError:
            pass

        # Allow cuddn to look for the optimal set of algorithms to improve runtime speed
        cudnn.benchmark = True

        # Dataloader
        loader = DataLoader(
            dataset=dataset, batch_size=run.batch_size, num_workers=run.num_workers, shuffle=run.shuffle
        )

        # Create generator models
        netG_A2B = Generator(channels, channels).to(run.device)
        netG_B2A = Generator(channels, channels).to(run.device)

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
        avg_mse_loss_A, avg_mse_loss_B = 0, 0
        cum_mse_loss_A, cum_mse_loss_B = 0, 0

        # Initiate root mean squared error (RMSE) losses variables
        avg_rmse_loss_A, avg_rmse_loss_B = 0, 0
        cum_rmse_loss_A, cum_rmse_loss_B = 0, 0

        # Intitiate FID score variables
        fid_score_A, fid_score_B = 0, 0

        # Iterate over the data
        for i, data in progress_bar:

            """ Read input data """

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
                # real_image_A = real_image_A_left
                real_image_A = torch.cat((real_image_A_left, real_image_A_right), dim=0)

                # Normalize again
                if channels == 1:
                    transforms.Normalize(mean=(0.5), std=(0.5))(real_image_A)
                elif channels == 3:
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(real_image_A)

            """ Generate output and de-normalize """

            # Generate output & convert to usable images (this makes sure they are in a [0, 1] range instead of [-1, 1])
            fake_image_A = 0.5 * (netG_B2A(real_image_B).data + 1.0)
            fake_image_B = 0.5 * (netG_A2B(real_image_A).data + 1.0)

            """ Convert to 2D numpy arrays """

            # 4d shape: [n, c, h, w] to 2d shape: [n, (c*h*w)], by doing: np.reshape(input_4d, (n, (c*h*w)))

            # Grayscale images become a 2D array after squeezing(1, h, w) and are therefore a 2D array
            np_real_image_A, np_real_image_B = (
                real_image_A.squeeze().cpu().numpy(),
                real_image_B.squeeze().cpu().numpy(),
            )
            np_fake_image_A, np_fake_image_B = (
                fake_image_A.squeeze().cpu().numpy(),
                fake_image_B.squeeze().cpu().numpy(),
            )

            # RGB images are become 3D arrays after sqeeuzing (3, h, w) and therefore need to be reshapen into a 2D array
            if channels != 1:

                np_real_image_A = np_real_image_A.reshape(
                    (np_real_image_A.shape[1] * np_real_image_A.shape[2]), np_real_image_A.shape[0]
                )
                np_real_image_B = np_real_image_B.reshape(
                    (np_real_image_B.shape[1] * np_real_image_B.shape[2]), np_real_image_B.shape[0]
                )
                np_fake_image_A = np_fake_image_A.reshape(
                    (np_fake_image_A.shape[1] * np_fake_image_A.shape[2]), np_fake_image_A.shape[0]
                )
                np_fake_image_B = np_fake_image_B.reshape(
                    (np_fake_image_B.shape[1] * np_fake_image_B.shape[2]), np_fake_image_B.shape[0]
                )

            """ Calculate MSE losses """

            # Calculate the mean square error (MSE) loss
            mse_loss_A = calculate_mse(np_fake_image_A, np_real_image_A)
            mse_loss_B = calculate_mse(np_fake_image_B, np_real_image_B)

            # Calculate the sum of all mean square error (MSE) losses
            cum_mse_loss_A += mse_loss_A
            cum_mse_loss_B += mse_loss_B

            # Calculate the average mean square error (MSE) loss
            avg_mse_loss_A = cum_mse_loss_A / (i + 1)
            avg_mse_loss_B = cum_mse_loss_B / (i + 1)

            """ Calculate RMSE losses """

            # Calculate the mean square error (MSE) loss
            rmse_loss_A = np.sqrt(mse_loss_A)
            rmse_loss_B = np.sqrt(mse_loss_B)

            # Calculate the sum of all root mean square error (RMSE) losses
            cum_rmse_loss_A += rmse_loss_A
            cum_rmse_loss_B += rmse_loss_B

            # Calculate the average root mean square error (RMSE) loss
            avg_rmse_loss_A = cum_rmse_loss_A / (i + 1)
            avg_rmse_loss_B = cum_rmse_loss_B / (i + 1)

            """ Define filepaths and save real- and fake images in A2B and B2A, respectively """

            # Filepath and filename for the real and generated output images
            filepath_real_A, filepath_real_B, filepath_fake_A, filepath_fake_B = (
                f"{DIR_RESULTS}/{RUN_PATH}/B2A/{i + 1:04d}___real.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A2B/{i + 1:04d}___real.png",
                f"{DIR_RESULTS}/{RUN_PATH}/B2A/{i + 1:04d}___fake_RMSE_{rmse_loss_A:.3f}.png",
                f"{DIR_RESULTS}/{RUN_PATH}/A2B/{i + 1:04d}___fake_RMSE_{rmse_loss_B:.3f}.png",
            )

            # Save real input images
            vutils.save_image(real_image_A.detach(), filepath_real_A, normalize=True)
            vutils.save_image(real_image_B.detach(), filepath_real_B, normalize=True)

            # Save generated (fake) output images
            vutils.save_image(fake_image_A.detach(), filepath_fake_A, normalize=True)
            vutils.save_image(fake_image_B.detach(), filepath_fake_B, normalize=True)

            """ Define filepaths and save real- and fake images in FID/A and FID/B, respectively """

            # Filepath for A2B real and fake images
            filepath_FID_A2B_REAL, filepath_FID_A2B_FAKE = (
                f"{DIR_RESULTS}/{RUN_PATH}/FID/A2B/real/{i + 1:04d}___real.png",
                f"{DIR_RESULTS}/{RUN_PATH}/FID/B2A/fake/{i + 1:04d}___fake.png",
            )

            # Filepath for B2A real and fake images
            filepath_FID_B2A_REAL, filepath_FID_B2A_FAKE = (
                f"{DIR_RESULTS}/{RUN_PATH}/FID/B2A/real/{i + 1:04d}___real.png",
                f"{DIR_RESULTS}/{RUN_PATH}/FID/A2B/fake/{i + 1:04d}___fake.png",
            )

            # Save real images of domain A
            vutils.save_image(real_image_A.detach(), filepath_FID_B2A_REAL, normalize=True)
            vutils.save_image(fake_image_A.detach(), filepath_FID_B2A_FAKE, normalize=True)

            # Save real images of domain B
            vutils.save_image(real_image_B.detach(), filepath_FID_A2B_REAL, normalize=True)
            vutils.save_image(fake_image_B.detach(), filepath_FID_A2B_FAKE, normalize=True)

            """ Print the progress bar """

            # Print a progress bar in terminal
            progress_bar.set_description(
                f"Process images {i + 1} of {len(loader)}  ||  "
                f"MSE A: {avg_mse_loss_A:.3f} ; MSE B: {avg_mse_loss_B:.3f}  ||  RMSE A: {avg_rmse_loss_A:.3f} ; RMSE B: {avg_rmse_loss_B:.3f}  ||  "
            )

            pass

        """ Calculate FID scores """

        # Define paths
        paths_B2A = [f"{DIR_RESULTS}/{RUN_PATH}/FID/B2A/fake", f"{DIR_RESULTS}/{RUN_PATH}/FID/B2A/real"]
        paths_A2B = [f"{DIR_RESULTS}/{RUN_PATH}/FID/A2B/fake", f"{DIR_RESULTS}/{RUN_PATH}/FID/A2B/real"]

        # Calculate the FID scores
        fid_score_A = calculate_fid_given_paths(paths=paths_B2A, batch_size=run.batch_size, device=run.device)
        fid_score_B = calculate_fid_given_paths(paths=paths_A2B, batch_size=run.batch_size, device=run.device)

        """ Print evaluation """

        # Print
        print(f"Average FID score domain A: {fid_score_A:.3f}")
        print(f"Average FID score domain B: {fid_score_B:.3f} <--")
        print(f"Average MSE loss  domain A: {avg_mse_loss_A:.3f}")
        print(f"Average MSE loss  domain B: {avg_mse_loss_B:.3f}")
        print(f"Average RMSE loss domain A: {avg_rmse_loss_A:.3f}")
        print(f"Average RMSE loss domain B: {avg_rmse_loss_B:.3f} <--")

        """ Save evaluation in a .txt file """

        # Create text file
        f = open(f"{DIR_RESULTS}/{RUN_PATH}/evaluation.txt", "w+")

        # Add evalution lines to text file
        f.write(f"Domain A = [real images of domain A] compared to [fake images A, i.e. net_G_B2A(real_image_B)]\n")
        f.write(f"Domain B = [real images of domain B] compared to [fake images B, i.e. net_G_A2B(real_image_A)]\n")
        f.write(f"\n")
        f.write(f"Average FID score domain A: {fid_score_A:.3f}\n")
        f.write(f"Average FID score domain B: {fid_score_B:.3f}\n")
        f.write(f"Average MSE loss  domain A: {avg_mse_loss_A:.3f}\n")
        f.write(f"Average MSE loss  domain B: {avg_mse_loss_B:.3f}\n")
        f.write(f"Average RMSE loss domain A: {avg_rmse_loss_A:.3f}\n")
        f.write(f"Average RMSE loss domain B: {avg_rmse_loss_B:.3f}\n")

        # Close file
        f.close()

    pass


# Parameters
PARAMETERS: OrderedDict = OrderedDict(
    # System configuration and reproducebility
    device=[torch.device("cuda" if torch.cuda.is_available() else "cpu")],
    num_workers=[8],
    manual_seed=[999],
    # Dataset
    dataset_group=["s2d"],
    dataset_name=["DIML"],
    dataset_mode=["train"],
    shuffle=[True],
    # Data dimensions
    batch_size=[1],
    channels=[1],
    # Model learning
    learning_rate_dis=[0.0002],
    learning_rate_gen=[0.0002],
    num_epochs=[200],
    decay_epochs=[100],
)


# Execute main code
if __name__ == "__main__":

    try:

        mydataloader = MyDataLoader()

        CHANNELS = 3
        GROUP = "s2d"

        test(
            # Parameters, dataset to use and channels
            parameters=PARAMETERS,
            dataset=mydataloader.get_dataset(GROUP, "Test_Set_RGB_DISPARITY", "test", (68, 120), CHANNELS, False),
            channels=CHANNELS,
            # Dataset- group and name
            dataset_group=GROUP,
            dataset_name=f"Test_Set_RGB_DISPARITY",
            # Model configuration, i.e. directory
            model_group=GROUP,
            model_folder="Test_Set_RGB_DISPARITY",
            model_date=f"2021-06-29",
            model_name=f"12.05.35___EP200_DE100_LRG0.0002_CH3",
            # Generator model names
            model_netG_A2B=f"net_G_A2B_epoch_80.pth",
            model_netG_B2A=f"net_G_B2A_epoch_80.pth",
        )

        """
        
        # Datset: s2d \ DIML \ test_disparity 
        # Model: s2d \ Test_Set_RGB_DISPARITY  
     
        test(
            # Parameters, dataset to use and channels
            parameters=PARAMETERS,
            dataset=mydataloader.get_dataset(GROUP, "DIML", "test_disparity", (68, 120), CHANNELS, False),
            channels=CHANNELS,
            # Dataset- group and name
            dataset_group=GROUP,
            dataset_name=f"Test_Set_RGB_DISPARITY",
            # Model configuration, i.e. directory
            model_group=GROUP,
            model_folder="Test_Set_RGB_DISPARITY",
            model_date=f"2021-06-21",
            model_name=f"20.27.41___EP100_DE050_LRG0.0002_CH3",
            # Generator model names
            model_netG_A2B=f"net_G_A2B.pth",
            model_netG_B2A=f"net_G_B2A.pth",
        )

        """

    except KeyboardInterrupt:

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
