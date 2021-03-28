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


# Clear terminal
os.system("cls")


# Constants: required directories
DIR_DATASET = f"./dataset"
DIR_OUTPUTS = f"./outputs"
DIR_RESULTS = f"./results"
DIR_WEIGHTS = f"./weights"


# Constants: dataset name
# NAME_DATASET = f"horse2zebra_000_999"
# NAME_DATASET = f"kitti_synthesized_000_999"
# NAME_DATASET = f"stereo_test"
# NAME_DATASET = f"DrivingStereo_demo_images"
# NAME_DATASET = f"Test_Set"                    # 2208 x 1242 resize by a factor 0.15 to (147, 83)
NAME_DATASET = f"QuickDevelop"  # 2208 x 1242 resize by a factor 0.15 to (147, 83)

# Constants: system
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants: parameters
# IMAGE_SIZE = (122, 35) # kitti_synthesized_000_999
# IMAGE_SIZE = (176, 79)  # DrivingStereo_demo_images
IMAGE_SIZE = (147, 83)  # Test_Set
RATIO_CROP = 0.82
RANDM_CROP = (int(IMAGE_SIZE[0] * RATIO_CROP), int(IMAGE_SIZE[1] * RATIO_CROP))
SHOW_IMG_FREQ = 10
SAVE_IMG_FREQ = 25

# Configure network parameters
PARAMETERS: OrderedDict = OrderedDict(
    device=[DEVICE],
    shuffle=[True],
    num_workers=[4],
    learning_rate=[0.0002],
    batch_size=[1],
    num_epochs=[5],
    decay_epochs=[2],
)


# Transformations on the datasets containing RGB stereo images
TRANSFORMATIONS_RGB: transforms = transforms.Compose(
    [
        transforms.Resize(size=IMAGE_SIZE, interpolation=Image.BICUBIC),
        transforms.RandomCrop(size=RANDM_CROP),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

# Transformations on the datasets containing grayscaled disparity maps
TRANSFORMATIONS_GRAY: transforms = transforms.Compose(
    [
        transforms.Resize(size=IMAGE_SIZE, interpolation=Image.BICUBIC),
        transforms.RandomCrop(size=RANDM_CROP),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ]
)


# Load dataset
def load_dataset(dir: str, name: str, mode: str, verbose: bool = True) -> StereoDisparityDataset:

    # Print start message if verbose is set to True
    verbose is True if print(f"Gathering the '{mode}' dataset of '{name}'") else None

    # Gather dataset
    dataset: StereoDisparityDataset = StereoDisparityDataset(
        root=f"./{dir}/{name}", mode=mode, transform=TRANSFORMATIONS_RGB
    )

    # Print completion message if verbose is set to True
    verbose is True if print(f"Loaded the '{mode}' dataset of '{name}' of length: {len(dataset)}") else None
    return dataset


# Training function
def train(dataset: StereoDisparityDataset):

    """ Insert documentation """

    print(f"Start training")

    # Iterate over every run, based on the configurated params
    for run in RunCycleBuilder.get_runs(PARAMETERS):

        # Clear occupied CUDA memory
        torch.cuda.empty_cache()

        # Store today's date in string format
        TODAY_DATE = datetime.today().strftime("%Y-%m-%d")
        TODAY_TIME = datetime.today().strftime("%H.%M.%S")

        # Create a unique name for this run
        RUN_NAME = f"{TODAY_TIME}___EP{run.num_epochs}_DE{run.decay_epochs}_LR{run.learning_rate}_BS{run.batch_size}"
        RUN_PATH = f"{NAME_DATASET}/{TODAY_DATE}/{RUN_NAME}"

        """ Insert a save params function as .txt file / incorporate in RunCycleManager """

        # Make required directories for storing the training output
        try:
            os.makedirs(os.path.join(DIR_OUTPUTS, RUN_PATH, "A"))
            os.makedirs(os.path.join(DIR_OUTPUTS, RUN_PATH, "B"))
            os.makedirs(os.path.join(DIR_OUTPUTS, RUN_PATH, "A", "epochs"))
            os.makedirs(os.path.join(DIR_OUTPUTS, RUN_PATH, "B", "epochs"))
        except OSError:
            pass

        # Create Generator and Discriminator models
        netG_A2B = Generator(in_channels=3, out_channels=3).to(run.device)
        netG_B2A = Generator(in_channels=3, out_channels=3).to(run.device)
        netD_A = Discriminator(in_channels=3, out_channels=3).to(run.device)
        netD_B = Discriminator(in_channels=3, out_channels=3).to(run.device)

        # Apply weights
        netG_A2B.apply(initialize_weights)
        netG_B2A.apply(initialize_weights)
        netD_A.apply(initialize_weights)
        netD_B.apply(initialize_weights)

        # define loss function (adversarial_loss)
        cycle_loss = torch.nn.L1Loss().to(run.device)
        identity_loss = torch.nn.L1Loss().to(run.device)
        adversarial_loss = torch.nn.MSELoss().to(run.device)

        # Optimizers
        optimizer_G_A2B = torch.optim.Adam(netG_A2B.parameters(), lr=run.learning_rate, betas=(0.5, 0.999))
        optimizer_G_B2A = torch.optim.Adam(netG_B2A.parameters(), lr=run.learning_rate, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=run.learning_rate, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=run.learning_rate, betas=(0.5, 0.999))

        # Learning rates
        lr_lambda = DecayLR(run.num_epochs, 0, run.decay_epochs).step
        lr_scheduler_G_A2B = torch.optim.lr_scheduler.LambdaLR(optimizer_G_A2B, lr_lambda=lr_lambda)
        lr_scheduler_G_B2A = torch.optim.lr_scheduler.LambdaLR(optimizer_G_B2A, lr_lambda=lr_lambda)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)

        # Buffers
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        # Dataloader
        loader = DataLoader(
            dataset=dataset, batch_size=run.batch_size, num_workers=run.num_workers, shuffle=run.shuffle
        )

        # Instantiate an instance of the RunManager class
        manager = RunCycleManager()

        # Track the start of the run
        manager.begin_run(run, loader, netG_A2B, netG_B2A, netD_A, netD_B)

        # Iterate through all the epochs
        for epoch in range(0, run.num_epochs):

            # Track the start of the epoch
            manager.begin_epoch()

            # Initiate mean squared error (MSE) losses variables
            avg_mse_loss_A, avg_mse_loss_B, avg_mse_loss_f_or_A, avg_mse_loss_f_or_B = 0, 0, 0, 0
            cum_mse_loss_A, cum_mse_loss_B, cum_mse_loss_f_or_A, cum_mse_loss_f_or_B = 0, 0, 0, 0

            # Create progress bar
            progress_bar = tqdm(enumerate(loader), total=len(loader))

            # Iterate over the data loader
            for i, data in progress_bar:

                # try:

                # Get image A and image B
                real_image_A_left = data["A_left"].to(run.device)
                real_image_A_right = data["A_right"].to(run.device)
                real_image_B = data["B"].to(run.device)

                # Concatenate left- and right view into one stereo image
                real_image_A = torch.cat((real_image_A_left, real_image_A_right), dim=-1)
                real_image_B = real_image_B

                # Real data label is 1, fake data label is 0.
                real_label = torch.full((run.batch_size, 1), 1, device=run.device, dtype=torch.float32)
                fake_label = torch.full((run.batch_size, 1), 0, device=run.device, dtype=torch.float32)

                """ (1) Update Generator networks: A2B and B2A """

                # Zero the gradients
                optimizer_G_A2B.zero_grad()
                optimizer_G_B2A.zero_grad()

                # Identity loss
                # G_B2A(A) should equal A if real A is fed
                identity_image_A = netG_B2A(real_image_A)
                loss_identity_A = identity_loss(identity_image_A, real_image_A) * 5.0

                # G_A2B(B) should equal B if real B is fed
                identity_image_B = netG_A2B(real_image_B)
                loss_identity_B = identity_loss(identity_image_B, real_image_B) * 5.0

                # GAN loss: D_A(G_A(A))
                fake_image_A = netG_B2A(real_image_B)
                fake_output_A = netD_A(fake_image_A)
                loss_GAN_B2A = adversarial_loss(fake_output_A, real_label)
                # print("loss_GAN_B2A:", loss_GAN_B2A)

                # GAN loss: D_B(G_B(B))
                fake_image_B = netG_A2B(real_image_A)
                fake_output_B = netD_B(fake_image_B)
                loss_GAN_A2B = adversarial_loss(fake_output_B, real_label)
                # print("loss_GAN_A2B:", loss_GAN_A2B)

                # Cycle loss
                recovered_image_A = netG_B2A(fake_image_B)
                loss_cycle_ABA = cycle_loss(recovered_image_A, real_image_A) * 10.0
                # print("loss_cycle_ABA:", loss_cycle_ABA)

                recovered_image_B = netG_A2B(fake_image_A)
                loss_cycle_BAB = cycle_loss(recovered_image_B, real_image_B) * 10.0
                # print("loss_cycle_BAB:", loss_cycle_BAB)

                # Combined loss and calculate gradients
                error_G = (
                    loss_identity_A
                    + loss_identity_B
                    + loss_GAN_A2B
                    + loss_GAN_B2A
                    + loss_cycle_ABA
                    + loss_cycle_BAB
                )

                # Calculate gradients for G_A and G_B
                error_G.backward()
                # print("error_G:", error_G)

                # Update the Generator networks
                optimizer_G_A2B.step()
                optimizer_G_B2A.step()

                """ (2) Update Discriminator network: A """

                # Set D_A gradients to zero
                optimizer_D_A.zero_grad()

                # Real A image loss
                real_output_A = netD_A(real_image_A)
                error_D_real_A = adversarial_loss(real_output_A, real_label)

                # Fake A image loss
                fake_image_A = fake_A_buffer.push_and_pop(fake_image_A)
                fake_output_A = netD_A(fake_image_A.detach())
                error_D_fake_A = adversarial_loss(fake_output_A, fake_label)

                # Combined loss and calculate gradients
                error_D_A = (error_D_real_A + error_D_fake_A) / 2

                # Calculate gradients for D_A
                error_D_A.backward()
                # Update D_A weights
                optimizer_D_A.step()

                """ (3) Update Discriminator network: B """

                # Set D_B gradients to zero
                optimizer_D_B.zero_grad()

                # Real B image loss
                real_output_B = netD_B(real_image_B)
                error_D_real_B = adversarial_loss(real_output_B, real_label)

                # Fake B image loss
                fake_image_B = fake_B_buffer.push_and_pop(fake_image_B)
                fake_output_B = netD_B(fake_image_B.detach())
                error_D_fake_B = adversarial_loss(fake_output_B, fake_label)

                # Combined loss and calculate gradients
                error_D_B = (error_D_real_B + error_D_fake_B) / 2

                # Calculate gradients for D_B
                error_D_B.backward()

                # Update D_B weights
                optimizer_D_B.step()

                """ (4) Network losses and input data regeneration """

                # Initiate a mean square error (MSE) loss function
                mse_loss = nn.MSELoss()

                """ Convert the tensors to usable image arrays """

                # Generate output
                _fake_image_A = netG_B2A(real_image_B)
                _fake_image_B = netG_A2B(real_image_A)

                # Generate original image from generated (fake) output
                _fake_original_image_A = netG_B2A(_fake_image_B)
                _fake_original_image_B = netG_A2B(_fake_image_A)

                # Convert to usable images
                fake_image_A = 0.5 * (_fake_image_A.data + 1.0)
                fake_image_B = 0.5 * (_fake_image_B.data + 1.0)

                # Convert to usable images
                fake_original_image_A = 0.5 * (_fake_original_image_A.data + 1.0)
                fake_original_image_B = 0.5 * (_fake_original_image_B.data + 1.0)

                """ Calculate losses for the generated (fake) output """

                # Calculate the mean square error (MSE) loss
                if i % 10 == 0: 
                    print("--- F_I(train):", fake_image_A.shape, fake_image_B.shape, real_image_A.shape, real_image_B.shape) 
              
                mse_loss_A = mse_loss(fake_image_A, real_image_B)
                mse_loss_B = mse_loss(fake_image_B, real_image_A)

                # Calculate the sum of all mean square error (MSE) losses
                cum_mse_loss_A += mse_loss_A
                cum_mse_loss_B += mse_loss_B

                # Calculate the average mean square error (MSE) loss
                avg_mse_loss_A = cum_mse_loss_A / (i + 1)
                avg_mse_loss_B = cum_mse_loss_B / (i + 1)

                """ Calculate losses for the generated (fake) original images """

                # Calculate the mean square error (MSE) for the generated (fake) originals A and B
                if i % 10 == 0: 
                    print("--- F_O(train):", fake_original_image_A.shape, fake_original_image_B.shape, real_image_A.shape, real_image_B.shape)
                mse_loss_f_or_A = mse_loss(fake_original_image_A, real_image_A)
                mse_loss_f_or_B = mse_loss(fake_original_image_B, real_image_B)

                # Calculate the average mean square error (MSE) for the fake originals A and B
                cum_mse_loss_f_or_A += mse_loss_f_or_A
                cum_mse_loss_f_or_B += mse_loss_f_or_B

                # Calculate the average mean square error (MSE) for the fake originals A and B
                avg_mse_loss_f_or_A = cum_mse_loss_f_or_A / (i + 1)
                avg_mse_loss_f_or_B = cum_mse_loss_f_or_B / (i + 1)

                """ (5) Save all generated network output and """

                def __save_realtime_output(_output_path=f"{DIR_OUTPUTS}/{RUN_PATH}") -> None:

                    # Filepath and filename for the real-time output images
                    (
                        filepath_real_A,
                        filepath_real_B,
                        filepath_fake_A,
                        filepath_fake_B,
                        filepath_f_or_A,
                        filepath_f_or_B,
                    ) = (
                        f"{_output_path}/A/real_sample.png",
                        f"{_output_path}/B/real_sample.png",
                        f"{_output_path}/A/fake_sample.png",
                        f"{_output_path}/B/fake_sample.png",
                        f"{_output_path}/A/fake_original.png",
                        f"{_output_path}/B/fake_original.png",
                    )
                    # Save real input images
                    vutils.save_image(real_image_A, filepath_real_A, normalize=True)
                    vutils.save_image(real_image_B, filepath_real_B, normalize=True)

                    # Save the generated (fake) image
                    vutils.save_image(fake_image_A.detach(), filepath_fake_A, normalize=True)
                    vutils.save_image(fake_image_B.detach(), filepath_fake_B, normalize=True)

                    # Save the generated (fake) original images
                    vutils.save_image(fake_original_image_A.detach(), filepath_f_or_A, normalize=True)
                    vutils.save_image(fake_original_image_B.detach(), filepath_f_or_B, normalize=True)

                    pass

                def __save_end_epoch_output(_output_path=f"{DIR_OUTPUTS}/{RUN_PATH}"):

                    # Filepath and filename for the per-epoch output images
                    (
                        filepath_real_A,
                        filepath_real_B,
                        filepath_fake_A,
                        filepath_fake_B,
                        filepath_f_or_A,
                        filepath_f_or_B,
                    ) = (
                        f"{_output_path}/A/epochs/EP{epoch}___real_sample.png",
                        f"{_output_path}/B/epochs/EP{epoch}___real_sample.png",
                        f"{_output_path}/A/epochs/EP{epoch}___fake_sample_MSE{mse_loss_A:.3f}.png",
                        f"{_output_path}/B/epochs/EP{epoch}___fake_sample_MSE{mse_loss_B:.3f}.png",
                        f"{_output_path}/A/epochs/EP{epoch}___fake_original_MSE{avg_mse_loss_A:.3f}.png",
                        f"{_output_path}/B/epochs/EP{epoch}___fake_original_MSE{avg_mse_loss_B:.3f}.png",
                    )

                    # Save real input images
                    vutils.save_image(real_image_A, filepath_real_A, normalize=True)
                    vutils.save_image(real_image_B, filepath_real_B, normalize=True)

                    # Save the generated (fake) image
                    vutils.save_image(fake_image_A.detach(), filepath_fake_A, normalize=True)
                    vutils.save_image(fake_image_B.detach(), filepath_fake_B, normalize=True)

                    # Save the generated (fake) original images
                    vutils.save_image(fake_original_image_A.detach(), filepath_f_or_A, normalize=True)
                    vutils.save_image(fake_original_image_B.detach(), filepath_f_or_B, normalize=True)

                    pass

                def __print_progress() -> None:

                    """ 
                        
                        # L_D(A)                = Loss discriminator A
                        # L_D(B)                = Loss discriminator B
                        # L_G(A2B)              = Loss generator A2B
                        # L_G(B2A)              = Loss generator B2A
                        # L_G_ID                = Combined oentity loss generators A2B + B2A
                        # L_G_GAN               = Combined GAN loss generators A2B + B2A
                        # L_G_CYCLE             = Combined cycle consistency loss Generators A2B + B2A
                        # G(A2B)_MSE(avg)       = Mean square error (MSE) loss over the generated output B vs. the real image A
                        # G(B2A)_MSE(avg)       = Mean square error (MSE) loss over the generated output A vs. the real image B
                        # G(A2B2A)_MSE(avg)     = Average mean square error (MSE) loss over the generated original image A vs. the real image A
                        # G(B2A2B)_MSE(avg)     = Average mean square error (MSE) loss over the generated original image B vs. the real image B

                    """

                    progress_bar.set_description(
                        f"[{epoch + 1}/{run.num_epochs}][{i + 1}/{len(loader)}] "
                        # f"L_D(A): {error_D_A.item():.2f} "
                        # f"L_D(B): {error_D_B.item():.2f} | "
                        f"L_D(A+B): {((error_D_A + error_D_B) / 2).item():.2f} | "
                        f"L_G(A2B): {loss_GAN_A2B.item():.2f} "
                        f"L_G(B2A): {loss_GAN_B2A.item():.2f} | "
                        # f"L_G_ID: {(loss_identity_A + loss_identity_B).item():.2f} "
                        # f"L_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.2f} "
                        # f"L_G_CYCLE: {(loss_cycle_ABA + loss_cycle_BAB).item():.2f} "
                        f"G(A2B)_MSE(avg): {(avg_mse_loss_A).item():.2f} "
                        f"G(B2A)_MSE(avg): {(avg_mse_loss_B).item():.2f} | "
                        f"G(A2B2A)_MSE(avg): {(avg_mse_loss_f_or_A).item():.2f} "
                        f"G(B2A2B)_MSE(avg): {(avg_mse_loss_f_or_B).item():.2f} "
                    )

                    pass

                # Save the real-time output images for every {SHOW_IMG_FREQ} images
                if i % SHOW_IMG_FREQ == 0:
                    __save_realtime_output(_output_path=f"{DIR_OUTPUTS}/{RUN_PATH}")

                # Save the network weights at the end of the epoch
                if i + 1 == len(loader) and epoch % SAVE_IMG_FREQ == 0:
                    __save_end_epoch_output(_output_path=f"{DIR_OUTPUTS}/{RUN_PATH}")

                # Print a progress bar in the terminal
                __print_progress()

                # except Exception as e:
                #     print(e)
                #     pass

            """ </for> for i, data in progress_bar: """

            # Make required directories for storing the model weights
            try:

                os.makedirs(os.path.join(DIR_WEIGHTS, RUN_PATH))
            except OSError:
                pass

            # Check points, save weights after each epoch
            torch.save(netG_A2B.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netG_A2B_epoch_{epoch}.pth")
            torch.save(netG_B2A.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netG_B2A_epoch_{epoch}.pth")
            torch.save(netD_A.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netD_A_epoch_{epoch}.pth")
            torch.save(netD_B.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netD_B_epoch_{epoch}.pth")

            # Update learning rates after each epoch
            lr_scheduler_G_A2B.step()
            lr_scheduler_G_B2A.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

            # Track the end of the epoch
            manager.end_epoch(netG_A2B, netG_B2A, netD_A, netD_B)

        """ </end> for epoch in range(0, run.num_epochs): """

        # Save last check points, after every run
        torch.save(netG_A2B.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netG_A2B.pth")
        torch.save(netG_B2A.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netG_B2A.pth")
        torch.save(netD_A.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netD_A.pth")
        torch.save(netD_B.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netD_B.pth")

        # Track the end of the run
        manager.end_run()

        # </end> for run in RunCycleBuilder.get_runs(params):

    # </end> def train():


# Testing function
def test(dataset: StereoDisparityDataset, path_to_folder: str, model_netG_A2B: str, model_netG_B2A: str) -> None:

    """ Insert documentation """

    # Iterate over every run, based on the configurated params
    for run in RunCycleBuilder.get_runs(PARAMETERS):

        # Store today's date in string format
        TODAY_DATE = datetime.today().strftime("%Y-%m-%d")
        TODAY_TIME = datetime.today().strftime("%H.%M.%S")

        # Create a unique name for this run
        RUN_NAME = f"{TODAY_TIME}___EP{run.num_epochs}_DE{run.decay_epochs}_LR{run.learning_rate}_BS{run.batch_size}"
        RUN_PATH = f"{NAME_DATASET}/{TODAY_DATE}/{RUN_NAME}"

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
            if i % 10 == 0: 
                print("--- F_I(test):", fake_image_A.shape, fake_image_B.shape, real_image_A.shape, real_image_B.shape)
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
            if i % 10 == 0: 
                print("--- F_O(test):", fake_image_A.shape, fake_image_B.shape, real_image_A.shape, real_image_B.shape)
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
            if i % SHOW_IMG_FREQ == 0:
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
        print("MSE(avg) A:", avg_mse_loss_A)
        print("MSE(avg) B:", avg_mse_loss_B)
        print("MSE(avg) A:", avg_mse_loss_f_or_A)
        print("MSE(avg) B:", avg_mse_loss_f_or_B)
        print("- Write a performance summary function & include more loss functions to test network performance. ")
        print("- Calculate MSE loss for the generated originals with correct out_channels in Generators.")
        print("- Start to organize train() into a TrainManager class or something, also for test()")
        print("- Seperate function for loss calculation.")

    # </end> def test():
    pass


# Execute main code
if __name__ == "__main__":

    try:

        # syn = Synthesis(mode="test")
        # syn.predict_depth()


        """ Train a neural network """
        dataset_train: StereoDisparityDataset = load_dataset(dir=DIR_DATASET, name=NAME_DATASET, mode="train", verbose=True)
        train(dataset=dataset_train)

        """ Test a neural network """
        dataset_test: StereoDisparityDataset = load_dataset(
            dir=DIR_DATASET,
            name=NAME_DATASET, mode="test", verbose=True
        )

        __DATASET, __DATE, __MODEL_NAME = (
            f"Test_Set",
            f"2021-03-26",
            f"16.38.24___EP100_DE50_LR0.0002_BS1",
        )
        test(
            dataset=dataset_test,
            path_to_folder=f"{DIR_WEIGHTS}/{__DATASET}/{__DATE}/{__MODEL_NAME}",
            model_netG_A2B=f"netG_A2B.pth",
            model_netG_B2A=f"netG_B2A.pth",
        )

        pass

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
