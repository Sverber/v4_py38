#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from multiprocessing import Process, freeze_support

import os
import sys
import glob
import time
import random
import argparse
import itertools
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset

from utils.functions.weight_init import weights_init

from utils.classes.DecayLR import DecayLR
from utils.classes.ImageDataset import ImageDataset
from utils.classes.ReplayBuffer import ReplayBuffer
from utils.classes.RunCycleBuilder import RunCycleBuilder
from utils.classes.RunCycleManager import RunCycleManager
from utils.classes.ImageDataset import ImageDataset

from utils.models.Discriminator import Discriminator
from utils.models.Generator import Generator


# Clear terminal
os.system("cls")


# Constants: directories
DIR_DATASET = f"./dataset"
DIR_OUTPUTS = f"./outputs"
DIR_WEIGHTS = f"./weights"
DIR_RESULTS = f"./results"


# Constants: dataset and run name
NAME_DATASET = f"horse2zebra_000_999"
NAME_RUN = f"{int(time.time())}_{NAME_DATASET}"


# Constants: parameters
IMAGE_SIZE = 128
RANDM_CROP = 100
PRINT_FREQ = 1


# Set the system- and training parameters
parameters = OrderedDict(
    device=[torch.device("cuda" if torch.cuda.is_available() else "cpu")],
    shuffle=[True],
    num_workers=[4],
    learning_rate=[0.05],
    batch_size=[1],
    num_epochs=[100],
    decay_epochs=[50],
)


# Transformations on dataset
dataset_transforms = transforms.Compose(
    [
        transforms.Resize(int(IMAGE_SIZE), Image.BICUBIC),
        transforms.RandomCrop(int(RANDM_CROP)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# Dataset for training
dataset_train = ImageDataset(
    root=f"./{DIR_DATASET}/{NAME_DATASET}", mode="train", unaligned=True, transform=dataset_transforms
)


# Dataset for testing
dataset_test = ImageDataset(
    root=f"./{DIR_DATASET}/{NAME_DATASET}", mode="test", unaligned=True, transform=dataset_transforms
)


# Training function
def train() -> None:

    # Make directories for training
    try:
        os.makedirs(os.path.join(DIR_OUTPUTS, NAME_RUN, "A"))
        os.makedirs(os.path.join(DIR_OUTPUTS, NAME_RUN, "B"))
        os.makedirs(os.path.join(DIR_WEIGHTS, NAME_RUN))
    except OSError:
        pass

    # Iterate over every run, based on the configurated parameters
    for run in RunCycleBuilder.get_runs(parameters):

        # Create Generator and Discriminator models
        netG_A2B = Generator().to(run.device)
        netG_B2A = Generator().to(run.device)
        netD_A = Discriminator().to(run.device)
        netD_B = Discriminator().to(run.device)

        # Apply weights
        netG_A2B.apply(weights_init)
        netG_B2A.apply(weights_init)
        netD_A.apply(weights_init)
        netD_B.apply(weights_init)

        # define loss function (adversarial_loss) and optimizer
        cycle_loss = torch.nn.L1Loss().to(run.device)
        identity_loss = torch.nn.L1Loss().to(run.device)
        adversarial_loss = torch.nn.MSELoss().to(run.device)

        # Optimizers
        optimizer_G_A2B = torch.optim.Adam(netG_A2B.parameters(), lr=run.learning_rate, betas=(0.5, 0.999),)
        optimizer_G_B2A = torch.optim.Adam(netG_B2A.parameters(), lr=run.learning_rate, betas=(0.5, 0.999),)
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
        loader = DataLoader(dataset_train, batch_size=run.batch_size, num_workers=run.num_workers, shuffle=run.shuffle)

        # Instantiate an instance of the RunManager class
        manager = RunCycleManager()

        for epoch in range(0, run.num_epochs):

            # Create progress bar
            progress_bar = tqdm(enumerate(loader), total=len(loader))

            # Iterate over the data loader
            for i, data in progress_bar:

                # Get image A and image B
                real_image_A = data["A"].to(run.device)
                real_image_B = data["B"].to(run.device)

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
                    loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
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

                """ (4) Track progress and save data """

                """ 
                    # L_D(B) = Loss discriminator B
                    # L_G(A2B) = Loss generator A2B
                    # L_G(B2A) = Loss generator B2A
                    # L_G_ID = Combined oentity loss generators A2B + B2A
                    # L_G_GAN = Combined GAN loss generators A2B + B2A
                    # L_G_CYCLE = Combined cycle consistency loss Generators A2B + B2A

                """

                # Print a progress bar in terminal
                progress_bar.set_description(
                    # L_D(A) = Loss discriminator A
                    f"[{epoch + 1}/{run.num_epochs}][{i + 1}/{len(loader)}] "
                    f"L_D(A): {error_D_A.item():.3f} "
                    f"L_D(B): {error_D_B.item():.3f} | "
                    f"L_G(A2B): {loss_GAN_A2B.item():.3f} "
                    f"L_G(B2A): {loss_GAN_B2A.item():.3f} | "
                    #
                    f"L_G_ID: {(loss_identity_A + loss_identity_B).item():.3f} "
                    f"L_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.3f} "
                    f"L_G_CYCLE: {(loss_cycle_ABA + loss_cycle_BAB).item():.3f} "
                )

                # Save the real/fake images
                if i % PRINT_FREQ == 0:

                    vutils.save_image(
                        real_image_A, f"{DIR_OUTPUTS}/{NAME_RUN}/A/_real_samples.png", normalize=True,
                    )
                    vutils.save_image(
                        real_image_B, f"{DIR_OUTPUTS}/{NAME_RUN}/B/_real_samples.png", normalize=True,
                    )

                    fake_image_A = 0.5 * (netG_B2A(real_image_B).data + 1.0)
                    fake_image_B = 0.5 * (netG_A2B(real_image_A).data + 1.0)

                    # Save the real-time fake images A & B to a .png
                    vutils.save_image(
                        fake_image_A.detach(), f"{DIR_OUTPUTS}/{NAME_RUN}/A/_fake_samples.png", normalize=True,
                    )
                    vutils.save_image(
                        fake_image_B.detach(), f"{DIR_OUTPUTS}/{NAME_RUN}/B/_fake_samples.png", normalize=True,
                    )

                    # Save the per-epoch fake images A & B to a .png
                    vutils.save_image(
                        fake_image_A.detach(),
                        f"{DIR_OUTPUTS}/{NAME_RUN}/A/fake_samples_epoch_{epoch}.png",
                        normalize=True,
                    )
                    vutils.save_image(
                        fake_image_B.detach(),
                        f"{DIR_OUTPUTS}/{NAME_RUN}/B/fake_samples_epoch_{epoch}.png",
                        normalize=True,
                    )

                    """
                    
                    # # Save the real-time fake image tensor to a .csv
                    # np_fake_image_A = fake_image_A.cpu().numpy()
                    # np_fake_image_B = fake_image_B.cpu().numpy()

                    # np_flatten_fake_image_A = np_fake_image_A.reshape(r)
                    # np_flatten_fake_image_B = np_fake_image_B.reshape(r)

                    # np.save(f"{DIR_OUTPUTS}/{NAME_RUN}/A/_fake_samples.csv", np_flatten_fake_image_A)
                    # np.save(f"{DIR_OUTPUTS}/{NAME_RUN}/B/_fake_samples.csv", np_flatten_fake_image_B)

                    """

                    pass

                # </end> for i, data in enumerate(loader):
                pass

            # Check points, save weights after each epoch
            torch.save(netG_A2B.state_dict(), f"{DIR_WEIGHTS}/{NAME_RUN}/netG_A2B_epoch_{epoch}.pth")
            torch.save(netG_B2A.state_dict(), f"{DIR_WEIGHTS}/{NAME_RUN}/netG_B2A_epoch_{epoch}.pth")
            torch.save(netD_A.state_dict(), f"{DIR_WEIGHTS}/{NAME_RUN}/netD_A_epoch_{epoch}.pth")
            torch.save(netD_B.state_dict(), f"{DIR_WEIGHTS}/{NAME_RUN}/netD_B_epoch_{epoch}.pth")

            # Update learning rates after each epoch
            lr_scheduler_G_A2B.step()
            lr_scheduler_G_B2A.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

            # </end> for epoch in range(0, run.num_epochs):
            pass

        # Save last check points, after every run
        torch.save(netG_A2B.state_dict(), f"{DIR_WEIGHTS}/{NAME_RUN}/netG_A2B.pth")
        torch.save(netG_B2A.state_dict(), f"{DIR_WEIGHTS}/{NAME_RUN}/netG_B2A.pth")
        torch.save(netD_A.state_dict(), f"{DIR_WEIGHTS}/{NAME_RUN}/netD_A.pth")
        torch.save(netD_B.state_dict(), f"{DIR_WEIGHTS}/{NAME_RUN}/netD_B.pth")

        # </end> for run in RunCycleBuilder.get_runs(parameters):
        pass

    # </end> def train():
    pass


# Testing function
def test(model_netG_A2B: str, model_netG_B2A: str) -> None:

    # Make directories for testing
    try:
        os.makedirs(os.path.join(DIR_RESULTS, NAME_RUN, "A"))
        os.makedirs(os.path.join(DIR_RESULTS, NAME_RUN, "B"))
    except:
        pass

    # Iterate over every run, based on the configurated parameters
    for run in RunCycleBuilder.get_runs(parameters):

        # Allow cuddn to look for the optimal set of algorithms to improve runtime speed
        cudnn.benchmark = True

        # Dataloader
        loader = DataLoader(dataset_test, batch_size=run.batch_size, num_workers=run.num_workers, shuffle=run.shuffle)

        # create model
        netG_A2B = Generator().to(run.device)
        netG_B2A = Generator().to(run.device)

        # Load state dicts
        netG_A2B.load_state_dict(torch.load(os.path.join(DIR_WEIGHTS, str(NAME_DATASET), model_netG_A2B)))
        netG_B2A.load_state_dict(torch.load(os.path.join(DIR_WEIGHTS, str(NAME_DATASET), model_netG_B2A)))

        # Set model mode
        netG_A2B.eval()
        netG_B2A.eval()

        # Create progress bar
        progress_bar = tqdm(enumerate(loader), total=len(loader))

        # Iterate over the data
        for i, data in progress_bar:
            # get batch size data
            real_images_A = data["A"].to(run.device)
            real_images_B = data["B"].to(run.device)

            # Generate output
            fake_image_A = 0.5 * (netG_B2A(real_images_B).data + 1.0)
            fake_image_B = 0.5 * (netG_A2B(real_images_A).data + 1.0)

            # Save image files
            vutils.save_image(fake_image_A.detach(), f"{DIR_RESULTS}/{NAME_RUN}/A/{i + 1:04d}.png", normalize=True)
            vutils.save_image(fake_image_B.detach(), f"{DIR_RESULTS}/{NAME_RUN}/B/{i + 1:04d}.png", normalize=True)

            # Print a progress bar in terminal
            progress_bar.set_description(f"Process images {i + 1} of {len(loader)}")


# Execute main code
if __name__ == "__main__":

    try:
        train()
        # test(model_netG_A2B="netG_A2B_epoch_4.pth", model_netG_B2A="netG_B2A_epoch_4.pth")
        pass

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
