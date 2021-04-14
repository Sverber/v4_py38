#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import csv
import sys
import time
import random
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
from itertools import product
from collections import namedtuple
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim

from torch.utils.data.dataloader import DataLoader
from synthesis.Synthesis import Synthesis

from utils.functions.initialize_weights import initialize_weights

from utils.classes.DecayLR import DecayLR
from utils.classes.ReplayBuffer import ReplayBuffer

from utils.models.cycle.Discriminator import Discriminator
from utils.models.cycle.Generators import Generator

from dataloaders import MyDataLoader

from test import DIR_WEIGHTS, test

os.system("cls")


class Epoch:
    def __init__(self):
        self.count = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = None
        self.duration = None


class Run:
    def __init__(self):
        self.params = None
        self.count = 0
        self.data = []
        self.start_time = None
        self.duration = None


class RunCycleManager:

    """ [ Insert documentation ] """

    def __init__(
        self,
        dataset,
        channels: int,
        parameters: OrderedDict,
        dir_dataset: str = "./dataset",
        dir_outputs: str = "./outputs",
        dir_results: str = "./results",
        dir_weights: str = "./weights",
        save_epoch_freq: int = 1,
        show_image_freq: int = 5,
    ) -> None:

        """ [ Insert documentation ] """

        # Arguments
        self.parameters = parameters
        self.dataset = dataset
        self.dataset_group = dataset.dataset_group
        self.channels = channels

        # Configuration
        self.DIR_DATASET = f"{dir_dataset}/{self.dataset_group}"
        self.DIR_OUTPUTS = f"{dir_outputs}/{self.dataset_group}"
        self.DIR_RESULTS = f"{dir_results}/{self.dataset_group}"
        self.DIR_WEIGHTS = f"{dir_weights}/{self.dataset_group}"
        self.RUN_PATH = None

        self.SAVE_EPOCH_FREQ = save_epoch_freq
        self.SHOW_IMAGE_FREQ = show_image_freq

        # Variables
        self.run = Run()
        self.epoch = Epoch()
        self.net_G_A2B = None
        self.net_G_B2A = None
        self.net_D_A = None
        self.net_D_B = None
        self.loader = None
        self.tb = None

        # MSE losses
        self.avg_mse_loss_A, self.avg_mse_loss_B, self.avg_mse_loss_f_or_A, self.avg_mse_loss_f_or_B = 0, 0, 0, 0
        self.cum_mse_loss_A, self.cum_mse_loss_B, self.cum_mse_loss_f_or_A, self.cum_mse_loss_f_or_B = 0, 0, 0, 0

        # Runs
        self.runs = self.build_cycle(parameters)

    def start_cycle(self) -> None:

        # Iterate over every run, based on the configurated params
        for run in self.runs:

            self.start_run(run, self.dataset)

            pass

    def start_run(self, run, dataset) -> None:

        # Clear occupied CUDA memory
        torch.cuda.empty_cache()

        # Set a random seed for reproducibility
        random.seed(run.manualSeed)
        torch.manual_seed(run.manualSeed)

        self.RUN_PATH = self.get_run_path(run, dataset.name, self.channels)

        # Make required directories for storing the training output
        self.makedirs(path=os.path.join(self.DIR_WEIGHTS, self.RUN_PATH), dir="weights")
        self.makedirs(path=os.path.join(self.DIR_OUTPUTS, self.RUN_PATH), dir="outputs")

        # [TO-DO] Also store a copy of the used PARAMETERS as a config.yaml in the output- and weight dir

        # Create csv for the logs file of this run
        with open(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/logs.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Epoch",
                    "Loss D(A)",
                    "Loss D(B)",
                    "Loss identity A",
                    "Loss identity B",
                    "Loss GAN(A2B)",
                    "Loss GAN(B2A)",
                    "Loss Cycle ABA",
                    "Loss Cycle BAB",
                    "G(A2B)_MSE(avg)",
                    "G(B2A)_MSE(avg)",
                    "G(A2B2A)_MSE(avg)",
                    "G(B2A2B)_MSE(avg)",
                ]
            )

        # Create Generator and Discriminator models
        self.net_G_A2B = Generator(in_channels=self.channels, out_channels=self.channels).to(run.device)
        self.net_G_B2A = Generator(in_channels=self.channels, out_channels=self.channels).to(run.device)
        self.net_D_A = Discriminator(in_channels=self.channels, out_channels=self.channels).to(run.device)
        self.net_D_B = Discriminator(in_channels=self.channels, out_channels=self.channels).to(run.device)

        # Apply weights
        self.net_G_A2B.apply(initialize_weights)
        self.net_G_B2A.apply(initialize_weights)
        self.net_D_A.apply(initialize_weights)
        self.net_D_B.apply(initialize_weights)

        # define loss function (adversarial_loss)
        self.cycle_loss = torch.nn.L1Loss().to(run.device)
        self.identity_loss = torch.nn.L1Loss().to(run.device)
        self.adversarial_loss = torch.nn.MSELoss().to(run.device)

        # Optimizers
        self.optimizer_G_A2B = torch.optim.Adam(self.net_G_A2B.parameters(), lr=run.learning_rate, betas=(0.5, 0.999))
        self.optimizer_G_B2A = torch.optim.Adam(self.net_G_B2A.parameters(), lr=run.learning_rate, betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.net_D_A.parameters(), lr=run.learning_rate, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.net_D_B.parameters(), lr=run.learning_rate, betas=(0.5, 0.999))

        # Learning rates
        self.lr_lambda = DecayLR(run.num_epochs, 0, run.decay_epochs).step
        self.lr_scheduler_G_A2B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G_A2B, lr_lambda=self.lr_lambda)
        self.lr_scheduler_G_B2A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G_B2A, lr_lambda=self.lr_lambda)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=self.lr_lambda)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=self.lr_lambda)

        # Buffers
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataloader
        loader = DataLoader(
            dataset=dataset, batch_size=run.batch_size, num_workers=run.num_workers, shuffle=run.shuffle
        )

        # Track the start of the run
        self.begin_run(run, loader)

        # Iterate through all the epochs
        for epoch in range(0, run.num_epochs):

            # Track the start of the epoch
            self.begin_epoch()

            # Create progress bar
            progress_bar = tqdm(enumerate(loader), total=len(loader))

            # Initiate the cumulative and average error for the discriminator
            self.cum_error_D_A, self.cum_error_D_B = 0, 0
            self.avg_error_D_A, self.avg_error_D_B = 0, 0

            # Iterate over the data loader
            for i, data in progress_bar:

                """ Private functions that regard the training of a GAN during a single run"""

                def __read_data() -> None:

                    # Get image A and image B
                    if self.dataset_group == "l2r":
                        self.real_image_A = data["left"].to(run.device)
                        self.real_image_B = data["right"].to(run.device)

                    elif self.dataset_group == "s2d":
                        real_image_A_left = data["A_left"].to(run.device)
                        real_image_A_right = data["A_right"].to(run.device)
                        real_image_B = data["B"].to(run.device)

                        # Concatenate left- and right view into one stereo image
                        self.real_image_A = torch.cat((real_image_A_left, real_image_A_right), dim=-1)
                        self.real_image_B = real_image_B

                    else:
                        raise Exception(f"Can not read input images, given group '{self.dataset_group}' is incorrect.")

                    # Real data label is 1, fake data label is 0.
                    self.real_label = torch.full((run.batch_size, 1), 1, device=run.device, dtype=torch.float32)
                    self.fake_label = torch.full((run.batch_size, 1), 0, device=run.device, dtype=torch.float32)

                    pass

                def __update_generators() -> None:

                    """ Update Generator networks: A2B and B2A """

                    # Zero the gradients
                    self.optimizer_G_A2B.zero_grad()
                    self.optimizer_G_B2A.zero_grad()

                    # Identity loss of B2A
                    # G_B2A(A) should equal A if real A is fed
                    self.identity_image_A = self.net_G_B2A(self.real_image_A)
                    self.loss_identity_A = self.identity_loss(self.identity_image_A, self.real_image_A) * 5.0

                    # Identity loss of A2B
                    # G_A2B(B) should equal B if real B is fed
                    self.identity_image_B = self.net_G_A2B(self.real_image_B)
                    self.loss_identity_B = self.identity_loss(self.identity_image_B, self.real_image_B) * 5.0

                    # GAN loss: D_A(G_A(A))
                    self.fake_image_A = self.net_G_B2A(self.real_image_B)
                    self.fake_output_A = self.net_D_A(self.fake_image_A)
                    self.loss_GAN_B2A = self.adversarial_loss(self.fake_output_A, self.real_label)

                    # GAN loss: D_B(G_B(B))
                    self.fake_image_B = self.net_G_A2B(self.real_image_A)
                    self.fake_output_B = self.net_D_B(self.fake_image_B)
                    self.loss_GAN_A2B = self.adversarial_loss(self.fake_output_B, self.real_label)

                    # Cycle loss
                    self.recovered_image_A = self.net_G_B2A(self.fake_image_B)
                    self.loss_cycle_ABA = self.cycle_loss(self.recovered_image_A, self.real_image_A) * 10.0

                    self.recovered_image_B = self.net_G_A2B(self.fake_image_A)
                    self.loss_cycle_BAB = self.cycle_loss(self.recovered_image_B, self.real_image_B) * 10.0

                    # Combined loss and calculate gradients
                    self.error_G = (
                        self.loss_identity_A
                        + self.loss_identity_B
                        + self.loss_GAN_A2B
                        + self.loss_GAN_B2A
                        + self.loss_cycle_ABA
                        + self.loss_cycle_BAB
                    )

                    # Calculate gradients for G_A and G_B
                    self.error_G.backward()

                    # Update the Generator networks
                    self.optimizer_G_A2B.step()
                    self.optimizer_G_B2A.step()

                    pass

                def __update_discriminators() -> None:

                    """ Update Discriminator networks A and B"""

                    # Set D_A gradients to zero
                    self.optimizer_D_A.zero_grad()

                    # Real A image loss
                    self.real_output_A = self.net_D_A(self.real_image_A)
                    self.error_D_real_A = self.adversarial_loss(self.real_output_A, self.real_label)

                    # Fake A image loss
                    self.fake_image_A = self.fake_A_buffer.push_and_pop(self.fake_image_A)
                    self.fake_output_A = self.net_D_A(self.fake_image_A.detach())
                    self.error_D_fake_A = self.adversarial_loss(self.fake_output_A, self.fake_label)

                    # Combined loss and calculate gradients
                    self.error_D_A = (self.error_D_real_A + self.error_D_fake_A) / 2

                    # Cumulative and average error of D_A
                    self.cum_error_D_A += self.error_D_A
                    self.avg_error_D_A = self.cum_error_D_A / (i + 1)

                    # Calculate gradients for D_A
                    self.error_D_A.backward()

                    # Update D_A weights
                    self.optimizer_D_A.step()

                    # Set D_B gradients to zero
                    self.optimizer_D_B.zero_grad()

                    # Real B image loss
                    self.real_output_B = self.net_D_B(self.real_image_B)
                    self.error_D_real_B = self.adversarial_loss(self.real_output_B, self.real_label)

                    # Fake B image loss
                    self.fake_image_B = self.fake_B_buffer.push_and_pop(self.fake_image_B)
                    self.fake_output_B = self.net_D_B(self.fake_image_B.detach())
                    self.error_D_fake_B = self.adversarial_loss(self.fake_output_B, self.fake_label)

                    # Combined loss and calculate gradients
                    self.error_D_B = (self.error_D_real_B + self.error_D_fake_B) / 2

                    # Cumulative and average error of D_A
                    self.cum_error_D_B += self.error_D_B
                    self.avg_error_D_B = self.cum_error_D_B / (i + 1)

                    # Calculate gradients for D_B
                    self.error_D_B.backward()

                    # Update D_B weights
                    self.optimizer_D_B.step()

                    pass

                def __regenerate_inputs() -> None:

                    """ Network losses and input data regeneration """

                    """

                        Note:

                        # This is the identity_image_B, perhaps copy the variable itself -> to-do
                        _fake_image_A = self.net_G_A2B(self.real_image_B)

                        # This is the identity_image_A, perhaps copy the variable itself -> to-do
                        _fake_image_B = self.net_G_B2A(self.real_image_A)

                    """

                    # Generate original image from generated (fake) output. So run, respectively, fake A & B image through B2A & A2B
                    self.fake_original_image_A = self.net_G_B2A(self.identity_image_A)
                    self.fake_original_image_B = self.net_G_A2B(self.identity_image_B)

                    # Convert to usable images
                    self.fake_image_A = 0.5 * (self.fake_image_A.data + 1.0)
                    self.fake_image_B = 0.5 * (self.fake_image_B.data + 1.0)

                    # Convert to usable images
                    self.fake_original_image_A = 0.5 * (self.fake_original_image_A.data + 1.0)
                    self.fake_original_image_B = 0.5 * (self.fake_original_image_B.data + 1.0)

                    pass

                def __update_losses() -> None:

                    """ Calculate a cumulative MSE loss and  """

                    # Initiate a mean square error (MSE) loss function
                    mse_loss = nn.MSELoss()

                    # Calculate the mean square error (MSE) loss
                    mse_loss_A = mse_loss(self.fake_image_B, self.real_image_A)
                    mse_loss_B = mse_loss(self.fake_image_A, self.real_image_B)

                    # Calculate the sum of all mean square error (MSE) losses
                    self.cum_mse_loss_A += mse_loss_A
                    self.cum_mse_loss_B += mse_loss_B

                    # Calculate the average mean square error (MSE) loss
                    self.avg_mse_loss_A = self.cum_mse_loss_A / (i + 1)
                    self.avg_mse_loss_B = self.cum_mse_loss_B / (i + 1)

                    """ Calculate losses for the generated (fake) original images """

                    # Calculate the mean square error (MSE) for the generated (fake) originals A and B
                    self.mse_loss_f_or_A = mse_loss(self.fake_original_image_A, self.real_image_A)
                    self.mse_loss_f_or_B = mse_loss(self.fake_original_image_B, self.real_image_B)

                    # Calculate the average mean square error (MSE) for the fake originals A and B
                    self.cum_mse_loss_f_or_A += self.mse_loss_f_or_A
                    self.cum_mse_loss_f_or_B += self.mse_loss_f_or_B

                    # Calculate the average mean square error (MSE) for the fake originals A and B
                    self.avg_mse_loss_f_or_A = self.cum_mse_loss_f_or_A / (i + 1)
                    self.avg_mse_loss_f_or_B = self.cum_mse_loss_f_or_B / (i + 1)

                    pass

                def __save_realtime_output() -> None:

                    """ (5) Save all generated network output and """

                    if i % self.SHOW_IMAGE_FREQ == 0:

                        # Filepath and filename for the real-time output images
                        (
                            filepath_real_A,
                            filepath_real_B,
                            filepath_fake_A,
                            filepath_fake_B,
                            filepath_f_or_A,
                            filepath_f_or_B,
                        ) = (
                            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/real_sample.png",
                            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/real_sample.png",
                            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/fake_sample.png",
                            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/fake_sample.png",
                            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/fake_original.png",
                            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/fake_original.png",
                        )

                        # Save real input images
                        vutils.save_image(self.real_image_A, filepath_real_A, normalize=True)
                        vutils.save_image(self.real_image_B, filepath_real_B, normalize=True)

                        # Save the generated (fake) image
                        vutils.save_image(self.fake_image_A.detach(), filepath_fake_A, normalize=True)
                        vutils.save_image(self.fake_image_B.detach(), filepath_fake_B, normalize=True)

                        # Save the generated (fake) original images
                        vutils.save_image(self.fake_original_image_A.detach(), filepath_f_or_A, normalize=True)
                        vutils.save_image(self.fake_original_image_B.detach(), filepath_f_or_B, normalize=True)

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
                        f"[{self.dataset_group.upper()}][{epoch}/{run.num_epochs}][{i + 1}/{len(loader)}] "
                        # f"L_D(A): {error_D_A.item():.2f} "
                        # f"L_D(B): {error_D_B.item():.2f} | "
                        f"L_D(A+B): {((self.error_D_A + self.error_D_B) / 2).item():.3f} | "
                        f"L_G(A2B): {self.loss_GAN_A2B.item():.3f} "
                        f"L_G(B2A): {self.loss_GAN_B2A.item():.3f} | "
                        # f"L_G_ID: {(loss_identity_A + loss_identity_B).item():.2f} "
                        # f"L_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.2f} "
                        # f"L_G_CYCLE: {(loss_cycle_ABA + loss_cycle_BAB).item():.2f} "
                        f"G(A2B)_MSE(avg): {(self.avg_mse_loss_A).item():.3f} "
                        f"G(B2A)_MSE(avg): {(self.avg_mse_loss_B).item():.3f} | "
                        f"G(A2B2A)_MSE(avg): {(self.avg_mse_loss_f_or_A).item():.3f} "
                        f"G(B2A2B)_MSE(avg): {(self.avg_mse_loss_f_or_B).item():.3f} "
                    )

                    pass

                """ Call the functions to train the GAN """

                # Read data
                __read_data()

                # Update generator networks
                __update_generators()

                # Update discriminator networks
                __update_discriminators()

                # Regerate original input
                __regenerate_inputs()

                # Re-generate "original" input image by running both A2B and B2A through, respectively, B2A and A2B
                __update_losses()

                # Save the real-time output images for every {SHOW_IMG_FREQ} images
                __save_realtime_output()

                # Print a progress bar in the terminal
                __print_progress()

            """ Private functions that regard the training of a GAN at the end of an epoch """

            def __update_learning_rate() -> None:

                self.lr_scheduler_G_A2B.step()
                self.lr_scheduler_G_B2A.step()
                self.lr_scheduler_D_A.step()
                self.lr_scheduler_D_B.step()

            def __save_end_epoch_output() -> None:

                # Filepath and filename for the per-epoch output images
                (
                    filepath_real_A,
                    filepath_real_B,
                    filepath_fake_A,
                    filepath_fake_B,
                    filepath_f_or_A,
                    filepath_f_or_B,
                ) = (
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/epochs/EP{epoch}___real_sample.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/epochs/EP{epoch}___real_sample.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/epochs/EP{epoch}___fake_sample_MSE.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/epochs/EP{epoch}___fake_sample_MSE.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/epochs/EP{epoch}___fake_original_MSE{self.avg_mse_loss_A:.3f}.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/epochs/EP{epoch}___fake_original_MSE{self.avg_mse_loss_B:.3f}.png",
                )

                # Save real input images
                vutils.save_image(self.real_image_A, filepath_real_A, normalize=True)
                vutils.save_image(self.real_image_B, filepath_real_B, normalize=True)

                # Save the generated (fake) image
                vutils.save_image(self.fake_image_A.detach(), filepath_fake_A, normalize=True)
                vutils.save_image(self.fake_image_B.detach(), filepath_fake_B, normalize=True)

                # Save the generated (fake) original images
                vutils.save_image(self.fake_original_image_A.detach(), filepath_f_or_A, normalize=True)
                vutils.save_image(self.fake_original_image_B.detach(), filepath_f_or_B, normalize=True)

                # Check point dir
                model_weights_dir = f"{self.DIR_WEIGHTS}/{self.RUN_PATH}"

                # Check points, save weights after each epoch
                torch.save(self.net_G_A2B.state_dict(), f"{model_weights_dir}/net_G_A2B/net_G_A2B_epoch_{epoch}.pth")
                torch.save(self.net_G_B2A.state_dict(), f"{model_weights_dir}/net_G_B2A/net_G_B2A_epoch_{epoch}.pth")
                torch.save(self.net_D_A.state_dict(), f"{model_weights_dir}/net_D_A/net_D_A_epoch_{epoch}.pth")
                torch.save(self.net_D_B.state_dict(), f"{model_weights_dir}/net_D_B/net_D_B_epoch_{epoch}.pth")

                pass

                # Save the network weights at the end of the epoch

            def __save_end_epoch_logs() -> None:

                with open(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/logs.csv", "a+", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            epoch,
                            f"{self.error_D_A.item():.4f}",
                            f"{self.error_D_B.item():.4f}",
                            f"{self.loss_identity_A.item():.4f}",
                            f"{self.loss_identity_B.item():.4f}",
                            f"{self.loss_GAN_A2B.item():.4f}",
                            f"{self.loss_GAN_B2A.item():.4f}",
                            f"{self.loss_cycle_ABA.item():.4f}",
                            f"{self.loss_cycle_BAB.item():.4f}",
                            f"{(self.avg_mse_loss_A).item():.4f}",
                            f"{(self.avg_mse_loss_B).item():.4f}",
                            f"{(self.avg_mse_loss_f_or_A).item():.4f}",
                            f"{(self.avg_mse_loss_f_or_B).item():.4f}",
                        ]
                    )

                pass

            """ Call the end-of-epoch functions """

            # Update learning rates after each epoch
            __update_learning_rate()

            # Save the output at the end of each epoch
            __save_end_epoch_output()

            # Save some logs at the end of each epoch
            __save_end_epoch_logs()

        """ Save final model """

        # Save last check points, after every run
        torch.save(self.net_G_A2B.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_G_A2B/net_G_A2B.pth")
        torch.save(self.net_G_B2A.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_G_B2A/net_G_B2A.pth")
        torch.save(self.net_D_A.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_D_A/net_D_A.pth")
        torch.save(self.net_D_B.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_D_B/net_D_B.pth")

        self.end_run()

        pass

    def build_cycle(self, parameters) -> list:

        run = namedtuple("Run", parameters.keys())

        self.runs = []

        for v in product(*parameters.values()):
            self.runs.append(run(*v))

        return self.runs

    # Deprecated
    def begin_run(self, run, loader) -> None:

        """ [ Insert documentation ] """

        self.run.start_time = time.time()
        self.run.params = run
        self.run.count += 1

        self.loader = loader

        # self.tb = SummaryWriter(comment=f"-{run}")

        # images, labels = next(iter(self.loader))
        # grid = torchvision.utils.make_grid(images)

        # self.tb.add_image("images", grid)
        # self.tb.add_graph(self.net_G_A2B, images.to(run.device))
        # self.tb.add_graph(self.net_G_A2B, images.to(run.device))
        # self.tb.add_graph(self.net_D_A, images.to(run.device))
        # self.tb.add_graph(self.net_D_B, images.to(run.device))

    # Deprecated
    def end_run(self) -> None:

        """ [ Insert documentation ] """

        # self.tb.close()
        # self.epoch.count = 0

        pass

    # Deprecated
    def begin_epoch(self) -> None:

        """ [ Insert documentation ] """

        self.epoch.start_time = time.time()
        self.epoch.count += 1
        self.epoch.loss = 0
        self.epoch.num_correct = 0

        pass

    # Deprecated
    def end_epoch(self) -> None:

        """ [ Insert documentation ] """

        # print("- [TO-DO] YOLOv5 or newer to implement object detection before")

        # self.epoch.duration = time.time() - self.epoch.start_time
        # self.run.duration = time.time() - self.run.start_time

        # loss = self.epoch.loss / len(self.loader.dataset)
        # accuracy = self.epoch.num_correct / len(self.loader.dataset)

        # self.tb.add_scalar("Loss", loss, self.epoch.count)
        # self.tb.add_scalar("Accuracy", accuracy, self.epoch.count)

        # for name, param in self.network.named_parameters():
        #     self.tb.add_histogram(name, param, self.epoch.count)
        #     self.tb.add_histogram(f"{name}.grad", param.grad, self.epoch.count)

        # results = OrderedDict()
        # results["run"] = self.run.count
        # results["epoch"] = self.epoch.count
        # results["loss"] = loss
        # results["accuracy"] = accuracy
        # results["epoch duration"] = self.epoch.duration
        # results["run duration"] = self.run.duration
        # for k, v in self.run.params._asdict().items():
        #     results[k] = v
        # self.run.data.append(results)

        # if print_df:
        #     pprint.pprint(
        #         pd.DataFrame.from_dict(self.run.data, orient="columns").sort_values("accuracy", ascending=False)
        #     )

    # Deprecated
    def track_loss(
        self, i, real_image_A, real_image_B, fake_image_A, fake_image_B, fake_original_image_A, fake_original_image_B
    ) -> None:

        """ [ Insert documentation ] """

        # Initiate a mean square error (MSE) loss function
        mse_loss = nn.MSELoss()

        # Calculate the mean square error (MSE) loss

        # print("")
        # print("fake_image_B.shape, real_image_A.shape", fake_image_B.shape, real_image_A.shape)
        # print("fake_image_A.shape, real_image_B.shape", fake_image_A.shape, real_image_B.shape)
        mse_loss_A = mse_loss(self.fake_image_B, self.real_image_A)
        mse_loss_B = mse_loss(self.fake_image_A, self.real_image_B)

        # Calculate the sum of all mean square error (MSE) losses
        self.cum_mse_loss_A += mse_loss_A
        self.cum_mse_loss_B += mse_loss_B

        # Calculate the average mean square error (MSE) loss
        self.avg_mse_loss_A = self.cum_mse_loss_A / (i + 1)
        self.avg_mse_loss_B = self.cum_mse_loss_B / (i + 1)

        """ Calculate losses for the generated (fake) original images """

        # Calculate the mean square error (MSE) for the generated (fake) originals A and B

        # print("fake_original_image_A.shape, real_image_B.shape", fake_original_image_A.shape, real_image_A.shape)
        # print("fake_original_image_B.shape, real_image_A.shape", fake_original_image_B.shape, real_image_B.shape)
        # print("")
        self.mse_loss_f_or_A = mse_loss(self.fake_original_image_A, self.real_image_A)
        self.mse_loss_f_or_B = mse_loss(self.fake_original_image_B, self.real_image_B)

        # Calculate the average mean square error (MSE) for the fake originals A and B
        self.cum_mse_loss_f_or_A += self.mse_loss_f_or_A
        self.cum_mse_loss_f_or_B += self.mse_loss_f_or_B

        # Calculate the average mean square error (MSE) for the fake originals A and B
        self.avg_mse_loss_f_or_A = self.cum_mse_loss_f_or_A / (i + 1)
        self.avg_mse_loss_f_or_B = self.cum_mse_loss_f_or_B / (i + 1)

        pass

        # self.epoch.loss += loss.item() * batch[0].shape[0]

    @staticmethod
    def makedirs(path: str, dir: str):

        if dir == "outputs":
            try:
                os.makedirs(os.path.join(path, "A"))
                os.makedirs(os.path.join(path, "B"))
                os.makedirs(os.path.join(path, "A", "epochs"))
                os.makedirs(os.path.join(path, "B", "epochs"))
            except OSError:
                pass

        elif dir == "weights":
            try:
                os.makedirs(os.path.join(path, "net_G_A2B"))
                os.makedirs(os.path.join(path, "net_G_B2A"))
                os.makedirs(os.path.join(path, "net_D_A"))
                os.makedirs(os.path.join(path, "net_D_B"))
            except OSError:
                pass

    @staticmethod
    def get_run_path(run, dataset_name, channels) -> str:

        # Store today's date in string format
        TODAY_DATE = datetime.today().strftime("%Y-%m-%d")
        TODAY_TIME = datetime.today().strftime("%H.%M.%S")

        digits = len(str(run.num_epochs))

        # Create a unique name for this run
        RUN_NAME = f"{TODAY_TIME}___EP{str(run.num_epochs).zfill(digits)}_DE{str(run.decay_epochs).zfill(digits)}_LR{run.learning_rate}_CH{channels}"

        RUN_PATH = f"{dataset_name}/{TODAY_DATE}/{RUN_NAME}"

        return RUN_PATH


PARAMETERS: OrderedDict = OrderedDict(
    device=[torch.device("cuda" if torch.cuda.is_available() else "cpu")],
    shuffle=[True],
    num_workers=[4],
    manualSeed=[999],
    learning_rate=[0.0002],
    batch_size=[1],
    num_epochs=[100],
    decay_epochs=[50],
)


# Execute main code
if __name__ == "__main__":

    try:

        mydataloader = MyDataLoader()

        """ Train a [L2R] model on the GRAYSCALE dataset """

        # l2r_dataset_train_GRAYSCALE = mydataloader.get_dataset("l2r", "Test_Set_GRAYSCALE", "train", (68, 120), 1, True)
        # l2r_manager_GRAYSCALE = RunCycleManager(l2r_dataset_train_GRAYSCALE, 1, PARAMETERS)
        # l2r_manager_GRAYSCALE.start_cycle()

        """ Train a [L2R] model on the RGB dataset """

        # l2r_dataset_train_RGB = mydataloader.get_dataset("l2r", "Test_Set_GRAYSCALE", "train", (68, 120), 1, True)
        # l2r_manager_RGB = RunCycleManager(l2r_dataset_train_RGB, 1, PARAMETERS)
        # l2r_manager_RGB.start_cycle()

        """ Train a [S2D] model on the GRAYSCALE dataset """

        # s2d_dataset_train_GRAYSCALE = mydataloader.get_dataset(
        #     "s2d", "Test_Set_GRAYSCALE", "train", (68, 120), 1, False
        # )
        # s2d_manager_GRAYSCALE = RunCycleManager(s2d_dataset_train_GRAYSCALE, 1, PARAMETERS)
        # s2d_manager_GRAYSCALE.start_cycle()

        """ Train a [S2D] model on the RGB dataset """

        s2d_dataset_train_RGB = mydataloader.get_dataset("s2d", "Test_Set_RGB", "train", (68, 120), 3, False)
        s2d_manager_RGB = RunCycleManager(s2d_dataset_train_RGB, 3, PARAMETERS)
        s2d_manager_RGB.start_cycle()

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
