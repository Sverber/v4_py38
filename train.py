#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

#%matplotlib inline

import os
import csv
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from itertools import product
from collections import namedtuple
from collections import OrderedDict


from utils.classes.DecayLR import DecayLR
from utils.classes.dataloaders import MyDataLoader
from utils.classes.ReplayBuffer import ReplayBuffer
from utils.models.cycle.Generators import Generator
from utils.models.cycle.Discriminator import Discriminator


# Clear the terminal
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
        show_graph_freq: int = 10,
        validation_percentage: float = 0.0,
    ) -> None:

        """ [ Insert documentation ] """

        # Arguments
        self.parameters = parameters
        self.dataset = dataset
        self.dataset_group = dataset.dataset_group
        self.channels = channels
        self.validation_percentage = validation_percentage

        # Configuration
        self.DIR_DATASET = f"{dir_dataset}/{self.dataset_group}"
        self.DIR_OUTPUTS = f"{dir_outputs}/{self.dataset_group}"
        self.DIR_RESULTS = f"{dir_results}/{self.dataset_group}"
        self.DIR_WEIGHTS = f"{dir_weights}/{self.dataset_group}"
        self.SHOW_IMAGE_FREQ = show_image_freq
        self.SHOW_GRAPH_FREQ = show_graph_freq
        self.SAVE_EPOCH_FREQ = save_epoch_freq
        self.RUN_PATH = None

        # Variables
        self.run = Run()
        self.epoch = Epoch()
        self.net_G_A2B = None
        self.net_G_B2A = None
        self.net_D_A = None
        self.net_D_B = None
        self.loader = None
        self.validation_batch_index = None
        self.batch_is_validation = False

        # Runs
        self.runs = self.__build_cycle(parameters)

    def start_cycle(self) -> None:

        # Iterate over every run, based on the configurated params
        for run in self.runs:

            # Clear occupied CUDA memory
            torch.cuda.empty_cache()

            # Set a random seed for reproducibility
            random.seed(run.manualSeed)
            torch.manual_seed(run.manualSeed)

            self.RUN_PATH = self.get_run_path(run, self.dataset.name, self.channels)

            # Make required directories for storing the training output
            self.makedirs(path=os.path.join(self.DIR_WEIGHTS, self.RUN_PATH), dir="weights")
            self.makedirs(path=os.path.join(self.DIR_OUTPUTS, self.RUN_PATH), dir="outputs")

            # Create a per-epoch csv log file
            self.__create_per_epoch_csv_logs()

            # Create Generator and Discriminator models
            self.net_G_A2B = Generator(in_channels=self.channels, out_channels=self.channels).to(run.device)
            self.net_G_B2A = Generator(in_channels=self.channels, out_channels=self.channels).to(run.device)
            self.net_D_A = Discriminator(in_channels=self.channels, out_channels=self.channels).to(run.device)
            self.net_D_B = Discriminator(in_channels=self.channels, out_channels=self.channels).to(run.device)

            # Apply weights
            self.net_G_A2B.apply(self.__initialize_weights)
            self.net_G_B2A.apply(self.__initialize_weights)
            self.net_D_A.apply(self.__initialize_weights)
            self.net_D_B.apply(self.__initialize_weights)

            # define loss functions
            self.cycle_loss = torch.nn.L1Loss().to(run.device)
            self.identity_loss = torch.nn.L1Loss().to(run.device)
            self.adversarial_loss = torch.nn.MSELoss().to(run.device)

            # Optimizers
            self.optimizer_G_A2B = torch.optim.Adam(
                self.net_G_A2B.parameters(), lr=run.learning_rate, betas=(0.5, 0.999)
            )
            self.optimizer_G_B2A = torch.optim.Adam(
                self.net_G_B2A.parameters(), lr=run.learning_rate, betas=(0.5, 0.999)
            )
            self.optimizer_D_A = torch.optim.Adam(self.net_D_A.parameters(), lr=run.learning_rate, betas=(0.5, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.net_D_B.parameters(), lr=run.learning_rate, betas=(0.5, 0.999))

            # Learning rates
            self.lr_lambda = DecayLR(run.num_epochs, 0, run.decay_epochs).step
            self.lr_scheduler_G_A2B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G_A2B, lr_lambda=self.lr_lambda)
            self.lr_scheduler_G_B2A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G_B2A, lr_lambda=self.lr_lambda)
            self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=self.lr_lambda)
            self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=self.lr_lambda)

            # Buffers train
            self.fake_A_buffer = ReplayBuffer()
            self.fake_B_buffer = ReplayBuffer()

            # Dataloader train set
            self.loader = DataLoader(
                dataset=self.dataset, batch_size=run.batch_size, num_workers=run.num_workers, shuffle=run.shuffle
            )

            # Determine the batch-index from which the validation set starts
            self.validation_batch_index: int = int(round(int(len(self.loader) * (1 - self.validation_percentage)), 0))

            # Keep track of the per-epoch losses
            self.losses_G_A, self.losses_G_B = [], []
            self.losses_D_A, self.losses_D_B = [], []

            # Keep track of the per-epoch noise factor and learning rate
            self.noise_factor_array = []

            """ Iterate over the epochs in the run """

            # Iterate through all the epochs
            for epoch in range(0, run.num_epochs):

                # Keep track of the per-batch losses during one epoch
                self.batch_losses_G_A, self.batch_losses_G_B = [], []
                self.batch_losses_D_A, self.batch_losses_D_B = [], []
                self.full_batch_losses_G_A, self.full_batch_losses_G_B = [], []

                # Create a per-batch csv log file
                self.__create_per_batch_csv_logs(epoch)

                # Set error variables at 0 at the begin of each epoch
                self.__set_error_variables_to_zero()

                """ Iterate over training data using the progress bar """

                # Create progress bar
                self.progress_bar = tqdm(enumerate(self.loader), total=len(self.loader))

                # Iterate over the data loader_train
                for i, data in self.progress_bar:

                    try:

                        """ Determine whether this batch is for training or validation """

                        self.batch_is_validation = True if i > self.validation_batch_index else False

                        """ Call the functions to train the GAN """

                        # Read data
                        self.__read_data(run, data)

                        # Update generator networks
                        self.__update_generators(i)

                        # Add noise to the discriminator input
                        self.__add_discriminator_noise(epoch, run)

                        # Update discriminator networks
                        self.__update_discriminators(i)

                        # Save the real-time output images for every {SHOW_IMG_FREQ} images
                        self.__save_realtime_output(i)

                        # Save per-epoch logs
                        self.__save_per_epoch_logs(epoch)

                        # Save the per-batch losses in a plot
                        self.__save_plot_per_batch(i, epoch)

                        # Print a progress bar in the terminal
                        self.__print_progress(i, epoch, run)

                    except Exception as e:

                        print(e)

                        pass

                """ Call the end-of-epoch functions """

                # Update learning rates after each epoch
                self.__update_learning_rate()

                # Save the output at the end of each epoch
                self.__save_end_epoch_output(epoch)

                # Save some logs at the end of each epoch
                self.__save_end_epoch_logs(epoch, run)

                # Save the losses in a plot of each epoch
                self.__save_plot_per_epoch(epoch)

            """ Save final model """

            # Save last check points, after every run
            torch.save(self.net_G_A2B.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_G_A2B/net_G_A2B.pth")
            torch.save(self.net_G_B2A.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_G_B2A/net_G_B2A.pth")
            torch.save(self.net_D_A.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_D_A/net_D_A.pth")
            torch.save(self.net_D_B.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_D_B/net_D_B.pth")

        pass

    def __build_cycle(self, parameters) -> list:

        run = namedtuple("Run", parameters.keys())

        self.runs = []

        for v in product(*parameters.values()):
            self.runs.append(run(*v))

        return self.runs

    """ [ Private functions ] Called once per batch """

    def __initialize_weights(self, m):

        """ Custom weights initialization called on net_G and net_D """

        classname = m.__class__.__name__

        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)

        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

        pass

    def __set_error_variables_to_zero(self) -> None:

        """ Variables for error tracking on train set """

        # Per batch error for D_A, D_B
        self.error_D_A, self.error_D_B = 0, 0

        # Average error on D_A, D_B
        self.cum_error_D_A, self.cum_error_D_B = 0, 0
        self.avg_error_D_A, self.avg_error_D_B = 0, 0

        # Per batch error for G
        self.error_G_A, self.error_G_B = 0, 0

        # Average error on G
        self.cum_error_G_A, self.avg_error_G_A = 0, 0
        self.cum_error_G_B, self.avg_error_G_B = 0, 0

        """ Variables for error tracking on validation set """

        # Per batch error for D_A, D_B
        self.v__error_D_A, self.v__error_D_B = 0, 0

        # Average error on D_A, D_B
        self.v__cum_error_D_A, self.v__cum_error_D_B = 0, 0
        self.v__avg_error_D_A, self.v__avg_error_D_B = 0, 0

        # Per batch error for G
        self.v__error_G_A, self.v__error_G_B = 0, 0

        # Average error on G
        self.v__cum_error_G_A, self.v__avg_error_G_A = 0, 0
        self.v__cum_error_G_B, self.v__avg_error_G_B = 0, 0

        pass

    def __create_per_epoch_csv_logs(self) -> None:

        """ Create a per-epoch csv logs file of the current run """

        # Create csv for the logs file of this run
        with open(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/logs.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Epoch",
                    "Noise factor",
                    "avg_error_D_B",
                    "avg_error_G_A",
                    "avg_error_G_B",
                    "v__avg_error_D_A",
                    "v__avg_error_D_B",
                    "v__avg_error_G_A",
                    "v__avg_error_G_B",
                ]
            )

        pass

    def __create_per_batch_csv_logs(self, epoch: int) -> None:

        """ Create a per-batch csv logs file of the current epoch """

        with open(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/logs/EP{epoch}__logs.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Epoch",
                    "error_D_A",
                    "error_D_B",
                    "error_G_A",
                    "error_G_B",
                    "v_error_D_A",
                    "v_error_D_B",
                    "v_error_G_A",
                    "v_error_G_B",
                ]
            )

        pass

    def __random_flip(self, real_label: torch.Tensor, fake_label: torch.Tensor, probability: float = 0.1):

        """ Randomly flip labels following a given probability """

        random_percentage = random.uniform(0, 1)

        if random_percentage > probability:
            return real_label, fake_label
        else:
            return fake_label, real_label

    def __smooth_one_hot(self, true_label: torch.Tensor, classes: int, smoothing: float = 0.1):

        """ Smoothen one-hot encoced labels y_ls = (1 - α) * y_hot + α / K """

        smooth_label = (1 - smoothing) * true_label + smoothing / classes

        return smooth_label

    def __read_data(self, run, data) -> None:

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
        self.real_label = torch.full((run.batch_size, self.channels), 1, device=run.device, dtype=torch.float32)
        self.fake_label = torch.full((run.batch_size, self.channels), 0, device=run.device, dtype=torch.float32)

        pass

    def __add_discriminator_noise(self, epoch, run) -> None:

        """ Add decaying Gaussian noise to discriminator A and B real/fake inputs """

        # Set mean and standard deviation
        mean, std = 0.5, 0.5

        # Calculate until which epoch there is noise
        noise_until_percentage = 0.90

        # Noise is gone at epoch 80, if the run is set to a 100 epochs
        noise_until_epoch = noise_until_percentage * run.num_epochs

        # Calculate noise factor
        if epoch > 0:
            self.noise_factor = round(1 - (epoch / noise_until_epoch), 3)
        else:
            self.noise_factor = 1

        # Create the noise for the real images
        noise_real_A = (torch.randn(self.real_image_A.size()) * std + mean).to(run.device)
        noise_real_B = (torch.randn(self.real_image_B.size()) * std + mean).to(run.device)

        # Create the noise for the fake images
        noise_fake_A = (torch.randn(self.fake_image_A.size()) * std + mean).to(run.device)
        noise_fake_B = (torch.randn(self.fake_image_B.size()) * std + mean).to(run.device)

        # Add decaying noise to the real images (only used by discriminator)
        self.real_image_A_noise = self.real_image_A + (noise_real_A * self.noise_factor)
        self.real_image_B_noise = self.real_image_B + (noise_real_B * self.noise_factor)

        # Add decaying noise to the fake images (only used by discriminator)
        self.fake_image_A_noise = self.fake_image_A + (noise_fake_A * self.noise_factor)
        self.fake_image_B_noise = self.fake_image_B + (noise_fake_B * self.noise_factor)

        """ Label smoothing and random flipping """

        # Smoothen the labels (only used by the discriminator)
        self.real_smooth_label = self.__smooth_one_hot(self.real_label, 2, 0.1)
        self.fake_smooth_label = self.__smooth_one_hot(self.fake_label, 2, 0.1)

        # Randomly flip the smooth labels (only used by the discriminator)
        self.real_smooth_label, self.fake_smooth_label = self.__random_flip(
            self.real_smooth_label, self.fake_smooth_label, 0.10
        )

        pass

    def __update_generators(self, i) -> None:

        """ Update Generator networks: A2B and B2A """

        # Only zero the gradients when using training data
        if self.batch_is_validation == False:

            # Zero the gradients
            self.optimizer_G_A2B.zero_grad()
            self.optimizer_G_B2A.zero_grad()

        """ Generator loss """

        # GAN loss: D_A(G_A(A))
        self.fake_image_A = self.net_G_B2A(self.real_image_B)
        self.fake_output_A = self.net_D_A(self.fake_image_A)
        self.loss_GAN_B2A = self.adversarial_loss(self.fake_output_A, self.real_label)

        # GAN loss: D_B(G_B(B))
        self.fake_image_B = self.net_G_A2B(self.real_image_A)
        self.fake_output_B = self.net_D_B(self.fake_image_B)
        self.loss_GAN_A2B = self.adversarial_loss(self.fake_output_B, self.real_label)

        """ Identity loss: helps to preserve colour and prevent reverse colour in the result  """

        lambda_A = 10  # 10 by default
        lambda_B = 10  # 10 by default
        lamba_identity = 0.5  # 0.5 by default

        # G_A2B should be identity if real_image_B is fed: || G_A2B(B) - B ||
        self.identity_image_A = self.net_G_A2B(self.real_image_B)
        self.loss_identity_A = self.identity_loss(self.identity_image_A, self.real_image_B) * lambda_B * lamba_identity

        # G_B2A should be identity if real_image_A is fed: || G_B2A(A) - A ||
        self.identity_image_B = self.net_G_B2A(self.real_image_A)
        self.loss_identity_B = self.identity_loss(self.identity_image_B, self.real_image_A) * lambda_A * lamba_identity

        """ Cycle loss """

        # Cycle loss
        self.recovered_image_A = self.net_G_B2A(self.fake_image_B)
        self.loss_cycle_ABA = self.cycle_loss(self.recovered_image_A, self.real_image_A) * lambda_A

        self.recovered_image_B = self.net_G_A2B(self.fake_image_A)
        self.loss_cycle_BAB = self.cycle_loss(self.recovered_image_B, self.real_image_B) * lambda_B

        """ Calculate the generator error """

        # Error G_A (removed: self.loss_identity_A)
        self.error_G_A = self.loss_GAN_A2B + self.loss_identity_A + self.loss_cycle_ABA

        # Error G_B (removed: self.loss_identity_B)
        self.error_G_B = self.loss_GAN_B2A + self.loss_identity_B + self.loss_cycle_BAB

        # Average error on G_A
        self.cum_error_G_A += self.error_G_A
        self.avg_error_G_A = self.cum_error_G_A / (i + 1)

        # Average error on G_B
        self.cum_error_G_B += self.error_G_B
        self.avg_error_G_B = self.cum_error_G_B / (i + 1)

        # Only update weights when using training data
        if self.batch_is_validation == False:

            # Calculate gradients for G_A and G_B
            self.error_G_A.backward()
            self.error_G_B.backward()

            # Update the Generator networks
            self.optimizer_G_A2B.step()
            self.optimizer_G_B2A.step()

        pass

    def __update_discriminators(self, i) -> None:

        """" Update discriminator A """

        # Only zero the gradient when using training data
        if self.batch_is_validation == False:

            # Set D_A gradients to zero
            self.optimizer_D_A.zero_grad()

        # Real A image loss
        self.real_output_A = self.net_D_A(self.real_image_A_noise)
        self.error_D_real_A = self.adversarial_loss(self.real_output_A, self.real_smooth_label)

        # Fake image A loss
        self.fake_image_A_noise = self.fake_A_buffer.push_and_pop(self.fake_image_A_noise)
        self.fake_output_A = self.net_D_A(self.fake_image_A_noise.detach())
        self.error_D_fake_A = self.adversarial_loss(self.fake_output_A, self.fake_smooth_label)

        # Combined loss and calculate gradients
        self.error_D_A = (self.error_D_real_A + self.error_D_fake_A) / 2

        # Cumulative and average error of D_A
        self.cum_error_D_A += self.error_D_A
        self.avg_error_D_A = self.cum_error_D_A / (i + 1)

        # Only update weights when using training data
        if self.batch_is_validation == False:

            # Calculate gradients for D_A
            self.error_D_A.backward()

            # Update D_A weights
            self.optimizer_D_A.step()

        """" Update discriminator B """

        # Only zero the gradient when using training data
        if self.batch_is_validation == False:

            # Set D_B gradients to zero
            self.optimizer_D_B.zero_grad()

        # Real B image loss
        self.real_output_B = self.net_D_B(self.real_image_B_noise)
        self.error_D_real_B = self.adversarial_loss(self.real_output_B, self.real_smooth_label)

        # Fake image B loss
        self.fake_image_B_noise = self.fake_B_buffer.push_and_pop(self.fake_image_B_noise)
        self.fake_output_B = self.net_D_B(self.fake_image_B_noise.detach())
        self.error_D_fake_B = self.adversarial_loss(self.fake_output_B, self.fake_smooth_label)

        # Combined loss and calculate gradients
        self.error_D_B = (self.error_D_real_B + self.error_D_fake_B) / 2

        # Cumulative and average error of D_A
        self.cum_error_D_B += self.error_D_B
        self.avg_error_D_B = self.cum_error_D_B / (i + 1)

        # Only update weights when using training data
        if self.batch_is_validation == False:

            # Calculate gradients for D_B
            self.error_D_B.backward()

            # Update D_B weights
            self.optimizer_D_B.step()

        pass

    # Currently not used, due to CUDA memory issues
    def __regenerate_inputs(self) -> None:

        """ Network losses and input data regeneration """

        """

            Note:

            # This is the identity_image_B, perhaps copy the variable itself -> to-do
            _fake_image_A = self.net_G_A2B(self.real_image_B)

            # This is the identity_image_A, perhaps copy the variable itself -> to-do
            _fake_image_B = self.net_G_B2A(self.real_image_A)

        """

        # # Generate original image from generated (fake) output. So run, respectively, fake A & B image through B2A & A2B
        # self.fake_original_image_A = self.net_G_B2A(self.identity_image_A)
        # self.fake_original_image_B = self.net_G_A2B(self.identity_image_B)

        # # Convert to usable images (changes the tensor range from [-1, 1] to [0, 1])
        # self.fake_image_A = 0.5 * (self.fake_image_A.data + 1.0)
        # self.fake_image_B = 0.5 * (self.fake_image_B.data + 1.0)

        # # Convert to usable images (changes the tensor range from [-1, 1] to [0, 1])
        # self.fake_original_image_A = 0.5 * (self.fake_original_image_A.data + 1.0)
        # self.fake_original_image_B = 0.5 * (self.fake_original_image_B.data + 1.0)

        pass

    # Currently not used, due to CUDA memory issues
    def __update_losses(self) -> None:

        """ Calculate a cumulative MSE loss and  """

        # # Initiate a mean square error (MSE) loss function
        # mse_loss = nn.MSELoss()

        # # Calculate the mean square error (MSE) loss
        # mse_loss_A = mse_loss(self.fake_image_B, self.real_image_A)
        # mse_loss_B = mse_loss(self.fake_image_A, self.real_image_B)

        # # Calculate the sum of all mean square error (MSE) losses
        # self.cum_mse_loss_A += mse_loss_A
        # self.cum_mse_loss_B += mse_loss_B

        # # Calculate the average mean square error (MSE) loss
        # self.avg_mse_loss_A = self.cum_mse_loss_A / (i + 1)
        # self.avg_mse_loss_B = self.cum_mse_loss_B / (i + 1)

        # """ Calculate losses for the generated (fake) original images """

        # # Calculate the mean square error (MSE) for the generated (fake) originals A and B
        # self.mse_loss_f_or_A = mse_loss(self.fake_original_image_A, self.real_image_A)
        # self.mse_loss_f_or_B = mse_loss(self.fake_original_image_B, self.real_image_B)

        # # Calculate the average mean square error (MSE) for the fake originals A and B
        # self.cum_mse_loss_f_or_A += self.mse_loss_f_or_A
        # self.cum_mse_loss_f_or_B += self.mse_loss_f_or_B

        # # Calculate the average mean square error (MSE) for the fake originals A and B
        # self.avg_mse_loss_f_or_A = self.cum_mse_loss_f_or_A / (i + 1)
        # self.avg_mse_loss_f_or_B = self.cum_mse_loss_f_or_B / (i + 1)

        pass

    def __save_realtime_output(self, i) -> None:

        """ (5) Save all generated network output and """

        if i % self.SHOW_IMAGE_FREQ == 0:

            if self.batch_is_validation == False:

                # Filepath and filename for the real-time TRAINING output images
                (
                    filepath_real_A,
                    filepath_real_B,
                    filepath_real_A_noise,
                    filepath_real_B_noise,
                    filepath_fake_A,
                    filepath_fake_B,
                    filepath_fake_A_noise,
                    filepath_fake_B_noise,
                    # filepath_f_or_A,
                    # filepath_f_or_B,
                ) = (
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/real_sample.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/real_sample.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/real_sample_noise.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/real_sample_noise.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/fake_sample.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/fake_sample.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/fake_sample_noise.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/fake_sample_noise.png",
                    # f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/fake_original.png",
                    # f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/fake_original.png",
                )

            else:
                #
                #  Filepath and filename for the real-time VALIDATION output images
                (
                    filepath_real_A,
                    filepath_real_B,
                    filepath_real_A_noise,
                    filepath_real_B_noise,
                    filepath_fake_A,
                    filepath_fake_B,
                    filepath_fake_A_noise,
                    filepath_fake_B_noise,
                    # filepath_f_or_A,
                    # filepath_f_or_B,
                ) = (
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/v__real_sample.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/v__real_sample.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/v__real_sample_noise.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/v__real_sample_noise.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/v__fake_sample.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/v__fake_sample.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/v__fake_sample_noise.png",
                    f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/v__fake_sample_noise.png",
                    # f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/v__fake_original.png",
                    # f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/v__fake_original.png",
                )

            # Save real input images
            vutils.save_image(self.real_image_A, filepath_real_A, normalize=True)
            vutils.save_image(self.real_image_B, filepath_real_B, normalize=True)

            # Save real input images with noise
            vutils.save_image(self.real_image_A_noise, filepath_real_A_noise, normalize=True)
            vutils.save_image(self.real_image_B_noise, filepath_real_B_noise, normalize=True)

            # Save the generated (fake) image
            vutils.save_image(self.fake_image_A.detach(), filepath_fake_A, normalize=True)
            vutils.save_image(self.fake_image_B.detach(), filepath_fake_B, normalize=True)

            # Save the generated (fake) image
            vutils.save_image(self.fake_image_A_noise.detach(), filepath_fake_A_noise, normalize=True)
            vutils.save_image(self.fake_image_B_noise.detach(), filepath_fake_B_noise, normalize=True)

            # # Save the generated (fake) original images
            # vutils.save_image(self.fake_original_image_A.detach(), filepath_f_or_A, normalize=True)
            # vutils.save_image(self.fake_original_image_B.detach(), filepath_f_or_B, normalize=True)

        pass

    def __save_per_epoch_logs(self, epoch: int) -> None:

        # Create csv for the logs file of this run
        with open(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/logs/EP{epoch}__logs.csv", "a+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Epoch",
                    f"{self.error_D_A:.4f}",
                    f"{self.error_D_B:.4f}",
                    f"{self.error_G_A:.4f}",
                    f"{self.error_G_B:.4f}",
                    f"{self.v__error_D_A:.4f}",
                    f"{self.v__error_D_B:.4f}",
                    f"{self.v__error_G_A:.4f}",
                    f"{self.v__error_G_B:.4f}",
                ]
            )

        pass

    def __print_progress(self, i, epoch, run) -> None:

        """ Print progress """

        self.progress_bar.set_description(
            f"[{self.dataset_group.upper()}][{epoch}/{run.num_epochs}][{i + 1}/{len(self.loader)}][val={self.batch_is_validation}][val_index={self.validation_batch_index}][nf={self.noise_factor:.3f}]  ||  "
            f"avg_error_D_A: {self.avg_error_D_A:.3f} ; "
            f"avg_error_D_B: {self.avg_error_D_B:.3f}  ||  "
            f"avg_error_G_A2B: {self.avg_error_G_A:.3f} ; "
            f"avg_error_G_B2A: {self.avg_error_G_B:.3f}  ||  "
        )

        pass

    """ [ Private functions ] Called once per epoch """

    def __update_learning_rate(self) -> None:

        self.lr_scheduler_G_A2B.step()
        self.lr_scheduler_G_B2A.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_D_B.step()

    def __save_end_epoch_output(self, epoch) -> None:

        # Filepath and filename for the per-epoch output images
        (
            filepath_real_A,
            filepath_real_B,
            filepath_real_A_noise,
            filepath_real_B_noise,
            filepath_fake_A,
            filepath_fake_B,
            filepath_fake_A_noise,
            filepath_fake_B_noise,
        ) = (
            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/epochs/EP{epoch}___real_sample.png",
            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/epochs/EP{epoch}___real_sample.png",
            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/epochs/EP{epoch}___real_sample_noise.png",
            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/epochs/EP{epoch}___real_sample_noise.png",
            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/epochs/EP{epoch}___fake_sample.png",
            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/epochs/EP{epoch}___fake_sample.png",
            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/A/epochs/EP{epoch}___fake_sample_noise.png",
            f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/B/epochs/EP{epoch}___fake_sample_noise.png",
        )

        # Save real input images
        vutils.save_image(self.real_image_A, filepath_real_A, normalize=True)
        vutils.save_image(self.real_image_B, filepath_real_B, normalize=True)

        # Save real input images with noise
        vutils.save_image(self.real_image_A_noise, filepath_real_A_noise, normalize=True)
        vutils.save_image(self.real_image_B_noise, filepath_real_B_noise, normalize=True)

        # Save the generated (fake) image
        vutils.save_image(self.fake_image_A.detach(), filepath_fake_A, normalize=True)
        vutils.save_image(self.fake_image_B.detach(), filepath_fake_B, normalize=True)

        # Save the generated (fake) image
        vutils.save_image(self.fake_image_A_noise.detach(), filepath_fake_A_noise, normalize=True)
        vutils.save_image(self.fake_image_B_noise.detach(), filepath_fake_B_noise, normalize=True)

        # Check point dir
        model_weights_dir = f"{self.DIR_WEIGHTS}/{self.RUN_PATH}"

        # Check points, save weights after each epoch
        torch.save(self.net_G_A2B.state_dict(), f"{model_weights_dir}/net_G_A2B/net_G_A2B_epoch_{epoch}.pth")
        torch.save(self.net_G_B2A.state_dict(), f"{model_weights_dir}/net_G_B2A/net_G_B2A_epoch_{epoch}.pth")
        torch.save(self.net_D_A.state_dict(), f"{model_weights_dir}/net_D_A/net_D_A_epoch_{epoch}.pth")
        torch.save(self.net_D_B.state_dict(), f"{model_weights_dir}/net_D_B/net_D_B_epoch_{epoch}.pth")

        pass

    def __save_end_epoch_logs(self, epoch, run) -> None:

        with open(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/logs.csv", "a+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    epoch,
                    f"{self.noise_factor:.3f}",
                    f"{self.avg_error_D_A:.5f}",
                    f"{self.avg_error_D_B:.5f}",
                    f"{self.avg_error_G_A:.5f}",
                    f"{self.avg_error_G_B:.5f}",
                    f"{self.v__avg_error_D_A:.5f}",
                    f"{self.v__avg_error_D_B:.5f}",
                    f"{self.v__avg_error_G_A:.5f}",
                    f"{self.v__avg_error_G_B:.5f}",
                ]
            )

        pass

    """ [ Private functions ] Plotting function """

    def __save_plot_per_batch(self, i, epoch) -> None:

        """ Append the current per-batch losses to the arrays containing the network losses """

        self.batch_losses_G_A.append(self.error_G_A.cpu().detach().numpy())
        self.batch_losses_G_B.append(self.error_G_B.cpu().detach().numpy())
        self.batch_losses_D_A.append(self.error_D_A.cpu().detach().numpy())
        self.batch_losses_D_B.append(self.error_D_B.cpu().detach().numpy())

        """ Plot losses """

        if i % self.SHOW_GRAPH_FREQ == 0:

            # Create figure
            self.per_batch_figure, self.per_batch_axes = plt.subplots(2)

            # Set titles
            self.per_batch_axes[0].set_title(f"Generator A and Generator B loss during epoch {epoch}")
            self.per_batch_axes[1].set_title(f"Discriminator A and Discriminator B loss during epoch {epoch}")

            # Set labels
            self.per_batch_axes[0].set(xlabel="Batch", ylabel="G Loss")
            self.per_batch_axes[1].set(xlabel="Batch", ylabel="D Loss")

            # Add gridlines
            self.per_batch_axes[0].grid()
            self.per_batch_axes[1].grid()

            # Plot generator values
            self.per_batch_axes[0].plot(self.batch_losses_G_A, label="G_A", color="tab:blue")
            self.per_batch_axes[0].plot(self.batch_losses_G_B, label="G_B", color="tab:orange")

            # Plot discriminator values
            self.per_batch_axes[1].plot(self.batch_losses_D_A, label="D_A", color="tab:blue")
            self.per_batch_axes[1].plot(self.batch_losses_D_B, label="D_B", color="tab:orange")

            # Add legends
            self.per_batch_axes[0].legend(loc="upper right", frameon=True).get_frame()
            self.per_batch_axes[1].legend(loc="upper right", frameon=True).get_frame()

            # Adjust layout and save
            self.per_batch_figure.tight_layout(h_pad=2.0, w_pad=0.0)
            self.per_batch_figure.savefig(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/plots/EP{epoch}__plot.png")

            # Close figure
            plt.close(self.per_batch_figure)
            plt.close("all")

        pass

    def __save_plot_per_epoch(self, epoch) -> None:

        """ Append the current average losses to the arrays containing the network losses """

        self.losses_G_A.append(self.avg_error_G_A.cpu().detach().numpy())
        self.losses_G_B.append(self.avg_error_G_B.cpu().detach().numpy())
        self.losses_D_A.append(self.avg_error_D_A.cpu().detach().numpy())
        self.losses_D_B.append(self.avg_error_D_B.cpu().detach().numpy())

        self.noise_factor_array.append(self.noise_factor)

        """ Plot losses """

        # Create figure
        self.per_epoch_figure, self.per_epoch_axes = plt.subplots(3)

        # Set titles
        self.per_epoch_axes[0].set_title(f"Generator A and Generator B loss during training")
        self.per_epoch_axes[1].set_title(f"Discriminator A and Discriminator B loss training")
        self.per_epoch_axes[2].set_title(f"Decaying noise factor during training")

        # Set labels
        self.per_epoch_axes[0].set(xlabel="Epoch", ylabel="G Loss")
        self.per_epoch_axes[1].set(xlabel="Epoch", ylabel="D Loss")
        self.per_epoch_axes[2].set(xlabel="Epoch", ylabel="Noise factor")

        # Add gridlines
        self.per_epoch_axes[0].grid()
        self.per_epoch_axes[1].grid()
        self.per_epoch_axes[2].grid()

        # Plot generator values
        self.per_epoch_axes[0].plot(self.losses_G_A, label="G_A", color="tab:blue")
        self.per_epoch_axes[0].plot(self.losses_G_B, label="G_B", color="tab:orange")

        # Plot discriminator values
        self.per_epoch_axes[1].plot(self.losses_D_A, label="D_A", color="tab:blue")
        self.per_epoch_axes[1].plot(self.losses_D_B, label="D_B", color="tab:orange")

        # Plot noise factor values
        self.per_epoch_axes[2].plot(self.noise_factor_array, label="Noise factor", color="tab:red")

        # Fill between noise factor values
        self.per_epoch_axes[2].fill_between(
            x=np.arange(epoch + 1), y1=0, y2=self.noise_factor_array, facecolor="tab:red", alpha=0.5
        )

        # Add legends
        self.per_epoch_axes[0].legend(loc="upper right", frameon=True).get_frame()
        self.per_epoch_axes[1].legend(loc="upper right", frameon=True).get_frame()
        self.per_epoch_axes[2].legend(loc="upper right", frameon=True).get_frame()

        # Adjust layout and save
        self.per_epoch_figure.tight_layout(h_pad=2.0, w_pad=0.0)
        self.per_epoch_figure.savefig(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/plot.png")

        # Close figure
        plt.close(self.per_epoch_figure)
        plt.close("all")

        pass


    """ Static methods """

    @staticmethod
    def makedirs(path: str, dir: str):

        if dir == "outputs":
            try:
                os.makedirs(os.path.join(path, "logs"))
                os.makedirs(os.path.join(path, "plots"))
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
    def get_run_path(run, dataset_name: str, channels: int, use_one_directory: bool = False) -> str:

        # Store today's date in string format
        TODAY_DATE = datetime.today().strftime("%Y-%m-%d")
        TODAY_TIME = datetime.today().strftime("%H.%M.%S")

        digits = len(str(run.num_epochs))

        # Create a unique name for this run
        RUN_NAME = f"{TODAY_TIME}___EP{str(run.num_epochs).zfill(digits)}_DE{str(run.decay_epochs).zfill(digits)}_LR{run.learning_rate}_CH{channels}"

        RUN_PATH = f"{dataset_name}/{TODAY_DATE}/{RUN_NAME}"

        if use_one_directory:
            RUN_PATH = f"QUICK_DEV_DIR"

        return RUN_PATH


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


# Execute main code
if __name__ == "__main__":

    # Clear the terminal
    os.system("cls")

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

        # s2d_dataset_train_GRAYSCALE = mydataloader.get_dataset("s2d", "Test_Set_GRAYSCALE", "train", (68, 120), 1, False)
        # s2d_manager_GRAYSCALE = RunCycleManager(s2d_dataset_train_GRAYSCALE, 1, PARAMETERS)
        # s2d_manager_GRAYSCALE.start_cycle()

        """ Train a [S2D] model on the RGB dataset """

        # s2d_dataset_train_RGB = mydataloader.get_dataset("s2d", "Test_Set_RGB_DISPARITY", "train", (68, 120), 3, False)
        # # s2d_dataset_train_RGB = mydataloader.get_dataset("s2d", "Test_Set_RGB_DISPARITY", "train", (40, 60), 3, False)
        # s2d_manager_RGB = RunCycleManager(s2d_dataset_train_RGB, 3, PARAMETERS)
        # s2d_manager_RGB.start_cycle()

        """ _____________ """

        # s2d_dataset_train_RGB = mydataloader.get_dataset("s2d", "Test_Set_RGB_DISPARITY", "train", (68, 120), 1, False)
        # s2d_manager_RGB = RunCycleManager(s2d_dataset_train_RGB, 1, PARAMETERS)
        # s2d_manager_RGB.start_cycle()

        """ _____________ """

        s2d_dataset_train = mydataloader.get_dataset("s2d", "DrivingStereoDemo", "train", (88, 40), 1, False)
        s2d_manager = RunCycleManager(s2d_dataset_train, 1, PARAMETERS)
        s2d_manager.start_cycle()

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
