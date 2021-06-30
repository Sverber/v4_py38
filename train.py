#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

#%matplotlib inline

import os
import csv
import sys
import copy
import time
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt

import pickle
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
from labml_helpers.module import Module

from utils.classes.DecayLR import DecayLR
from utils.classes.dataloaders import MyDataLoader
from utils.classes.ReplayBuffer import ReplayBuffer
from utils.models.cycle.Generators import Generator
from utils.models.cycle.Discriminator import Discriminator
from utils.tools.yaml_reader import read_yaml
from utils.tools.pytorch_fid.src.pytorch_fid.fid_score import calculate_fid_given_paths

from numpy import cov, disp
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm


class GradientPenalty(Module):
    def __call__(self, x: torch.Tensor, f: torch.Tensor):

        batch_size = x.shape[0]

        gradients, *_ = torch.autograd.grad(outputs=f, inputs=x, grad_outputs=f.new_ones(f.shape), create_graph=True)

        gradients = gradients.reshape(batch_size, -1)

        norm = gradients.norm(2, dim=-1)

        return torch.mean((norm - 1) ** 2)


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


class RunTrainManager:

    """ [ Insert documentation ] """

    def __init__(
        self,
        dataset,
        parameters: OrderedDict,
        models_directory: str,
        use_pretrained_weights: bool,
        dir_dataset: str = "./dataset",
        dir_outputs: str = "./outputs",
        dir_results: str = "./results",
        dir_weights: str = "./weights",
        save_epoch_freq: int = 1,
        show_image_freq: int = 10,
        show_graph_freq: int = 20,
        validation_percentage: float = 0.0,
    ) -> None:

        """ [ Insert documentation ] """

        # Arguments: parameters
        self.parameters = parameters
        self.channels = parameters["channels"][0]

        # Arguments: usage of pre-trained weights
        self.use_pretrained_weights = use_pretrained_weights
        self.models_directory = models_directory

        # Arguments: dataset
        self.dataset = dataset
        self.dataset_group = dataset.dataset_group
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
        self.freq_update_discriminator = 2
        self.start_epoch = 0

        # Build cycle runs for pre-trained weights, if any
        self.runs = self.__init_cycle(parameters, use_pretrained_weights)

        pass

    """ [ Public ] Starts the training loop """

    def start_cycle(self) -> None:

        """ Start training  """

        # Iterate over every run, based on the configurated params
        for run in self.runs:

            # Initialize the networks
            self.__init_networks(run)

            # Declare the other variables for this run
            self.__declare_variables_run(run)

            """ Iterate over the epochs in the run """

            # Iterate through all the epochs
            for epoch in range(self.start_epoch, run.num_epochs):

                """ Declare all per-epoch variables """

                # Declare the other variables for this epoch
                self.__declare_variables_epoch(epoch)

                """ Iterate over training data using the progress bar """

                # Create progress bar
                self.progress_bar = tqdm(enumerate(self.loader), total=len(self.loader))

                # Iterate over the data loader_train
                for i, data in self.progress_bar:

                    # Determine whether this batch is for training or validation
                    self.batch_is_validation = True if i > self.validation_batch_index else False

                    # Read data
                    self.__read_data(i, epoch, run, data)

                    # Make reference image
                    self.__set_reference_data(i, epoch)

                    # Update generator networks
                    self.__update_generators(i)

                    # Add noise to the discriminator input
                    self.__add_discriminator_noise(run, epoch)

                    # Update discriminator networks
                    self.__update_discriminators(i)

                    # Update RMSE loss on generated images
                    self.__update_rmse_loss(i)

                    # Save the real-time output images for every {SHOW_IMG_FREQ} images
                    self.__save_realtime_output(i)

                    # Save per-epoch logs
                    self.__save_per_epoch_logs(epoch)

                    # Save the per-batch losses in a plot
                    self.__save_plot_per_batch(i, epoch)

                    # Print a progress bar in the terminal
                    self.__print_progress(run, i, epoch)

                    pass

                """ Call the end-of-epoch functions """


                # Update FID Score on the generated images for current epoch, once every 5 epochs
                self.__update_fid_score(run, i, 1)

                # Update learning rates after each epoch
                self.__update_learning_rate()

                # Save latest model weights
                self.__save_weights(epoch, False)

                # Save latests meta data
                self.__save_meta_data(run, epoch)

                # Save a snapshot of the generated output using the reference images
                self.__save_reference_snapshot(run, epoch)

                # Save some .csv logs at the end of each epoch
                self.__save_end_epoch_logs(run, epoch)

                # Save some .png plots
                self.__save_plot_per_epoch()

                pass

            """ Save final model """

            # Save final model weights
            self.__save_weights(epoch, True)

            # Save final meta data
            self.__save_meta_data(run, epoch)

            pass

        pass

    """" [ Private ] Build the training cycle according the parameters """

    def __init_cycle(self, parameters, use_pretrained_weights: bool) -> None:

        self.runs = self.__build_cycle(parameters)

        # If no pre-trained weights are used, build cycl
        if use_pretrained_weights == True:

            # Load pickled parameters
            self.parameters = self.load_pickle(os.path.join(self.models_directory, "parameters.pickle"))

            # Load metadata and runs
            self.metadata = self.load_pickle(os.path.join(self.models_directory, "metadata.pickle"))

        else:

            # Define runs
            self.runs = self.__build_cycle(self.parameters)

        return self.runs

    def __init_networks(self, run) -> None:

        """ Initialize the networks and other components/variables """

        # Clear occupied CUDA memory
        torch.cuda.empty_cache()

        # Set a random seed for reproducibility
        random.seed(run.manual_seed)
        torch.manual_seed(run.manual_seed)

        # Create the directory path for this run
        self.RUN_PATH = self.get_run_path(run, self.dataset.name, self.channels)

        # Make required directories for storing the training output
        self.makedirs(path=os.path.join(self.DIR_OUTPUTS, self.RUN_PATH), dir="outputs")

        # Make required directories for storing the model weights
        self.makedirs(path=os.path.join(self.DIR_WEIGHTS, self.RUN_PATH), dir="weights")

        # Create a per-epoch csv log file
        self.__create_per_epoch_csv_logs()

        """ Create network models, use pre-trained weight or new weights """

        if self.use_pretrained_weights == False:

            # Load the un-trained weights into our models
            self.__load_weights_untrained(run)

        else:

            # Load the pre-trained model weights into our models
            self.__load_weights_pretrained(False, self.models_directory)

        pass

    def __build_cycle(self, parameters) -> list:

        """ Create a list of runs using all configured parameter combinations """

        run = namedtuple("Run", parameters.keys())

        self.runs = []

        for v in product(*parameters.values()):
            self.runs.append(run(*v))

        return self.runs

    """ [ Private ] Functions to load the networks and define other components- and variables """

    def __load_weights_untrained(self, run) -> None:

        # Define loss functions
        self.cycle_loss = torch.nn.L1Loss().to(run.device)
        self.identity_loss = torch.nn.L1Loss().to(run.device)
        self.adversarial_loss = torch.nn.MSELoss().to(run.device)

        # Create Generator and Discriminator models # in_channels; # out_channels
        self.net_G_A2B = Generator(in_channels=self.channels, out_channels=self.channels).to(run.device)
        self.net_G_B2A = Generator(in_channels=self.channels, out_channels=self.channels).to(run.device)
        self.net_D_A = Discriminator(in_channels=self.channels, out_channels=self.channels).to(run.device)
        self.net_D_B = Discriminator(in_channels=self.channels, out_channels=self.channels).to(run.device)

        # Apply weights
        self.net_G_A2B.apply(self.__initialize_weights)
        self.net_G_B2A.apply(self.__initialize_weights)
        self.net_D_A.apply(self.__initialize_weights)
        self.net_D_B.apply(self.__initialize_weights)

        # Optimizers
        self.optimizer_G_A2B = torch.optim.Adam(
            self.net_G_A2B.parameters(), lr=run.learning_rate_gen, betas=(0.5, 0.999)
        )
        self.optimizer_G_B2A = torch.optim.Adam(
            self.net_G_B2A.parameters(), lr=run.learning_rate_gen, betas=(0.5, 0.999)
        )
        self.optimizer_D_A = torch.optim.Adam(self.net_D_A.parameters(), lr=run.learning_rate_dis, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.net_D_B.parameters(), lr=run.learning_rate_dis, betas=(0.5, 0.999))

        # Learning rates
        self.lr_lambda = DecayLR(run.num_epochs, self.start_epoch, run.decay_epochs).step
        self.lr_scheduler_G_A2B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G_A2B, lr_lambda=self.lr_lambda)
        self.lr_scheduler_G_B2A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G_B2A, lr_lambda=self.lr_lambda)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=self.lr_lambda)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=self.lr_lambda)

        pass

    def __load_weights_pretrained(self, final_model: bool, models_directory: str) -> None:

        # Read meta data
        self.metadata = self.load_pickle(os.path.join(self.models_directory, "metadata.pickle"))

        # Set starting epoch
        self.start_epoch = self.metadata["start_epoch"]

        # Redefine run
        channels = int(self.metadata["channels"])

        # Use current device
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Decompose filepath into multipe paths for the latest models
        filepath_G_B2A = os.path.join(models_directory, "latest", "net_G_B2A.pth")
        filepath_G_A2B = os.path.join(models_directory, "latest", "net_G_A2B.pth")
        filepath_D_A = os.path.join(models_directory, "latest", "net_D_A.pth")
        filepath_D_B = os.path.join(models_directory, "latest", "net_D_B.pth")

        # Create generator models
        self.net_G_B2A = Generator(channels, channels).to(DEVICE)
        self.net_G_A2B = Generator(channels, channels).to(DEVICE)

        # Create discriminator models
        self.net_D_A = Discriminator(channels, channels).to(DEVICE)
        self.net_D_B = Discriminator(channels, channels).to(DEVICE)

        # Use final model weights instead if requested
        if final_model == True:

            filepath_G_B2A = os.path.join(models_directory, "net_G_B2A", "net_G_B2A.pth")
            filepath_G_A2B = os.path.join(models_directory, "net_G_A2B", "net_G_A2B.pth")
            filepath_D_A = os.path.join(models_directory, "net_D_A", "net_D_A.pth")
            filepath_D_B = os.path.join(models_directory, "net_D_B", "net_D_B.pth")

        # Load state dicts of onto Generators
        self.net_G_B2A.load_state_dict(torch.load(filepath_G_B2A))
        self.net_G_A2B.load_state_dict(torch.load(filepath_G_A2B))

        # Load state dicts of onto Discriminators
        self.net_D_A.load_state_dict(torch.load(filepath_D_A))
        self.net_D_B.load_state_dict(torch.load(filepath_D_B))

        # Set model modes
        self.net_G_B2A.eval()
        self.net_G_A2B.eval()
        self.net_D_A.eval()
        self.net_D_B.eval()

        # Define loss functions
        self.cycle_loss = torch.nn.L1Loss().to(DEVICE)
        self.identity_loss = torch.nn.L1Loss().to(DEVICE)
        self.adversarial_loss = torch.nn.MSELoss().to(DEVICE)

        # Optimizers
        self.optimizer_G_B2A = self.metadata["optimizer_G_B2A"]
        self.optimizer_G_A2B = self.metadata["optimizer_G_A2B"]
        self.optimizer_D_A = self.metadata["optimizer_D_A"]
        self.optimizer_D_B = self.metadata["optimizer_D_B"]

        # Learning rates
        self.lr_lambda = self.metadata["lr_lambda"]
        self.lr_scheduler_G_A2B = self.metadata["lr_scheduler_G_A2B"]
        self.lr_scheduler_G_B2A = self.metadata["lr_scheduler_G_B2A"]
        self.lr_scheduler_D_A = self.metadata["lr_scheduler_D_A"]
        self.lr_scheduler_D_B = self.metadata["lr_scheduler_D_B"]

        pass

    def __declare_variables_run(self, run) -> None:

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
        self.losses_G_A2B, self.losses_G_B2A = [], []
        self.losses_G_A2B_adv, self.losses_G_B2A_adv = [], []
        self.losses_D_A, self.losses_D_B = [], []

        # Keep track of the per-epoch average MSE loss
        self.avg_mse_loss_generated_A_array = []
        self.avg_mse_loss_generated_B_array = []

        # Keep track of the per-epoch average MSE loss
        self.avg_rmse_loss_generated_A_array = []
        self.avg_rmse_loss_generated_B_array = []

        # # Keep track of the per-epoch average MSE loss
        self.avg_fid_score_generated_A_array = []
        self.avg_fid_score_generated_B_array = []

        # Keep track of the FID scores
        self.fid_score_A, self.fid_score_B = 0, 0
        self.fid_score_reference_A, self.fid_score_reference_B = 0, 0

        # Keep track of the per-epoch noise factor
        self.noise_factor_array = []

        pass

    def __declare_variables_epoch(self, epoch: int) -> None:

        # Declare references
        self.reference_image_A, self.reference_image_B = None, None
        self.reference_fake_label, self.reference_real_label = None, None

        # Keep track of the per-batch losses during one epoch
        self.batch_losses_G_A2B, self.batch_losses_G_B2A = [], []
        self.batch_losses_G_A2B_adv, self.batch_losses_G_B2A_adv = [], []
        self.batch_losses_D_A, self.batch_losses_D_B = [], []

        self.batch_losses_error_D_real_A, self.batch_losses_error_D_real_B = [], []
        self.batch_losses_error_D_fake_A, self.batch_losses_error_D_fake_B = [], []

        self.full_batch_losses_G_A, self.full_batch_losses_G_B = [], []

        # Keep track of the MSE losses of the generated images
        self.cum_mse_loss_generated_A, self.cum_mse_loss_generated_B = 0, 0
        self.avg_mse_loss_generated_A, self.avg_mse_loss_generated_B = 0, 0

        # Keep track of the RMSE losses of the generated images
        self.cum_rmse_loss_generated_A, self.cum_rmse_loss_generated_B = 0, 0
        self.avg_rmse_loss_generated_A, self.avg_rmse_loss_generated_B = 0, 0

        # Keep track of the FID scores of the generated images
        self.cum_fid_score_generated_A, self.cum_fid_score_generated_B = 0, 0
        self.avg_fid_score_generated_A, self.avg_fid_score_generated_B = 0, 0

        # Create a per-batch csv log file
        self.__create_per_batch_csv_logs(epoch)

        # Set error variables at 0 at the begin of each epoch
        self.__set_error_variables_to_zero()

        pass

    """ [ Private ] Functions to initialize variables and weights """

    def __initialize_weights(self, m):

        """ Custom weights initialization called on a Generator or Discriminator network """

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
        self.error_G_A2B, self.error_G_B2A = 0, 0

        # Average error on G
        self.cum_error_G_A2B, self.avg_error_G_A2B = 0, 0
        self.cum_error_G_B2A, self.avg_error_G_B2A = 0, 0

        # Average error on G
        self.cum_error_G_A2B_adv_loss, self.avg_error_G_A2B_adv_loss = 0, 0
        self.cum_error_G_B2A_adv_loss, self.avg_error_G_B2A_adv_loss = 0, 0

        """ Variables for error tracking on validation set """

        # Per batch error for D_A, D_B
        self.v__error_D_A, self.v__error_D_B = 0, 0

        # Average error on D_A, D_B
        self.v__cum_error_D_A, self.v__cum_error_D_B = 0, 0
        self.v__avg_error_D_A, self.v__avg_error_D_B = 0, 0

        # Per batch error for G
        self.v__error_G_A2B, self.v__error_G_B2A = 0, 0

        # Average error on G
        self.v__cum_error_G_A2B, self.v__avg_error_G_A2B = 0, 0
        self.v__cum_error_G_B2A, self.v__avg_error_G_B2A = 0, 0

        pass

    """ [ Private ] Functions to create a .csv file """

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
                    "loss_GAN_A2B",
                    "loss_GAN_B2A",
                    "loss_cycle_ABA",
                    "loss_cycle_BAB",
                    "loss_identity_A2B",
                    "loss_identity_B2A",
                ]
            )

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
                    "FID Score A",
                    "FID Score B",
                    "Ref. FID Score A",
                    "Ref. FID Score B",
                    "Ref. RMSE Score A",
                    "Ref. RMSE Score B",
                    "Average MSE loss A",
                    "Average MSE loss B",
                    "Average RMSE loss A",
                    "Average RMSE loss B",
                    "Average FID score A",
                    "Average FID score B",
                    "avg_error_D_A",
                    "avg_error_D_B",
                    "avg_error_G_A",
                    "avg_error_G_B",
                    "cycle_loss_A",
                    "cycle_loss_B",
                    "adv_loss_A",
                    "adv_loss_B",
                    "idt_loss_A",
                    "idt_loss_B",
                ]
            )

        pass

    """ [ Private ] Functions to read- and adjust the training data """

    def __random_flip(self, real_label: torch.Tensor, fake_label: torch.Tensor, probability: float = 0.25):

        """ Randomly flip labels following a given probability """

        random_percentage = random.uniform(0, 1)

        if random_percentage > probability:
            return real_label, fake_label
        else:
            return fake_label, real_label

    def __smooth_one_hot(self, true_label: torch.Tensor, classes: int, smoothing: float = 0.25):

        """ Smoothen one-hot encoced labels y_ls = (1 - α) * y_hot + α / K """

        random_noise = random.uniform(1, 2)

        smooth_label = (1 - smoothing) * true_label + (smoothing * random_noise) / classes

        return smooth_label

    def __read_data(self, i, epoch, run, data) -> None:

        """ Read a left-to-right dataset or a stereo-to-disparity dataset """

        # Get image A and image B from a l2r dataset
        if self.dataset_group == "l2r":
            self.real_image_A = data["left"].to(run.device)
            self.real_image_B = data["right"].to(run.device)

        # Get image A and B from a s2d dataset
        elif self.dataset_group == "s2d":
            real_image_A_left = data["A_left"].to(run.device)
            real_image_A_right = data["A_right"].to(run.device)
            real_image_B = data["B"].to(run.device)

            # Store the images non-locally to access it elsewhere
            self.real_image_A_left = real_image_A_left
            self.real_image_A_right = real_image_A_right
            self.real_image_B = real_image_B

            # Add the left- and right into one image (image A)
            # self.real_image_A = real_image_A_left
            # self.real_image_A = torch.cat((real_image_A_left, real_image_A_right), dim=0)
            self.real_image_A = torch.add(real_image_A_left, real_image_A_right)

            # Normalize again
            if self.channels == 1:
                transforms.Normalize(mean=(0.5), std=(0.5))(self.real_image_A)
            elif self.channels == 3:
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(self.real_image_A)

        else:
            raise Exception(f"Can not read input images, given group '{self.dataset_group}' is incorrect.")

        if epoch == 0 and i == 0:

            self.reference_image_A = copy.deepcopy(self.real_image_A)
            self.reference_image_B = copy.deepcopy(self.real_image_B)


        # Real data label is 1, fake data label is 0.
        self.real_label = torch.full((run.batch_size, self.channels), 1, device=run.device, dtype=torch.float32)
        self.fake_label = torch.full((run.batch_size, self.channels), 0, device=run.device, dtype=torch.float32)

        pass

    def __set_reference_data(self, i: int, epoch: int,) -> None:

        """ Store global reference images for A and B, only for the first epoch and image """

        # Only run this the first batch of the first epoch
        if i == 0 and epoch == 0:

            # Store image tensors
            self.reference_image_A = copy.deepcopy(self.real_image_A)
            self.reference_image_B = copy.deepcopy(self.real_image_B)

            # Store labels
            self.reference_real_label = copy.deepcopy(self.real_label)
            self.reference_fake_label = copy.deepcopy(self.fake_label)

        pass

    def __add_discriminator_noise(self, run, epoch: int) -> None:

        """ Add decaying Gaussian noise to discriminator A and B real/fake inputs """

        # Set mean and standard deviation
        mean, std = 0.5, 0.5

        # Calculate until which epoch there is noise
        self.NOISE_UNTIL_PERCENTAGE = 0.0  # default: 0.80
        self.RANDOM_FLIP_PERCENTAGE = 0.0  # default: 0.10
        self.SMOOTHENING_PERCENTAGE = 0.0  # default: 0.10

        restrictions = [self.NOISE_UNTIL_PERCENTAGE, self.RANDOM_FLIP_PERCENTAGE, self.SMOOTHENING_PERCENTAGE]

        # If all three restrictions are set to 0.0, just copy the images onto the distorted images
        if all(value == 0.0 for value in [restrictions]):

            self.generated_image_B2A_noise = self.generated_image_B2A
            self.generated_image_A2B_noise = self.generated_image_A2B

            self.real_image_A_noise = self.real_image_A
            self.real_image_B_noise = self.real_image_B

            self.real_smooth_label = self.real_label
            self.fake_smooth_label = self.fake_label

        # Else run the restrictions
        else:

            # Noise is gone at epoch x
            noise_until_epoch = self.NOISE_UNTIL_PERCENTAGE * run.num_epochs

            # Calculate noise factor
            if self.NOISE_UNTIL_PERCENTAGE == 0:
                self.noise_factor = 0
            elif epoch > 0 and epoch <= int(round(run.num_epochs * self.NOISE_UNTIL_PERCENTAGE)):
                self.noise_factor = round(1 - (epoch / noise_until_epoch), 3)
            elif epoch > int(round(run.num_epochs * self.NOISE_UNTIL_PERCENTAGE)):
                self.noise_factor = 0
            else:
                self.noise_factor = 1

            # Create the noise for the real images
            noise_real_A = (torch.randn(self.real_image_A.size()) * std + mean).to(run.device)
            noise_real_B = (torch.randn(self.real_image_B.size()) * std + mean).to(run.device)

            # Create the noise for the fake images
            noise_fake_A = (torch.randn(self.generated_image_B2A.size()) * std + mean).to(run.device)
            noise_fake_B = (torch.randn(self.generated_image_A2B.size()) * std + mean).to(run.device)

            # Add decaying noise to the real images (only used by discriminator)
            self.real_image_A_noise = self.real_image_A + (noise_real_A * self.noise_factor)
            self.real_image_B_noise = self.real_image_B + (noise_real_B * self.noise_factor)

            # Add decaying noise to the fake images (only used by discriminator)
            self.generated_image_B2A_noise = self.generated_image_B2A + (noise_fake_A * self.noise_factor)
            self.generated_image_A2B_noise = self.generated_image_A2B + (noise_fake_B * self.noise_factor)

            """ Label smoothing and random flipping """

            # Smoothen the labels (only used by the discriminator)
            self.real_smooth_label = self.__smooth_one_hot(self.real_label, 2, self.SMOOTHENING_PERCENTAGE)
            self.fake_smooth_label = self.__smooth_one_hot(self.fake_label, 2, self.SMOOTHENING_PERCENTAGE)

            # Randomly flip the smooth labels (only used by the discriminator)
            self.real_smooth_label, self.fake_smooth_label = self.__random_flip(
                self.real_smooth_label, self.fake_smooth_label, self.RANDOM_FLIP_PERCENTAGE
            )

        pass

    """ [ Private ] Functions to update the networks """

    def __update_generators(self, i: int) -> None:

        """ Update Generator networks: A2B and B2A """

        # Only zero the gradients when using training data
        if self.batch_is_validation == False:

            # Zero the gradients
            self.optimizer_G_A2B.zero_grad()
            self.optimizer_G_B2A.zero_grad()

        """ Generator losses """

        # GAN loss: D_A(G_B2A(B))
        self.generated_image_B2A = self.net_G_B2A(self.real_image_B)
        self.generated_output_A = self.net_D_A(self.generated_image_B2A)
        self.loss_GAN_B2A = self.adversarial_loss(self.generated_output_A, self.real_label)

        # GAN loss: D_B(G_A2B(A))
        self.generated_image_A2B = self.net_G_A2B(self.real_image_A)
        self.generated_output_B = self.net_D_B(self.generated_image_A2B)
        self.loss_GAN_A2B = self.adversarial_loss(self.generated_output_B, self.real_label)

        """ Identity loss: helps to preserve colour and prevent reverse colour in the result  """

        lambda_A2B = 10  # 10 by default
        lambda_B2A = 10  # 10 by default
        lambda_id = 0.5  # 0.5 by default, set to 0.0 for s2d because colour preservation is not desired

        # Identity loss should not be used for s2d datasets (colour shouldn't be preseved in the result)
        if self.dataset_group == "s2d":

            self.loss_identity_B2A = 0
            self.loss_identity_A2B = 0

        elif lambda_id == 0.0:

            self.loss_identity_B2A = 0
            self.loss_identity_A2B = 0

        else:

            # G_B2A should be fed the real A
            self.identity_image_B2A = self.net_G_B2A(self.real_image_A)
            self.loss_identity_B2A = (
                self.identity_loss(self.identity_image_B2A, self.real_image_A) * lambda_B2A * lambda_id
            )

            # G_A2B should be fed the real B
            self.identity_image_A2B = self.net_G_A2B(self.real_image_B)
            self.loss_identity_A2B = (
                self.identity_loss(self.identity_image_A2B, self.real_image_B) * lambda_A2B * lambda_id
            )

        """ Cycle loss, re-generates the original input image. So: A -> B -> A and B -> A -> B """

        # Apply a gradient penalty to the cycle consistency-loss
        # Check if done, read 3.3 https://ssnl.github.io/better_cycles/report.pdf

        # gradient_penalty = GradientPenalty()

        # self.gradient_penalty_A = gradient_penalty(self.generated_image_B2A, self.generated_output_A)
        # self.gradient_penalty_B = gradient_penalty(self.generated_image_A2B, self.generated_output_B)

        # Cycle loss: A -> B -> A
        self.recovered_image_A = self.net_G_B2A(self.generated_image_A2B)
        self.loss_cycle_ABA = self.cycle_loss(self.recovered_image_A, self.real_image_A) * lambda_A2B

        # Cycle loss: B -> A -> B
        self.recovered_image_B = self.net_G_A2B(self.generated_image_B2A)
        self.loss_cycle_BAB = self.cycle_loss(self.recovered_image_B, self.real_image_B) * lambda_B2A

        """ Calculate the generator errors """

        # Only update weights when using training data
        if self.batch_is_validation == False:

            # Error G_A2B
            self.error_G_A2B = self.loss_GAN_A2B + self.loss_identity_A2B + self.loss_cycle_ABA

            # Error G_B2A
            self.error_G_B2A = self.loss_GAN_B2A + self.loss_identity_B2A + self.loss_cycle_BAB

            # Average error on G_A2B
            self.cum_error_G_A2B += self.error_G_A2B
            self.avg_error_G_A2B = self.cum_error_G_A2B / (i + 1)

            # Average error on G_B2A
            self.cum_error_G_B2A += self.error_G_B2A
            self.avg_error_G_B2A = self.cum_error_G_B2A / (i + 1)

            # Average error on G_A2B (adversarial loss only, so we can compare to the Discriminator)
            self.cum_error_G_A2B_adv_loss += self.loss_GAN_A2B
            self.avg_error_G_A2B_adv_loss = self.cum_error_G_A2B_adv_loss / (i + 1)

            # Average error on G_B2A (adversarial loss only, so we can compare to the Discriminator)
            self.cum_error_G_B2A_adv_loss += self.loss_GAN_B2A
            self.avg_error_G_B2A_adv_loss = self.cum_error_G_B2A_adv_loss / (i + 1)

            # Calculate gradients for G_A and G_B
            self.error_G_A2B.backward()
            self.error_G_B2A.backward()

            # Update the Generator networks
            self.optimizer_G_A2B.step()
            self.optimizer_G_B2A.step()

        pass

    def __update_discriminators(self, i: int) -> None:

        """ Update discriminator A """

        # Only zero the gradient when using training data
        if self.batch_is_validation == False:

            # Set D_A gradients to zero
            self.optimizer_D_A.zero_grad()

        # Real A image loss
        self.real_output_A = self.net_D_A(self.real_image_A_noise)
        self.error_D_real_A = self.adversarial_loss(self.real_output_A, self.real_smooth_label)

        # Fake image A loss
        self.generated_image_B2A_noise = self.fake_A_buffer.push_and_pop(self.generated_image_B2A_noise)
        self.generated_output_A = self.net_D_A(self.generated_image_B2A_noise.detach())
        self.error_D_fake_A = self.adversarial_loss(self.generated_output_A, self.fake_smooth_label)

        # Only update weights when using training data
        if self.batch_is_validation == False:

            # Combined loss and calculate gradients
            self.error_D_A = (self.error_D_real_A + self.error_D_fake_A) / 2

            # Cumulative and average error of D_A
            self.cum_error_D_A += self.error_D_A
            self.avg_error_D_A = self.cum_error_D_A / (i + 1)

            # Calculate gradients for D_A
            self.error_D_A.backward()

            # Update D_A weights
            self.optimizer_D_A.step()

        else:

            # Combined loss and calculate gradients
            self.v__error_D_A = (self.error_D_real_A + self.error_D_fake_A) / 2

            # Cumulative and average validation error of D_A
            self.v__cum_error_D_A += self.v__error_D_A
            self.v__avg_error_D_A = self.v__cum_error_D_A / (i + 1)

        """ Update discriminator B """

        # Only zero the gradient when using training data
        if self.batch_is_validation == False:

            # Set D_B gradients to zero
            self.optimizer_D_B.zero_grad()

        # Real B image loss
        self.real_output_B = self.net_D_B(self.real_image_B_noise)
        self.error_D_real_B = self.adversarial_loss(self.real_output_B, self.real_smooth_label)

        # Fake image B loss
        self.generated_image_A2B_noise = self.fake_B_buffer.push_and_pop(self.generated_image_A2B_noise)
        self.generated_output_B = self.net_D_B(self.generated_image_A2B_noise.detach())
        self.error_D_fake_B = self.adversarial_loss(self.generated_output_B, self.fake_smooth_label)

        # Only update weights when using training data
        if self.batch_is_validation == False:

            # Combined loss and calculate gradients
            self.error_D_B = (self.error_D_real_B + self.error_D_fake_B) / 2

            # Cumulative and average error of D_A
            self.cum_error_D_B += self.error_D_B
            self.avg_error_D_B = self.cum_error_D_B / (i + 1)

            # Calculate gradients for D_B
            self.error_D_B.backward()

            # Update D_B weights
            self.optimizer_D_B.step()

        else:

            # Combined validation loss and calculate gradients
            self.v__error_D_B = (self.error_D_real_B + self.error_D_fake_B) / 2

            # Cumulative and average validation error of D_B
            self.v__cum_error_D_B += self.v__error_D_B
            self.v__avg_error_D_B = self.v__cum_error_D_B / (i + 1)

        pass

    def __update_rmse_loss(self, i: int) -> None:

        """ Calculate a per-batch, cumulative and average (R)MSE loss of the generated outputs of generators A2B and B2A """

        # Initiate a mean square error (MSE) loss function
        mse_loss = nn.MSELoss()

        # Calculate the mean square error (MSE) loss
        mse_loss_generated_A = mse_loss(self.generated_image_B2A.detach(), self.real_image_A)
        mse_loss_generated_B = mse_loss(self.generated_image_A2B.detach(), self.real_image_B)

        # Calculate the sum of all mean square error (MSE) losses
        self.cum_mse_loss_generated_A += mse_loss_generated_A
        self.cum_mse_loss_generated_B += mse_loss_generated_B

        # Calculate the average mean square error (MSE) loss
        self.avg_mse_loss_generated_A = self.cum_mse_loss_generated_A / (i + 1)
        self.avg_mse_loss_generated_B = self.cum_mse_loss_generated_B / (i + 1)

        """ Calculate a per-batch, cumulative and average (R)MSE loss of the generated outputs of generators A2B and B2A """

        # Calculate the root mean square error (RMSE) loss
        rmse_loss_generated_A = np.sqrt(mse_loss_generated_A.cpu())
        rmse_loss_generated_B = np.sqrt(mse_loss_generated_B.cpu())

        # Calculate the sum of all root mean square error (RMSE) losses
        self.cum_rmse_loss_generated_A += rmse_loss_generated_A
        self.cum_rmse_loss_generated_B += rmse_loss_generated_B

        # Calculate the average root mean square error (RMSE) loss
        self.avg_rmse_loss_generated_A = self.cum_rmse_loss_generated_A / (i + 1)
        self.avg_rmse_loss_generated_B = self.cum_rmse_loss_generated_B / (i + 1)

        pass

    def __update_fid_score(self, run, i: int, interval: int) -> None:

        """ Calculate the FID Score """

        if i % interval == 0:

            paths_B2A = [
                f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/features/B2A/fake",
                f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/features/B2A/real",
            ]

            paths_A2B = [
                f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/features/A2B/fake",
                f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/features/A2B/real",
            ]

            self.fid_score_A = calculate_fid_given_paths(paths=paths_B2A, batch_size=run.batch_size, device=run.device)
            self.fid_score_B = calculate_fid_given_paths(paths=paths_A2B, batch_size=run.batch_size, device=run.device)

        pass

    def __update_learning_rate(self) -> None:

        """ Update the learning rate of each network """

        self.lr_scheduler_G_A2B.step()
        self.lr_scheduler_G_B2A.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_D_B.step()

    """ [ Private ] Functions to print-, save- and plot """

    def __print_progress(self, run, i: int, epoch: int) -> None:

        """ Print progress """

        self.progress_bar.set_description(
            #
            f"[{self.dataset_group.upper()}][{epoch + 1}/{run.num_epochs}][{i + 1}/{len(self.loader)}]"
            f"[{self.noise_factor:.2f}|{self.SMOOTHENING_PERCENTAGE:.2f}|{self.RANDOM_FLIP_PERCENTAGE:.2f}]  ||  "
            #
            f"D_A, D_B: {self.avg_error_D_A:.3f} ; {self.avg_error_D_B:.3f} || "
            f"G_A2B, G_B2A: {self.avg_error_G_A2B_adv_loss:.3f} ; {self.avg_error_G_B2A_adv_loss:.3f} || "
            f"FID_A, FID_B [train]: {self.fid_score_A:.3f} ; {self.fid_score_B:.3f} ||  "
            f"FID_A, FID_B [valid]: {self.fid_score_reference_A:.3f} ; {self.fid_score_reference_B:.3f} ||"
            #
        )

        pass

    def __save_realtime_output(self, i: int) -> None:

        """ Save per-epoch all the generated images for the FID score calculation """

        # Write images real-time to feature vector directory
        filepath_fake_A = f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/features/B2A/fake/{i + 1:04d}___fake_A.png"
        filepath_real_A = f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/features/B2A/real/{i + 1:04d}___real_A.png"
        filepath_fake_B = f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/features/A2B/fake/{i + 1:04d}___fake_B.png"
        filepath_real_B = f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/features/A2B/real/{i + 1:04d}___real_B.png"

        # Save all the generated (fake) images
        vutils.save_image(self.generated_image_B2A_noise.detach(), filepath_fake_A, normalize=True)
        vutils.save_image(self.generated_image_A2B_noise.detach(), filepath_fake_B, normalize=True)
        vutils.save_image(self.real_image_A.detach(), filepath_real_A, normalize=True)
        vutils.save_image(self.real_image_B.detach(), filepath_real_B, normalize=True)

        """ Save used data and network outputs, real-time as a png image (useful for visualization during training) """

        if i % self.SHOW_IMAGE_FREQ == 0:

            # Write images real-time to feature vector directory
            rt_filepath_fake_A = f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/B2A___fake_A.png"
            rt_filepath_real_A = f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/B2A___real_A.png"
            rt_filepath_fake_B = f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/A2B___fake_B.png"
            rt_filepath_real_B = f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/realtime/A2B___real_B.png"

            # Save all the generated (fake) images
            vutils.save_image(self.generated_image_A2B_noise.detach(), rt_filepath_fake_B, normalize=True)
            vutils.save_image(self.generated_image_B2A_noise.detach(), rt_filepath_fake_A, normalize=True)
            vutils.save_image(self.real_image_B.detach(), rt_filepath_real_B, normalize=True)
            vutils.save_image(self.real_image_A.detach(), rt_filepath_real_A, normalize=True)

        pass

    def __save_per_epoch_logs(self, epoch: int) -> None:

        """ Stores the current errors and losses as a newline to the previously created .csv file """

        # Create csv for the logs file of this run
        with open(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/logs/EP{epoch}__logs.csv", "a+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Epoch",
                    f"{self.error_D_A:.3f}",
                    f"{self.error_D_B:.3f}",
                    f"{self.error_G_A2B:.3f}",
                    f"{self.error_G_B2A:.3f}",
                    f"{self.loss_GAN_A2B:.3f}",
                    f"{self.loss_GAN_B2A:.3f}",
                    f"{self.loss_cycle_ABA:.3f}",
                    f"{self.loss_cycle_BAB:.3f}",
                    f"{self.loss_identity_A2B:.3f}",
                    f"{self.loss_identity_B2A:.3f}",
                ]
            )

        pass

    def __save_weights(self, epoch: int, done_training: bool) -> None:

        """ Make directories """

        # Make required directories for storing the training output
        self.makedirs(path=os.path.join(self.DIR_WEIGHTS, self.RUN_PATH), dir="weights")

        """ Save network weights """

        # Save final models
        if done_training:

            # Save final check points, after every run
            torch.save(self.net_G_A2B.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_G_A2B/net_G_A2B.pth")
            torch.save(self.net_G_B2A.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_G_B2A/net_G_B2A.pth")
            torch.save(self.net_D_A.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_D_A/net_D_A.pth")
            torch.save(self.net_D_B.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/net_D_B/net_D_B.pth")

        # Save latests models
        else:

            # Save latest check points, after every run
            torch.save(self.net_G_A2B.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/latest/net_G_A2B.pth")
            torch.save(self.net_G_B2A.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/latest/net_G_B2A.pth")
            torch.save(self.net_D_A.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/latest/net_D_A.pth")
            torch.save(self.net_D_B.state_dict(), f"{self.DIR_WEIGHTS}/{self.RUN_PATH}/latest/net_D_B.pth")

        pass

    def __save_meta_data(self, run, epoch: int) -> None:

        """ Save meta data and the parameters as .pickle files in """

        # Save run.parameters as a pickle
        self.save_pickle(self.parameters, "parameters.pickle", f"{self.DIR_WEIGHTS}/{self.RUN_PATH}")

        # Define meta data
        self.metadata = {
            "channels": int(run.channels),
            "start_epoch": int(epoch),
            "learning_rate_gen": float(run.learning_rate_gen),
            "learning_rate_dis": float(run.learning_rate_dis),
            "lr_lambda": self.lr_lambda,
            "lr_scheduler_G_A2B": self.lr_scheduler_G_A2B,
            "lr_scheduler_G_B2A": self.lr_scheduler_G_B2A,
            "lr_scheduler_D_A": self.lr_scheduler_D_A,
            "lr_scheduler_D_B": self.lr_scheduler_D_B,
            "optimizer_G_B2A": self.optimizer_G_B2A,
            "optimizer_G_A2B": self.optimizer_G_A2B,
            "optimizer_D_A": self.optimizer_D_A,
            "optimizer_D_B": self.optimizer_D_B,
        }

        # Save the meta data as a pickle
        self.save_pickle(self.metadata, "metadata.pickle", f"{self.DIR_WEIGHTS}/{self.RUN_PATH}")

        pass

    def __save_reference_snapshot(self, run, epoch: int) -> None:

        """ Save the output generated using the references images """

        # Create output for reference images A and B
        self.reference_A = self.net_G_B2A(self.reference_image_B)
        self.reference_B = self.net_G_A2B(self.reference_image_A)

        """ Calculate RMSE losses """

        # Define RMSE loss
        mse_loss = nn.MSELoss()

        # Calculate MSE
        mse_loss_B2A = mse_loss(self.reference_A.detach(), self.reference_image_A)
        mse_loss_A2B = mse_loss(self.reference_B.detach(), self.reference_image_B)

        # Calculate RMSE
        self.rmse_loss_reference_A = np.sqrt(mse_loss_B2A.cpu())
        self.rmse_loss_reference_B = np.sqrt(mse_loss_A2B.cpu())

        """ Create filepaths for saving """

        # Define the common filepath
        COMMON_PATH = f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/references"

        # Filepaths for generated reference data
        filepath_reference_A_real = f"{COMMON_PATH}/B2A/{0:03d}___A_RMSE0.000.png"
        filepath_reference_B_real = f"{COMMON_PATH}/A2B/{0:03d}___A_RMSE0.000.png"

        # Save original reference data only once
        if epoch == 0:

            # Save the original reference images
            vutils.save_image(self.reference_image_A.detach(), filepath_reference_A_real, normalize=True)
            vutils.save_image(self.reference_image_B.detach(), filepath_reference_B_real, normalize=True)

        # Filepaths for generated reference data
        filepath_reference_A = f"{COMMON_PATH}/B2A/{epoch + 1:03d}___A_RMSE{self.rmse_loss_reference_A:.3f}.png"
        filepath_reference_B = f"{COMMON_PATH}/A2B/{epoch + 1:03d}___B_RMSE{self.rmse_loss_reference_B:.3f}.png"

        # Save generated reference images every epoch
        if epoch % self.SAVE_EPOCH_FREQ == 0:

            # Save generated reference data + ground truth data in one image
            vutils.save_image(self.reference_A.detach(), filepath_reference_A, normalize=True)
            vutils.save_image(self.reference_B.detach(), filepath_reference_B, normalize=True)

        pass

    def __save_end_epoch_logs(self, run, epoch) -> None:

        """ Saves the noise, mse losses and average errors to a newline in the previously created .csv file (at the end of each epoch) """

        with open(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/logs.csv", "a+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    epoch,
                    f"{self.noise_factor:.3f}",
                    f"{self.fid_score_A:.3f}",
                    f"{self.fid_score_B:.3f}",
                    f"{self.fid_score_reference_A:.3f}"
                    f"{self.fid_score_reference_B:.3f}"
                    f"{self.rmse_loss_reference_A:.3f}"
                    f"{self.rmse_loss_reference_B:.3f}"
                    f"{self.avg_mse_loss_generated_A:.3f}",
                    f"{self.avg_mse_loss_generated_B:.3f}",
                    f"{self.avg_rmse_loss_generated_A:.3f}",
                    f"{self.avg_rmse_loss_generated_B:.3f}",
                    f"{self.avg_fid_score_generated_A:.3f}",
                    f"{self.avg_fid_score_generated_B:.3f}",
                    f"{self.avg_error_D_A:.3f}",
                    f"{self.avg_error_D_B:.3f}",
                    f"{self.avg_error_G_A2B:.3f}",
                    f"{self.avg_error_G_B2A:.3f}",
                    f"{self.loss_cycle_ABA:.3f}",
                    f"{self.loss_cycle_BAB:.3f}",
                    f"{self.loss_GAN_B2A:.3f}",
                    f"{self.loss_GAN_A2B:.3f}",
                    f"{self.loss_identity_B2A:.3f}",
                    f"{self.loss_identity_A2B:.3f}",
                ]
            )

        pass

    def __save_plot_per_batch(self, i, epoch) -> None:

        """ Append the current per-batch losses to the arrays containing the network losses """

        # Total losses for all networks
        self.batch_losses_G_A2B.append(self.error_G_A2B.cpu().detach().numpy())
        self.batch_losses_G_B2A.append(self.error_G_B2A.cpu().detach().numpy())
        self.batch_losses_D_A.append(self.error_D_A.cpu().detach().numpy())
        self.batch_losses_D_B.append(self.error_D_B.cpu().detach().numpy())

        # Individual
        self.batch_losses_error_D_real_A.append(self.error_D_real_A.cpu().detach().numpy())
        self.batch_losses_error_D_real_B.append(self.error_D_real_B.cpu().detach().numpy())
        self.batch_losses_error_D_fake_A.append(self.error_D_fake_A.cpu().detach().numpy())
        self.batch_losses_error_D_fake_B.append(self.error_D_fake_B.cpu().detach().numpy())

        # Adversarial losses only of all networks
        self.batch_losses_G_A2B_adv.append(self.loss_GAN_A2B.cpu().detach().numpy())
        self.batch_losses_G_B2A_adv.append(self.loss_GAN_B2A.cpu().detach().numpy())

        """ Plot losses """

        if i % self.SHOW_GRAPH_FREQ == 0:

            # Create figure
            self.per_batch_figure, self.per_batch_axes = plt.subplots(4, 1, figsize=(8, 12))

            # Set titles
            self.per_batch_axes[0].set_title(f"Adversarial loss of generators A2B & B2A (epoch {epoch})")
            self.per_batch_axes[1].set_title(f"Adversarial loss of discriminators A & B (epoch {epoch})")
            self.per_batch_axes[2].set_title(f"Adversarial loss of Generator A2B and Discriminator B (epoch {epoch})")
            self.per_batch_axes[3].set_title(f"Adversarial loss of Generator B2A and Discriminator A (epoch {epoch})")

            # Set labels
            self.per_batch_axes[0].set(xlabel="Batch", ylabel="G total Loss")
            self.per_batch_axes[1].set(xlabel="Batch", ylabel="D total Loss")
            self.per_batch_axes[2].set(xlabel="Batch", ylabel="Adv. loss")
            self.per_batch_axes[3].set(xlabel="Batch", ylabel="Adv. loss")

            # Add gridlines
            self.per_batch_axes[0].grid()
            self.per_batch_axes[1].grid()
            self.per_batch_axes[2].grid()
            self.per_batch_axes[3].grid()

            # Plot generator values
            # self.per_batch_axes[0].plot(self.batch_losses_G_A2B, label="Total loss G_A2B", color="tab:blue")
            # self.per_batch_axes[0].plot(self.batch_losses_G_B2A, label="Total loss G_B2A", color="tab:orange")
            self.per_batch_axes[0].plot(self.batch_losses_G_A2B_adv, label="Adv. loss G_A2B", color="tab:blue")
            self.per_batch_axes[0].plot(self.batch_losses_G_B2A_adv, label="Adv. loss G_B2A", color="tab:orange")

            # Plot discriminator values
            self.per_batch_axes[1].plot(self.batch_losses_D_A, label="D_A", color="tab:blue")
            self.per_batch_axes[1].plot(self.batch_losses_D_B, label="D_B", color="tab:orange")

            # Plot the adversarial losses of all the A2B networks
            self.per_batch_axes[2].plot(self.batch_losses_G_A2B_adv, label="G_A2B", color="tab:blue")
            self.per_batch_axes[2].plot(self.batch_losses_D_B, label="D_B", color="tab:orange")

            # Plot the adversarial losses of all the B2A networks
            self.per_batch_axes[3].plot(self.batch_losses_G_B2A_adv, label="G_B2A", color="tab:blue")
            self.per_batch_axes[3].plot(self.batch_losses_D_A, label="D_A", color="tab:orange")

            # Add legends
            self.per_batch_axes[0].legend(loc="lower left", frameon=True).get_frame()
            self.per_batch_axes[1].legend(loc="lower left", frameon=True).get_frame()
            self.per_batch_axes[2].legend(loc="lower left", frameon=True).get_frame()
            self.per_batch_axes[3].legend(loc="lower left", frameon=True).get_frame()

            # Adjust layout and save
            self.per_batch_figure.tight_layout(h_pad=2.0, w_pad=0.0)
            self.per_batch_figure.savefig(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/plots/EP{epoch}__plot.png")

            # Close figure
            plt.close(self.per_batch_figure)
            plt.close("all")

        pass

    def __save_plot_per_epoch(self) -> None:

        """ Append the current average losses, mse losses and noise factors to the arrays containing the network losses """

        # Generator adversarial losses
        self.losses_G_A2B_adv.append(self.avg_error_G_A2B_adv_loss.cpu().detach().numpy())
        self.losses_G_B2A_adv.append(self.avg_error_G_B2A_adv_loss.cpu().detach().numpy())

        # Generator total losses
        self.losses_G_A2B.append(self.avg_error_G_A2B.cpu().detach().numpy())
        self.losses_G_B2A.append(self.avg_error_G_B2A.cpu().detach().numpy())

        # Discriminator adversarial (total) losses
        self.losses_D_A.append(self.avg_error_D_A.cpu().detach().numpy())
        self.losses_D_B.append(self.avg_error_D_B.cpu().detach().numpy())

        # MSE losses
        self.avg_mse_loss_generated_A_array.append(self.avg_mse_loss_generated_A.cpu().detach().numpy())
        self.avg_mse_loss_generated_B_array.append(self.avg_mse_loss_generated_B.cpu().detach().numpy())

        # RMSE losses
        self.avg_rmse_loss_generated_A_array.append(self.avg_rmse_loss_generated_A.cpu().detach().numpy())
        self.avg_rmse_loss_generated_B_array.append(self.avg_rmse_loss_generated_B.cpu().detach().numpy())

        # FID score
        self.avg_fid_score_generated_A_array.append(self.fid_score_A)
        self.avg_fid_score_generated_B_array.append(self.fid_score_B)

        # Noise factor
        self.noise_factor_array.append(self.noise_factor)

        """ Plot losses """

        # Create figure
        self.per_epoch_figure, self.per_epoch_axes = plt.subplots(6, 1, figsize=(12, 18))

        # self.per_batch_axes[0].set_title(f"Total loss of generator A2B & B2A (epoch {epoch})")
        # Set titles
        self.per_epoch_axes[0].set_title(f"The total loss (adv. + cc.) of Generators A2B & B2A (during training)")
        self.per_epoch_axes[1].set_title(f"The total loss (adv.) of Discriminators A & B (during training)")
        self.per_epoch_axes[2].set_title(f"Decaying noise factor (during training)")
        self.per_epoch_axes[3].set_title(f"Average MSE losses per-epoch of the generated images (during training)")
        self.per_epoch_axes[4].set_title(f"Average RMSE losses per-epoch of the generated images (during training)")
        self.per_epoch_axes[5].set_title(f"Per-epoch FID score of the generated images (during training)")

        # Set labels
        self.per_epoch_axes[0].set(xlabel="Epoch", ylabel="G loss")
        self.per_epoch_axes[1].set(xlabel="Epoch", ylabel="D loss")
        self.per_epoch_axes[2].set(xlabel="Epoch", ylabel="Noise factor")
        self.per_epoch_axes[3].set(xlabel="Epoch", ylabel="MSE")
        self.per_epoch_axes[4].set(xlabel="Epoch", ylabel="RMSE")
        self.per_epoch_axes[5].set(xlabel="Epoch", ylabel="FID")

        # Add gridlines
        self.per_epoch_axes[0].grid()
        self.per_epoch_axes[1].grid()
        self.per_epoch_axes[2].grid()
        self.per_epoch_axes[3].grid()
        self.per_epoch_axes[4].grid()
        self.per_epoch_axes[5].grid()

        # Plot generator values
        # self.per_epoch_axes[0].plot(self.losses_G_A2B, label="Total loss G_A2B", color="tab:blue")
        # self.per_epoch_axes[0].plot(self.losses_G_B2A, label="Total loss G_B2A", color="tab:orange")
        self.per_epoch_axes[0].plot(self.losses_G_A2B_adv, label="Adv. loss G_A2B", color="tab:blue")
        self.per_epoch_axes[0].plot(self.losses_D_B, label="Adv. loss D_B", color="tab:orange")

        # Plot discriminator values
        self.per_epoch_axes[1].plot(self.losses_G_B2A_adv, label="Adv. loss G_B2A", color="tab:blue")
        self.per_epoch_axes[1].plot(self.losses_D_A, label="Adv. loss D_A", color="tab:orange")

        # Plot noise factor values
        self.per_epoch_axes[2].plot(self.noise_factor_array, label="Noise factor", color="tab:gray")

        # Plot MSE loss values
        self.per_epoch_axes[3].plot(self.avg_mse_loss_generated_A_array, label="MSE Loss B2A", color="forestgreen")
        self.per_epoch_axes[3].plot(self.avg_mse_loss_generated_B_array, label="MSE Loss A2B", color="red")

        # Plot RMSE loss values
        self.per_epoch_axes[4].plot(self.avg_rmse_loss_generated_A_array, label="RMSE Loss B2A", color="forestgreen")
        self.per_epoch_axes[4].plot(self.avg_rmse_loss_generated_B_array, label="RMSE Loss A2B", color="red")

        # Plot FID scores values
        self.per_epoch_axes[5].plot(self.avg_fid_score_generated_A_array, label="FID score B2A", color="forestgreen")
        self.per_epoch_axes[5].plot(self.avg_fid_score_generated_B_array, label="FID score A2B", color="red")

        # Add legends
        self.per_epoch_axes[0].legend(loc="lower left", frameon=True).get_frame()
        self.per_epoch_axes[1].legend(loc="lower left", frameon=True).get_frame()
        self.per_epoch_axes[2].legend(loc="lower left", frameon=True).get_frame()
        self.per_epoch_axes[3].legend(loc="lower left", frameon=True).get_frame()
        self.per_epoch_axes[4].legend(loc="lower left", frameon=True).get_frame()
        self.per_epoch_axes[5].legend(loc="lower left", frameon=True).get_frame()

        # Adjust layout and save
        self.per_epoch_figure.tight_layout(h_pad=2.0, w_pad=0.0)
        self.per_epoch_figure.savefig(f"{self.DIR_OUTPUTS}/{self.RUN_PATH}/plot.png")

        # Close figure
        plt.close(self.per_epoch_figure)
        plt.close("all")

        pass

    """ [ Public ] Functions to save and load pickle files """

    def save_pickle(self, data, name: str, filepath_full: str):

        """ Save data as pickle """

        filepath_full = f"{filepath_full}/{name}"

        with open(filepath_full, "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
            outfile.close()

        return None

    def load_pickle(self, filepath_full: str) -> pickle:

        filepath_full = filepath_full

        """ Load data from pickle """

        with open(filepath_full, "rb") as file:
            return pickle.load(file)

    """ [ Static ] Functions to calculate the FID Score, create directories and get the run path """

    @staticmethod
    def calculate_fid(act1, act2):

        """ Calclate FID score for a source- and target image
    
        d^2 = ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2)), where:

            d^2:    FID,
            mu_1:   mu1,
            mu_2:   mu2,
            C_1:    sigma1,
            C_2:    sigma2,
            Tr:     trace,


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

    @staticmethod
    def makedirs(path: str, dir: str):

        """ Create the required directories to store the output data/images """

        if dir == "outputs":
            try:
                os.makedirs(os.path.join(path, "references", "B2A"))
                os.makedirs(os.path.join(path, "references", "A2B"))
                #
                os.makedirs(os.path.join(path, "realtime", "features", "B2A", "fake"))
                os.makedirs(os.path.join(path, "realtime", "features", "B2A", "real"))
                os.makedirs(os.path.join(path, "realtime", "features", "A2B", "fake"))
                os.makedirs(os.path.join(path, "realtime", "features", "A2B", "real"))
                os.makedirs(os.path.join(path, "logs"))
                os.makedirs(os.path.join(path, "plots"))
                # os.makedirs(os.path.join(path, "A"))
                # os.makedirs(os.path.join(path, "B"))
                # os.makedirs(os.path.join(path, "A", "epochs"))
                # os.makedirs(os.path.join(path, "B", "epochs"))
            except OSError:
                pass

        elif dir == "weights":
            try:
                os.makedirs(os.path.join(path, "net_G_A2B"))
                os.makedirs(os.path.join(path, "net_G_B2A"))
                os.makedirs(os.path.join(path, "net_D_A"))
                os.makedirs(os.path.join(path, "net_D_B"))
                os.makedirs(os.path.join(path, "latest"))

            except OSError:
                pass

    @staticmethod
    def get_run_path(run, dataset_name: str, channels: int, use_one_directory: bool = False) -> str:

        """ Create and return the directory path for the passed run """

        # Store today's date in string format
        TODAY_DATE = datetime.today().strftime("%Y-%m-%d")
        TODAY_TIME = datetime.today().strftime("%H.%M.%S")

        # Determine how many digits need to USED
        digits = len(str(run.num_epochs))

        # Create a unique name for this run
        RUN_NAME = f"{TODAY_TIME}___EP{str(run.num_epochs).zfill(digits)}_DE{str(run.decay_epochs).zfill(digits)}_LRG{run.learning_rate_gen}_CH{channels}"

        # Combine the paths to a single path
        RUN_PATH = f"{dataset_name}/{TODAY_DATE}/{RUN_NAME}"

        # Only enable this boolean during development. It overwrite the previous data to avoid deleting a lot of created folders
        if use_one_directory:
            RUN_PATH = f"QUICK_DEV_DIR"

        return RUN_PATH


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAMETERS: OrderedDict = OrderedDict(
    # System configuration and reproducebility
    device=[DEVICE],
    num_workers=[8],
    manual_seed=[999],
    # Dataset
    dataset_group=["s2d"],
    dataset_name=["Test_Set_RGB_DISPARITY"],
    dataset_mode=["train"],
    shuffle=[True],
    # Data dimensions
    batch_size=[1],
    channels=[3],
    # Model learning
    learning_rate_dis=[0.0002],
    learning_rate_gen=[0.0002],
    num_epochs=[200],
    decay_epochs=[100],
    #
)


# Execute main code
if __name__ == "__main__":

    # Clear the terminal
    os.system("cls")

    try:

        # Basic settings
        group = PARAMETERS["dataset_group"][0]
        channels = PARAMETERS["channels"][0]

        # Get dataset
        dataset = MyDataLoader().get_dataset(group, "Test_Set_RGB_DISPARITY", "train", (100, 180), channels, False)

        # Define folder "run path"
        folder_runpath = os.path.join("Test_Set_RGB_DISPARITY", "2021-06-30", "")

        # Define models folder path
        models_directory = os.path.join("weights", "s2d", folder_runpath)

        # Initiate manager
        manager = RunTrainManager(dataset, PARAMETERS, models_directory, False)
        # manager = RunTrainManager(dataset, PARAMETERS, models_directory, True)

        # Start training
        manager.start_cycle()

        """

        # Test_Set_RGB_DISPARITY,       originally (1242, 2208),    train at (68, 120) # (100, 180)
        # DIODE,                        originally (768, 1024),     train at (40, 54)
        # DIML,                         originally (384, 640),      train at (60, 100) # (100, 168)

        # Test_Set_RGB_DISPARITY
        dataset = mydataloader.get_dataset(group, "Test_Set_RGB_DISPARITY", "train", (68, 120), channels, False)
        manager = RunTrainManager(dataset, channels, PARAMET_ERS)
        manager.start_cycle()

        # DIML DATASET
        # dataset = mydataloader.get_dataset("s2d", "DIML", "test_disparity", (64, 106), channels, False)
        # manager = RunTrainManager(dataset, channels, PARAMETERS)
        # manager.start_cycle()

        """

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

