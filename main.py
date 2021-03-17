#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
import sys
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from synthesis.Synthesis import Synthesis

from utils.functions.initialize_weights import initialize_weights

from utils.classes.DecayLR import DecayLR
from utils.classes.ReplayBuffer import ReplayBuffer
from utils.classes.RunCycleBuilder import RunCycleBuilder
from utils.classes.RunCycleManager import RunCycleManager
from utils.classes.DisparityDataset import DisparityDataset

from utils.models.cycle.Discriminator import Discriminator
from utils.models.cycle.Generators import Generator, OneToOneGenerator, OneToMultiGenerator, MultiToOneGenerator


# Clear terminal
# os.system("cls")
# print("Hello there!")


# Constants: required directories
DIR_DATASET = f"./dataset"
DIR_OUTPUTS = f"./outputs"
DIR_RESULTS = f"./results"
DIR_WEIGHTS = f"./weights"


# Constants: dataset name
# NAME_DATASET = f"horse2zebra_000_999"
# NAME_DATASET = f"kitti_synthesized_000_999"
# NAME_DATASET = f"stereo_test"
NAME_DATASET = f"DrivingStereo_demo_images"


# Constants: system
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Constants: parameters
# IMAGE_SIZE = (122, 35) # kitti_synthesized_000_999
IMAGE_SIZE = (176, 69) # DrivingStereo_demo_images
RATIO_CROP = 0.82
RANDM_CROP = (int(IMAGE_SIZE[0] * RATIO_CROP), int(IMAGE_SIZE[1] * RATIO_CROP))
PRINT_FREQ = 5


# Configure network parameters
PARAMETERS: OrderedDict = OrderedDict(
    device=[DEVICE],
    shuffle=[True],
    num_workers=[4],
    learning_rate=[0.0002],
    batch_size=[1],
    num_epochs=[2],
    decay_epochs=[1],
)


# Transformations on the datasets
TRANSFORMATIONS: transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE, Image.BICUBIC),
        transforms.RandomCrop(RANDM_CROP),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# Load dataset
def load_dataset(mode: str = "train", verbose: bool = True) -> DisparityDataset:

    # Print start message if verbose is set to True
    verbose is True if print(f"Gathering the {mode} dataset.") else None

    # Gather dataset
    dataset: DisparityDataset = DisparityDataset(
        root=f"./{DIR_DATASET}/{NAME_DATASET}", mode=mode, transform=TRANSFORMATIONS
    )

    # Print completion message if verbose is set to True
    verbose is True if print(f"Loaded train data, length: {len(dataset)}.") else None

    return dataset


# Testing function
def test(dataset: DisparityDataset, path_to_folder: str, model_netG_A2B: str, model_netG_B2A: str) -> None:

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

        # create model
        netG_A2B = MultiToOneGenerator().to(run.device)
        netG_B2A = OneToMultiGenerator().to(run.device)

        # Load state dicts
        netG_A2B.load_state_dict(torch.load(os.path.join(str(path_to_folder), model_netG_A2B)))
        netG_B2A.load_state_dict(torch.load(os.path.join(str(path_to_folder), model_netG_B2A)))

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
            vutils.save_image(fake_image_A.detach(), f"{DIR_RESULTS}/{RUN_PATH}/A/{i + 1:04d}.png", normalize=True)
            vutils.save_image(fake_image_B.detach(), f"{DIR_RESULTS}/{RUN_PATH}/B/{i + 1:04d}.png", normalize=True)

            # Print a progress bar in terminal
            progress_bar.set_description(f"Process images {i + 1} of {len(loader)}")

    # </end> def test():
    pass


# Training function
def train(dataset: DisparityDataset) -> None:

    """ Insert documentation """

    print(f"Start training")

    # Iterate over every run, based on the configurated params
    for run in RunCycleBuilder.get_runs(PARAMETERS):

        # Store today's date in string format
        TODAY_DATE = datetime.today().strftime("%Y-%m-%d")
        TODAY_TIME = datetime.today().strftime("%H.%M.%S")

        # Create a unique name for this run
        RUN_NAME = f"{TODAY_TIME}___EP{run.num_epochs}_DE{run.decay_epochs}_LR{run.learning_rate}_BS{run.batch_size}"
        RUN_PATH = f"{NAME_DATASET}/{TODAY_DATE}/{RUN_NAME}"

        """ Insert a save params function as .txt file / incorporate in RunCycleManager """

        # Make required directories for storing the training output and model weights
        try:
            os.makedirs(os.path.join(DIR_OUTPUTS, RUN_PATH, "A"))
            os.makedirs(os.path.join(DIR_OUTPUTS, RUN_PATH, "B"))
            os.makedirs(os.path.join(DIR_OUTPUTS, RUN_PATH, "A", "epochs"))
            os.makedirs(os.path.join(DIR_OUTPUTS, RUN_PATH, "B", "epochs"))
            os.makedirs(os.path.join(DIR_WEIGHTS, RUN_PATH))
        except OSError:
            pass

        # Create Generator and Discriminator models
        netG_A2B = OneToMultiGenerator().to(run.device)
        netG_B2A = MultiToOneGenerator().to(run.device)
        netD_A = Discriminator().to(run.device)
        netD_B = Discriminator().to(run.device)

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
        manager.begin_run(run, run.device, netG_A2B, netG_B2A, netD_A, netD_B, loader)

        # Iterate through all the epochs
        for epoch in range(0, run.num_epochs):

            # Track the start of the epoch
            manager.begin_epoch()

            # Create progress bar
            progress_bar = tqdm(enumerate(loader), total=len(loader))

            # Iterate over the data loader
            for i, data in progress_bar:

                try:

                    # Get image A and image B
                    real_image_A_left = data["A_left"].to(run.device)
                    real_image_A_right = data["A_right"].to(run.device)
                    real_image_B = data["B"].to(run.device)

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
                        # L_D(A) = Loss discriminator A
                        # L_D(B) = Loss discriminator B
                        # L_G(A2B) = Loss generator A2B
                        # L_G(B2A) = Loss generator B2A
                        # L_G_ID = Combined oentity loss generators A2B + B2A
                        # L_G_GAN = Combined GAN loss generators A2B + B2A
                        # L_G_CYCLE = Combined cycle consistency loss Generators A2B + B2A
                    """

                    # Print a progress bar in terminal
                    progress_bar.set_description(
                        f"[{epoch + 1}/{run.num_epochs}][{i + 1}/{len(loader)}] "
                        f"L_D(A): {error_D_A.item():.3f} "
                        f"L_D(B): {error_D_B.item():.3f} | "
                        #
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
                            real_image_A, f"{DIR_OUTPUTS}/{RUN_PATH}/A/real_samples.png", normalize=True,
                        )
                        vutils.save_image(
                            real_image_B, f"{DIR_OUTPUTS}/{RUN_PATH}/B/real_samples.png", normalize=True,
                        )

                        fake_image_A = 0.5 * (netG_B2A(real_image_B).data + 1.0)
                        fake_image_B = 0.5 * (netG_A2B(real_image_A).data + 1.0)

                        # Save the real-time fake images A & B to a .png
                        vutils.save_image(
                            fake_image_A.detach(), f"{DIR_OUTPUTS}/{RUN_PATH}/A/fake_samples.png", normalize=True,
                        )
                        vutils.save_image(
                            fake_image_B.detach(), f"{DIR_OUTPUTS}/{RUN_PATH}/B/fake_samples.png", normalize=True,
                        )

                        # Save the per-epoch fake images A & B to a .png
                        vutils.save_image(
                            fake_image_A.detach(),
                            f"{DIR_OUTPUTS}/{RUN_PATH}/A/epochs/fake_samples_epoch_{epoch}.png",
                            normalize=True,
                        )
                        vutils.save_image(
                            fake_image_B.detach(),
                            f"{DIR_OUTPUTS}/{RUN_PATH}/B/epochs/fake_samples_epoch_{epoch}.png",
                            normalize=True,
                        )

                        # # Flatten the 4D tensor to a 1D numpy array
                        # np_fake_image_A = real_image_A.reshape(1, -1).squeeze().cpu().numpy()
                        # np_fake_image_B = real_image_B.reshape(1, -1).squeeze().cpu().numpy()

                        # # Save the real-time fake image tensor to a .csv for numerical-based debugging
                        # np.savetxt(f"{DIR_OUTPUTS}/{RUN_PATH}/A/fake_samples.csv", np_fake_image_A, delimiter=",")
                        # np.savetxt(f"{DIR_OUTPUTS}/{RUN_PATH}/B/fake_samples.csv", np_fake_image_B, delimiter=",")

                except Exception as e:
                    print(e)
                    pass

                # </end> for i, data in progress_bar:

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
            manager.end_epoch()

            # </end> for epoch in range(0, run.num_epochs):

        # Save last check points, after every run
        torch.save(netG_A2B.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netG_A2B.pth")
        torch.save(netG_B2A.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netG_B2A.pth")
        torch.save(netD_A.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netD_A.pth")
        torch.save(netD_B.state_dict(), f"{DIR_WEIGHTS}/{RUN_PATH}/netD_B.pth")

        # Track the end of the run
        manager.end_run()

        # </end> for run in RunCycleBuilder.get_runs(params):

    # </end> def train():


# Execute main code
if __name__ == "__main__":

    try:

        # syn = Synthesis(mode="test")
        # syn.predict_depth()

        dataset_train = load_dataset(mode="train", verbose=True)
        train(dataset=dataset_train)

        # dataset_test = load_dataset(mode="test", verbose=True)

        # test(
        #     dataset=dataset_test,
        #     path_to_folder=f"{DIR_WEIGHTS}/kitti_synthesized_000_999/2021-03-11/15.17.59___EP100_DE50_LR0.0002_BS6",
        #     model_netG_A2B=f"netG_A2B_epoch_37.pth",
        #     model_netG_B2A=f"netG_B2A_epoch_37.pth",
        # )

        pass

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
