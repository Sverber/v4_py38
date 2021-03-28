import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    """ Insert documentation """

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            #
            # Input layer
            #
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            #
            # Upsampling
            #
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            #
            # Upsampling
            #
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #
            # Upsampling
            #
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            #
            # Output layer
            #
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x


class DiscriminatorGrayScaled(nn.Module):

    """ Insert documentation """

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            #
            # Input layer
            #
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #
            # Upsampling
            #
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #
            # Upsampling
            #
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #
            # Upsampling
            #
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #
            # Output layer
            #
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x

