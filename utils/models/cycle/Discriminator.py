import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    """ Insert documentation """

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = (6, 6)  # originally (4, 4)

        self.dropout2d = 0.5

        self.main = nn.Sequential(
            #
            # Input layer
            #
            nn.Conv2d(
                in_channels=self.in_channels, out_channels=64, kernel_size=self.kernel_size, stride=2, padding=1
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            # Upsampling
            #
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            # Upsampling
            #
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            # Upsampling
            #
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            # Output layer
            #
            nn.Conv2d(in_channels=512, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=1),
        )

    def forward(self, x):

        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x


class __Discriminator(nn.Module):

    """ Insert documentation """

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = (4, 4)  # originally (4, 4)

        self.main = nn.Sequential(
            #
            # Input layer
            #
            nn.Conv2d(
                in_channels=self.in_channels, out_channels=64, kernel_size=self.kernel_size, stride=2, padding=1
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.5),
            #
            # Upsampling
            #
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.5),
            #
            # Upsampling
            #
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.5),
            #
            # Upsampling
            #
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.5),
            #
            # Output layer
            #
            nn.Conv2d(in_channels=512, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=1),
        )

    def forward(self, x):

        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x

