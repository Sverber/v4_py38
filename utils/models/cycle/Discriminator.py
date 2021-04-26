import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):

    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(
        self, sigma=0.1, is_relative_detach=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x

class Discriminator(nn.Module):

    """ Insert documentation """

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = (4, 4)  # originally (4, 4)

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
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            # Output layer
            #
            nn.Conv2d(
                in_channels=512, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=2, padding=1
            ),
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

