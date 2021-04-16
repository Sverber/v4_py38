import torch.nn as nn

from .ResidualBlock import ResidualBlock


class Generator(nn.Module):

    """ Insert documentation """

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dropout2d = 0.5

        self.main = nn.Sequential(
            #
            # Input layer
            #
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(7, 7), stride=1, padding=0),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            # Downsampling
            #
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            # Residual blocks
            #
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            #
            # Upsampling
            #
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            # Upsampling
            #
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            # Output layer
            #
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=(7, 7), stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):

        """ Insert documentation """

        return self.main(x)


class __Generator(nn.Module):

    """ Insert documentation """

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.main = nn.Sequential(
            #
            # Input layer
            #
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(7, 7), stride=1, padding=0),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            #
            # Downsampling
            #
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            #
            # Residual blocks
            #
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            ResidualBlock(in_channels=256),
            #
            # Upsampling
            #
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            #
            # Upsampling
            #
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            #
            # Output layer
            #
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=(7, 7), stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):

        """ Insert documentation """

        return self.main(x)


# class OneToMultiGenerator(nn.Module):

#     """ Insert documentation """

#     def __init__(self):

#         super(OneToMultiGenerator, self).__init__()

#         self.main = nn.Sequential(
#             #
#             # Input layer
#             #
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=self._in_channels, out_channels=64, kernel_size=(7, 7)),
#             nn.InstanceNorm2d(num_features=64),
#             nn.ReLU(inplace=True),
#             #
#             # Downsampling
#             #
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(num_features=128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 256, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(num_features=256),
#             nn.ReLU(inplace=True),
#             #
#             # Residual blocks
#             #
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             #
#             # Upsampling
#             #
#             nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(num_features=128),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(num_features=64),
#             nn.ReLU(inplace=True),
#             #
#             # Output layer
#             #
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(64, 3, 7),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         return self.main(x)


# class MultiToOneGenerator(nn.Module):

#     """ Insert documentation """

#     def __init__(self):

#         super(MultiToOneGenerator, self).__init__()

#         self.main = nn.Sequential(
#             #
#             # Input layer
#             #
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(3, 64, 7),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#             #
#             # Downsampling
#             #
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 256, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(inplace=True),
#             #
#             # Residual blocks
#             #
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             #
#             # Upsampling
#             #
#             nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#             #
#             # Output layer
#             #
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(64, 3, 7),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         return self.main(x)


# class OneToMultiGenerator_1C_TO_3C(nn.Module):

#     """ Insert documentation """

#     def __init__(self, in_channels: int = 3, out_channels: int = 3):

#         super(OneToMultiGenerator_1C_TO_3C, self).__init__()

#         self._in_channels = in_channels
#         self._out_channels = out_channels

#         self.main = nn.Sequential(
#             #
#             # Input layer
#             #
#             nn.ReflectionPad2d(padding=3),
#             nn.Conv2d(in_channels=self._in_channels, out_channels=64, kernel_size=(7, 7)),
#             nn.InstanceNorm2d(num_features=64),
#             nn.ReLU(inplace=True),
#             #
#             # Downsampling
#             #
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(num_features=128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
#             nn.InstanceNorm2d(num_features=256),
#             nn.ReLU(inplace=True),
#             #
#             # Residual blocks
#             #
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             ResidualBlock(in_channels=256),
#             #
#             # Upsampling
#             #
#             nn.ConvTranspose2d(
#                 in_channels=256, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
#             ),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(
#                 in_channels=128, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
#             ),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#             #
#             # Output layer
#             #
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=64, out_channels=self._out_channels, kernel_size=(7, 7)),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         return self.main(x)


# class MultiToOneGenerator_3C_TO_1C(nn.Module):

#     """ Insert documentation """

#     def __init__(self, in_channels: int = 3, out_channels: int = 3):

#         super(MultiToOneGenerator_3C_TO_1C, self).__init__()

#         self._in_channels = in_channels
#         self._out_channels = out_channels

#         self.main = nn.Sequential(
#             #
#             # Input layer
#             #
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(self._in_channels, 64, 7),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#             #
#             # Downsampling
#             #
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 256, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(inplace=True),
#             #
#             # Residual blocks
#             #
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             ResidualBlock(256),
#             #
#             # Upsampling
#             #
#             nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#             #
#             # Output layer
#             #
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(64, self._out_channels, 7),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         return self.main(x)
