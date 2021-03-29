import torch.nn as nn

from .ResidualBlock import ResidualBlock


class __Generator(nn.Module):

    """ Insert documentation """

    def __init__(self):

        super().__init__()

        self.main = nn.Sequential(
            #
            # Input layer
            #
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            #
            # Downsampling
            #
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            #
            # Residual blocks
            #
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            #
            # Upsampling
            #
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            #
            # Output layer
            #
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):

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
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(7, 7)),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            #
            # Downsampling
            #
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(inplace=True),
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
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            #
            # Output layer
            #
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=(7, 7)),
            nn.Tanh(),
        )

        self.main_inversed = nn.Sequential(
            #
            # Input layer
            #
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=self.out_channels, out_channels=64, kernel_size=(7, 7)),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            #
            # Downsampling
            #
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(inplace=True),
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
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            #
            # Output layer
            #
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=self.in_channels, kernel_size=(7, 7)),
            nn.Tanh(),
        )

    def forward(self, x):

        """ Insert documentation """

        print(f"- Inversed direction: {x[0].size()[0] == self.in_channels}")

        if x[0].size()[0] == self.in_channels:
            print(f"- Using: in_channels={self.in_channels}; out={self.out_channels}")
            return self.main(x)
        else:
            print(f"- Using: in_channels={self.out_channels}; out={self.in_channels}")
            return self.main_inversed(x)


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
