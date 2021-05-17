import torch.nn as nn

from .ResidualBlock import ResidualBlock


class Generator(nn.Module):

    """ Generator network of the GAN """

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dropout2d = 0.5

        """

        Can solve checkerboard artifact issue by switching from ConvTranspose2d to nearest-neighbor upsampling
        
        from:

        nn.ConvTranspose2d(
            in_channels=128, 
            out_channels=64, 
            kernel_size=(3, 3), 
            stride=2, 
            padding=1, 
            output_padding=1
        ),

        to: 

        nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0),

        """

        self.main = nn.Sequential(
            #
            # Input layer
            #
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(7, 7), stride=1, padding=0),
            nn.InstanceNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            # Downsampling
            #
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
            # nn.ConvTranspose2d(
            #     in_channels=256, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
            # ),
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),                     
            nn.ReflectionPad2d(1),                                              
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),            
            nn.InstanceNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.dropout2d),
            #
            # Upsampling
            #
            # nn.ConvTranspose2d(
            #     in_channels=128, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1
            # ),
            #
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True), 
            nn.ReflectionPad2d(1),                                              
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),             
            nn.InstanceNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
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

    """ Original Generator network that is used in the CycleGAN """

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