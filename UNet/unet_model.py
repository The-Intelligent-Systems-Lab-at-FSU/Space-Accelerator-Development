import os
import numpy as np

import torch
from torch import optim, nn


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 3, 2)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1, (0, 1, 1))
        self.upconv1 = self.expand_block(32*2, out_channels, 2, 0, (0, 1, 1))

        self.out = torch.nn.Softmax(dim=1)

    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # upsampling part
        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        # N C T H W -> N C H W
        out = torch.mean(upconv1, 2)

        return out

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(out_channels, out_channels,
                            kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding, output_padding):

        expand = nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels,
                            kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(out_channels, out_channels,
                            kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3,
                                     stride=2, padding=1, output_padding=output_padding)
        )
        return expand
