import math
import gin

import torch
import torch.nn as nn
import torch.nn.functional as F


@gin.configurable
class Unet_pruned(nn.Module):
    @staticmethod
    def contracting_block(in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=out_channels,
                out_channels=out_channels,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2),
        )
        return block

    @staticmethod
    def expansive_block(in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=mid_channel,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=mid_channel,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.ConvTranspose2d(
                in_channels=mid_channel,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        return block

    @staticmethod
    def mask_block(in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=mid_channel,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=mid_channel,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.ConvTranspose2d(
                kernel_size=kernel_size,
                in_channels=mid_channel,
                out_channels=out_channels,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Softmax(dim=1),
        )
        return block

    def __init__(self, num_classes=4):
        super().__init__()
        # encoder
        self.conv_encode_1 = self.contracting_block(in_channels=4, out_channels=32)
        self.conv_encode_2 = self.contracting_block(in_channels=32, out_channels=64)
        self.conv_encode_3 = self.contracting_block(in_channels=64, out_channels=128)

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=256, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=128, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # Decode
        self.conv_decode_3 = self.expansive_block(256, 128, 64)
        self.conv_decode_2 = self.expansive_block(128, 64, 32)
        self.mask_layer = self.mask_block(64, 64, num_classes)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block_1 = self.conv_encode_1(x)
        encode_block_2 = self.conv_encode_2(encode_block_1)
        encode_block_3 = self.conv_encode_3(encode_block_2)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_block_3)
        # Decode
        decode_block_3 = self.crop_and_concat(bottleneck1, encode_block_3)
        cat_layer_2 = self.conv_decode_3(decode_block_3)
        decode_block_2 = self.crop_and_concat(cat_layer_2, encode_block_2)
        cat_layer_1 = self.conv_decode_2(decode_block_2)
        decode_block_1 = self.crop_and_concat(cat_layer_1, encode_block_1)
        mask_layer = self.mask_layer(decode_block_1)

        return mask_layer

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
