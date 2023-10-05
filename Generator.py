import torch
from torch import nn
import utils


class ResNetGenerator(nn.Module):
    def __init__(self, hid_channels, in_channels, out_channels, num_resblocks):
        super().__init__()
        self.model = nn.Sequential(
            # downsampling path
            nn.ReflectionPad2d(3),
            utils.Downsampling(
                in_channels,
                hid_channels,
                kernel_size=7,
                stride=1,
                padding=0,
                lrelu=False,
            ),  # 64x256x256
            utils.Downsampling(
                hid_channels, hid_channels * 2, kernel_size=3, lrelu=False
            ),  # 128x128x128
            utils.Downsampling(
                hid_channels * 2, hid_channels * 4, kernel_size=3, lrelu=False
            ),  # 256x64x64
            # residual blocks
            *[
                utils.ResBlock(hid_channels * 4) for _ in range(num_resblocks)
            ],  # 256x64x64
            # upsampling path
            utils.Upsampling(
                hid_channels * 4, hid_channels * 2, kernel_size=3, output_padding=1
            ),  # 128x128x128
            utils.Upsampling(
                hid_channels * 2, hid_channels, kernel_size=3, output_padding=1
            ),  # 64x256x256
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                hid_channels, out_channels, kernel_size=7, stride=1, padding=0
            ),  # 3x256x256
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class UNetGenerator(nn.Module):
    def __init__(self, hid_channels, in_channels, out_channels):
        super().__init__()
        self.downsampling_path = nn.Sequential(
            utils.Downsampling(in_channels, hid_channels, norm=False),  # 64x128x128
            utils.Downsampling(hid_channels, hid_channels * 2),  # 128x64x64
            utils.Downsampling(hid_channels * 2, hid_channels * 4),  # 256x32x32
            utils.Downsampling(hid_channels * 4, hid_channels * 8),  # 512x16x16
            utils.Downsampling(hid_channels * 8, hid_channels * 8),  # 512x8x8
            utils.Downsampling(hid_channels * 8, hid_channels * 8),  # 512x4x4
            utils.Downsampling(hid_channels * 8, hid_channels * 8),  # 512x2x2
            utils.Downsampling(
                hid_channels * 8, hid_channels * 8, norm=False
            ),  # 512x1x1, instance norm does not work on 1x1
        )
        self.upsampling_path = nn.Sequential(
            utils.Upsampling(
                hid_channels * 8, hid_channels * 8, dropout=True
            ),  # (512+512)x2x2
            utils.Upsampling(
                hid_channels * 16, hid_channels * 8, dropout=True
            ),  # (512+512)x4x4
            utils.Upsampling(
                hid_channels * 16, hid_channels * 8, dropout=True
            ),  # (512+512)x8x8
            utils.Upsampling(hid_channels * 16, hid_channels * 8),  # (512+512)x16x16
            utils.Upsampling(hid_channels * 16, hid_channels * 4),  # (256+256)x32x32
            utils.Upsampling(hid_channels * 8, hid_channels * 2),  # (128+128)x64x64
            utils.Upsampling(hid_channels * 4, hid_channels),  # (64+64)x128x128
        )
        self.feature_block = nn.Sequential(
            nn.ConvTranspose2d(
                hid_channels * 2, out_channels, kernel_size=4, stride=2, padding=1
            ),  # 3x256x256
            nn.Tanh(),
        )

    def forward(self, x):
        skips = []
        for down in self.downsampling_path:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for up, skip in zip(self.upsampling_path, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        return self.feature_block(x)


def get_gen(gen_name, hid_channels, num_resblocks, in_channels=3, out_channels=3):
    if gen_name == "unet":
        return UNetGenerator(hid_channels, in_channels, out_channels)
    elif gen_name == "resnet":
        return ResNetGenerator(hid_channels, in_channels, out_channels, num_resblocks)
    else:
        raise NotImplementedError(f"Generator name '{gen_name}' not recognized.")
