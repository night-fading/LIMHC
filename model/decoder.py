from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from datasets import trainDataset
from model.encoder import encoder
from model.hrnet import HighResolutionNet
from utils.transforms import replaceImage, distort, correctSubImage, get_max_preds


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )


class decoder(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, out_channels)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits


if __name__ == "__main__":
    img_t, img_cropped_t, position = trainDataset('../../data/LIMHC', '../.cache').__getitem__(3)
    net = encoder()
    img_cropped_t = img_cropped_t.unsqueeze(0)
    x = net.forward(img_cropped_t)
    x = torch.clamp(x, min=0.0, max=1.0)
    # print((x > 1.0).any())
    x = replaceImage(img_t, x, position=position)
    x, pos = distort(x, position)
    # x = perspectiveTransform(x, position)
    print(type(x))
    x_d = x.detach()
    plt.imshow(x_d.permute(1, 2, 0))
    plt.show()
    x = x.unsqueeze(0)
    hrnet = HighResolutionNet()
    y = hrnet(x)
    # print(x.shape)
    preds, maxvals = get_max_preds(y)
    print(preds.to(torch.int8).squeeze(0).tolist())
    x = correctSubImage(x.squeeze(0), preds.to(torch.int8).squeeze(0).tolist())
    x_d = x.detach()
    print(x_d.shape)
    plt.imshow(x_d.permute(1, 2, 0))
    plt.show()
    decoder_net = decoder()
    x = decoder_net(x.unsqueeze(0))
    x_d = x.detach().squeeze(0)
    print(x_d.shape)
    plt.imshow(x_d.permute(1, 2, 0))
    plt.show()