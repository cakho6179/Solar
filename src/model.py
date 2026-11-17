"""Model definition and input representation.

The architecture and the 3-channel preprocessing here are *identical* to the ones
used to train ``models/best_model.pth`` (see ``train.py``). Parameter names are
kept stable (``stem``, ``enc1..4``, ``up*``, ``dec*``, ``head``) so the
checkpoint loads with ``strict=True``.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Sequential):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class FilamentResNet34UNet(nn.Module):
    """ResNet-34 encoder + transposed-convolution U-Net decoder.

    Input : (B, 3, H, W) float tensor in [0, 1] (see ``make_3channel_image``),
            H and W divisible by 32.
    Output: (B, 1, H, W) logits.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        enc = models.resnet34(weights=weights)

        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.enc1 = enc.layer1  # 64 ch,  1/4
        self.enc2 = enc.layer2  # 128 ch, 1/8
        self.enc3 = enc.layer3  # 256 ch, 1/16
        self.enc4 = enc.layer4  # 512 ch, 1/32

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = ConvBlock(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec0 = ConvBlock(32, 32)

        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)  # 64, 1/2
        x1 = self.enc1(self.pool(x0))  # 64, 1/4
        x2 = self.enc2(x1)  # 128, 1/8
        x3 = self.enc3(x2)  # 256, 1/16
        x4 = self.enc4(x3)  # 512, 1/32

        d4 = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x0], dim=1))
        d0 = self.dec0(self.up0(d1))
        return self.head(d0)


def make_3channel_image(img_gray: np.ndarray) -> np.ndarray:
    """Build the 3-channel float32 input (H, W, 3) in [0, 1] from a uint8 grayscale image.

    ch0: grayscale H-alpha intensity / 255
    ch1: CLAHE (clip limit 2.0, 8x8 tiles) / 255
    ch2: Sobel gradient magnitude, divided by its per-image maximum
    """
    ch0 = img_gray.astype(np.float32) / 255.0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch1 = clahe.apply(img_gray).astype(np.float32) / 255.0

    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    g_max = grad.max()
    ch2 = grad / (g_max + 1e-6) if g_max > 0 else grad

    return np.stack([ch0, ch1, ch2], axis=2)
