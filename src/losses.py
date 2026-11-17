"""Training losses: BCE + soft Dice + soft-skeleton clDice (Shit et al., CVPR 2021)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss weights used for the released checkpoint.
BCE_WEIGHT = 0.40
DICE_WEIGHT = 0.35
CL_DICE_WEIGHT = 0.25


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 4) -> torch.Tensor:
    img = torch.clamp(img, 0.0, 1.0)
    skel = F.relu(img - soft_open(img))
    for _ in range(iters):
        img = soft_erode(img)
        skel = skel + F.relu(img - soft_open(img))
    return torch.clamp(skel, 0.0, 1.0)


class SoftclDiceLoss(nn.Module):
    def __init__(self, iters: int = 4, smooth: float = 1.0) -> None:
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        skel_pred = soft_skeletonize(y_pred, iters=self.iters)
        skel_true = soft_skeletonize(y_true, iters=self.iters)
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-7)
        return 1.0 - cl_dice


class FilamentLoss(nn.Module):
    """0.40 * BCE + 0.35 * Dice + 0.25 * clDice (3 skeletonization iterations)."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.cldice = SoftclDiceLoss(iters=3)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = torch.sum(probs * targets)
        dice = 1.0 - (2.0 * inter + 1.0) / (torch.sum(probs) + torch.sum(targets) + 1.0)
        cldice = self.cldice(probs, targets)
        return BCE_WEIGHT * bce + DICE_WEIGHT * dice + CL_DICE_WEIGHT * cldice
