import torch
from torch.nn import functional as F


def binary_cross_entropy_with_logits(input, target):
    return F.binary_cross_entropy_with_logits(input, target)


def dice_coefficient(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def combined_loss(input, target, alpha):
    bce = binary_cross_entropy_with_logits(input, target)
    dice = dice_coefficient(input, target)
    return alpha * bce - torch.log(dice)
