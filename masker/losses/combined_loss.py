import torch
from torch.nn import functional as F


def binary_cross_entropy_with_logits(input, target):
    return F.binary_cross_entropy_with_logits(input, target)


def binary_cross_entropy(input, target):
    return F.binary_cross_entropy(input, target)


def dice_coefficient(input, target):
    smooth = 1.
    iflat = input.view(input.size(0), -1)  # now iflat.shape is [batch, height * width]
    tflat = target.view(target.size(0), -1)  # now tflat.shape is [batch, height * width]
    intersection = (iflat * tflat).sum(1)  # output.shape is [batch]

    return (2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth)


# https://arxiv.org/abs/2209.06078
def combined_loss(input, target, alpha):
    target = target.to(input.dtype)

    bce = binary_cross_entropy(input, target)
    dice = dice_coefficient(input, target).mean()
    return alpha * bce - torch.log(dice)
