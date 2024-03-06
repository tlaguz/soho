import torch
from torch.nn import functional as F


#def binary_cross_entropy_with_logits(input, target):
#    pos_weight = torch.tensor([1000.0, 0.0001], device=input.device).view(2, 1, 1)
#    return F.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight)

def binary_cross_entropy_with_logits(input, target):
    class_counts = torch.sum(target, dim=(0, 2, 3))  # Sum over batch and spatial dimensions
    total_count = torch.numel(target) / target.size(1)  # Calculate total number of elements
    pos_weight = (total_count - class_counts) / class_counts  # Calculate pos_weight for each class

    pos_weight = pos_weight.to(input.device).view(-1, 1, 1)  # Use same device as the input tensor and reshape
    return F.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight)


def binary_cross_entropy(input, target):
    return F.binary_cross_entropy(input, target)


def dice_coefficient(input, target):
    smooth = 1.
    iflat = input.view(input.size(0), -1)  # now iflat.shape is [batch, height * width]
    tflat = target.view(target.size(0), -1)  # now tflat.shape is [batch, height * width]
    intersection = (iflat * tflat).sum(1)  # output.shape is [batch]

    return (2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth)


def dice_coefficient_loss(input, target):
    return 1 - dice_coefficient(input, target).mean()


# https://arxiv.org/abs/2209.06078
def combined_loss(logits, target, alpha):
    target = target.to(logits.dtype)

    bce = binary_cross_entropy_with_logits(logits, target)
    dice = dice_coefficient(torch.sigmoid(logits), target).mean()
    return alpha * bce - torch.log(dice)
