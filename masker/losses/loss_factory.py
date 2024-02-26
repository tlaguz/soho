import torch
from torch import nn
from torch.nn import functional as F

from masker.losses.combined_loss import combined_loss


def create_loss():
    #loss = F.binary_cross_entropy_with_logits
    loss = lambda input, target: combined_loss(input, target, 0.5)
    return loss
