"""
    author: Masahiro Hayashi
    Note: Weighted Cross Entropy Loss is the original loss release with U-net
"""

import torch
from torch.nn import functional as F
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

class Weighted_Cross_Entropy_Loss(nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self):
        super(Weighted_Cross_Entropy_Loss, self).__init__()

    def forward(self, pred, target, weights):
        n, c, H, W = pred.shape
        # # Calculate log probabilities
        logp = F.log_softmax(pred, dim=1)

        # Gather log probabilities with respect to target
        logp = torch.gather(logp, 1, target.view(n, 1, H, W))

        # Multiply with weights
        weighted_logp = (logp * weights).view(n, -1)

        # Rescale so that loss is in approx. same interval
        weighted_loss = weighted_logp.sum(1) / weights.view(n, -1).sum(1)

        # Average over mini-batch
        weighted_loss = -weighted_loss.mean()

        return weighted_loss