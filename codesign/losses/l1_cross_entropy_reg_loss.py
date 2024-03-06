import torch.nn as nn
from torch.nn import L1Loss
from torch.nn import CrossEntropyLoss

class L1CrossEntropyRegLoss(nn.Module):
    """
    L1 & Cross Entropy (hybrid) module.
    """

    def __init__(self, lambda_cross_entropy=0.5):
        super().__init__()
        self.l1_loss = L1Loss()
        self.cross_entropy_loss = CrossEntropyLoss()
        self.lambda_cross_entropy = lambda_cross_entropy

    @property
    def name(self):
        return "l1_cross_entropy_reg_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "seg_reg"

    def forward(self, pred, gt, Xs, Ys):
        l1_loss = self.l1_loss(Xs, Ys)
        cross_entropy_loss = self.cross_entropy_loss(pred, gt.float())
        return l1_loss + self.lambda_cross_entropy * cross_entropy_loss