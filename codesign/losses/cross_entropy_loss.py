import torch.nn as nn
from torch.nn import CrossEntropyLoss as TorchCrossEntropyLoss

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = TorchCrossEntropyLoss()

    @property
    def name(self):
        return "cross_entropy_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "cls"

    def forward(self, pred, gt):
        return self.loss(pred, gt.long())

class CrossEntropy(CrossEntropyLoss):
    @property
    def name(self):
        return "cross_entropy"