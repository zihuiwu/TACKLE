import torch.nn as nn
from torch.nn import L1Loss as TorchL1Loss

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = TorchL1Loss()
    
    @property
    def name(self):
        return "l1_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "recon"

    def forward(self, pred, gt):
        return self.loss(pred, gt)

class L1(L1Loss):
    @property
    def name(self):
        return "l1"

    @property
    def mode(self):
        return "min"
