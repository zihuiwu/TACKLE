import torch
import torch.nn as nn
from torch.nn import MSELoss as TorchMSELoss

class MSELoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.loss = TorchMSELoss(reduction=reduction)
    
    @property
    def name(self):
        return "mse_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "recon"

    def forward(self, pred, gt):
        gt = torch.cat([gt, torch.zeros_like(gt)], dim=1)
        return self.loss(pred, gt)

class MSE(MSELoss):
    @property
    def name(self):
        return "mse"

    @property
    def mode(self):
        return "min"
