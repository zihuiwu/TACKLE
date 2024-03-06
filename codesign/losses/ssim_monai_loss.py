import torch.nn as nn
from monai.losses.ssim_loss import SSIMLoss

class SSIMMonaiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_monai_loss = SSIMLoss()

    @property
    def name(self):
        return "ssim_monai_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "recon"

    def forward(self, pred, gt):
        return self.ssim_monai_loss(pred, gt)

class SSIMMonai(SSIMMonaiLoss):
    def forward(self, pred, gt):
        return super().forward(pred, gt)

    @property
    def name(self):
        return "ssim_monai"

    @property
    def mode(self):
        return "max"