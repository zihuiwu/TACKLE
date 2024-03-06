import torch.nn as nn
from .psnr_loss import PSNRLoss
from .psnr_local_loss import PSNRLocalLoss

class PSNREnhanceLoss(nn.Module):
    """
    PSNR (enhance) module.
    """

    def __init__(self, local_loss_coeff=0.5):
        super().__init__()
        self.psnr_loss = PSNRLoss()
        self.psnr_local_loss = PSNRLocalLoss()
        self.local_loss_coeff = local_loss_coeff

    @property
    def name(self):
        return "psnr_enhance_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "local_enhance"

    def forward(self, xs, Xs, Ys, bbox):
        psnr_loss = self.psnr_loss(xs, Ys)
        psnr_local_loss = self.psnr_local_loss(Xs, Ys, bbox)
        return (1-self.local_loss_coeff) * psnr_loss + self.local_loss_coeff * psnr_local_loss

class PSNREnhance(PSNREnhanceLoss):
    def forward(self, xs, Xs, Ys, bbox):
        return - super().forward(xs, Xs, Ys, bbox)
    
    @property
    def name(self):
        return "psnr_enhance"

    @property
    def mode(self):
        return "max"
