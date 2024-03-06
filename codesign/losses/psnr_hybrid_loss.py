import torch.nn as nn
from .psnr_loss import PSNRLoss
from .psnr_local_loss import PSNRLocalLoss

class PSNRHybridLoss(nn.Module):
    """
    PSNR (hybrid) module.
    """

    def __init__(self, local_loss_coeff=0.5):
        super().__init__()
        self.psnr_loss = PSNRLoss()
        self.psnr_local_loss = PSNRLocalLoss()
        self.local_loss_coeff = local_loss_coeff

    @property
    def name(self):
        return "psnr_hybrid_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "local_recon"

    def forward(self, Xs, Ys, bbox):
        psnr_loss = self.psnr_loss(Xs, Ys)
        psnr_local_loss = self.psnr_local_loss(Xs, Ys, bbox)
        return (1-self.local_loss_coeff) * psnr_loss + self.local_loss_coeff * psnr_local_loss

class PSNRHybrid(PSNRHybridLoss):
    def forward(self, Xs, Ys, bbox):
        return - super().forward(Xs, Ys, bbox)
    
    @property
    def name(self):
        return "psnr_hybrid"

    @property
    def mode(self):
        return "max"
