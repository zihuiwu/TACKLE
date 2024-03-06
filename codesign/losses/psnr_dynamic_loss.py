import torch.nn as nn
from .psnr_loss import PSNRLoss
from .psnr_local_loss import PSNRLocalLoss

class PSNRDynamicHybridLoss(nn.Module):
    """
    PSNR (dynamic hybrid) module.
    """

    def __init__(self):
        super().__init__()
        self.psnr_loss = PSNRLoss()
        self.psnr_local_loss = PSNRLocalLoss()

    @property
    def name(self):
        return "psnr_dynamic_hybrid_loss"

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

class PSNRDynamicHybrid(PSNRDynamicHybridLoss):
    def forward(self, Xs, Ys, bbox):
        return - super().forward(Xs, Ys, bbox)
    
    @property
    def name(self):
        return "psnr_dynamic_hybrid"

    @property
    def mode(self):
        return "max"
