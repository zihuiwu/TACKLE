from torch import nn 
from monai.losses import DiceLoss
from .psnr_loss import PSNRLoss

class DicePSNRRegLoss(nn.Module):
    def __init__(self, lambda_psnr):
        super().__init__()
        self.dice_monai_loss = DiceLoss(include_background=False, softmax=True)
        self.psnr_loss = PSNRLoss()
        self.lambda_psnr = lambda_psnr
    
    @property
    def name(self):
        return "dice_psnr_reg_loss"

    @property
    def mode(self):
        return "min"
    
    @property
    def task(self):
        return "seg_reg"

    def forward(self, pred, gt, Xs, Ys):
        return self.dice_monai_loss(pred, gt) + self.lambda_psnr * self.psnr_loss(Xs, Ys)
