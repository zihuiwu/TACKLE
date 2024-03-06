import torch
import torch.nn as nn

class PSNRLocalLoss(nn.Module):
    """
    PSNR (local) module.
    """

    def __init__(self):
        super().__init__()
    
    def _mse_local(self, Xs, Ys, bbox):
        dims = tuple(range(1, len(Xs.shape)))
        return ((Ys * bbox - Xs * bbox) ** 2).sum(dim=dims) / (bbox.sum(dim=dims)+1e-20)

    @property
    def name(self):
        return "psnr_local_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "local_recon"

    def forward(self, Xs, Ys, bbox):
        mse_local = self._mse_local(Xs, Ys, bbox)

        # data_range_local = [(Y*bbox).max() for Y in Ys]
        data_range = [Y.max() for Y in Ys]
        data_range = torch.stack(data_range, dim=0)

        valid_mask = (mse_local != 0.0).detach()
        psnr_local = 10 * torch.log10((data_range[valid_mask] ** 2) / (mse_local[valid_mask]))

        return -torch.mean(psnr_local) 

class PSNRLocal(PSNRLocalLoss):
    def forward(self, Xs, Ys, bbox):
        return - super().forward(Xs, Ys, bbox)
    
    @property
    def name(self):
        return "psnr_local"

    @property
    def mode(self):
        return "max"
