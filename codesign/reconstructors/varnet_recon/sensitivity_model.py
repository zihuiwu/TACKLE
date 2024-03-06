import torch
import torch.nn as nn
from .unet import NormUnet
from codesign.samplers.modules import preselect
from typing import Tuple
from ...utils.multicoil_ops import (
    _rss_complex,
    _ifft2c
)

class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.
    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int = 8,
        num_pools: int = 4,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / _rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        acs_ratio: float = 0.0,
    ) -> torch.Tensor:
        if self.mask_center:
            masked_kspace = masked_kspace * preselect(
                mask * 0,
                dim=[-2, -1],
                preselect_num=0,
                preselect_ratio=acs_ratio,
                value=1.0 # set preselect region to be 1 
            ).unsqueeze(1)

        masked_kspace = torch.view_as_real(masked_kspace)
        # convert to image space
        images, batches = self.chans_to_batch_dim(_ifft2c(masked_kspace))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )