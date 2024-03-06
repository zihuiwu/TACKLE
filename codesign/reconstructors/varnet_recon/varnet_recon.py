import torch
import torch.nn as nn
from .unet import NormUnet
from .sensitivity_model import SensitivityModel
from codesign.utils.complex_to_chan import complex_to_chan
from codesign.utils.ifftn import ifftn_
from codesign.utils.multicoil_ops import (
    _fft2c,
    _rss,
    _ifft2c,
    _sens_expand,
    _sens_reduce
)

class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.
    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module, dc_weight: float, adj_dc_weight: bool):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()
        self.model = model
        
        if adj_dc_weight:
            self.dc_weight = nn.Parameter(dc_weight*torch.ones(1))
        else:
            self.dc_weight = dc_weight

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor 
    ) -> torch.Tensor:
        soft_dc = (current_kspace * mask - ref_kspace) * self.dc_weight

        num_coil = current_kspace.shape[1]

        if num_coil == 1:  # single coil 
            model_term = _fft2c(self.model(_ifft2c(current_kspace)))
        else:  # multi coil 
            model_term = _sens_expand(
                self.model(_sens_reduce(current_kspace, sens_maps)), sens_maps
            )
        
        return current_kspace - soft_dc - model_term


class VarNetReconstructor(nn.Module):
    """
    A full variational network model.
    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
        acs_ratio: int = 0,
        dc_weight: float = 0.5,
        adj_dc_weight: bool = True
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools), dc_weight, adj_dc_weight) 
                for _ in range(num_cascades)]
        )
        self.acs_ratio = acs_ratio
        self.sens_net = SensitivityModel()

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        recon_zf = complex_to_chan(ifftn_(masked_kspace, dim=[-2, -1]), chan_dim=1, num_chan=1)
        if recon_zf.dim() == 5:
            # multi-coil 
            recon_zf = _rss(recon_zf, dim=2)
            sens_maps = self.sens_net(masked_kspace, mask, self.acs_ratio)
        else:
            # single coil 
            sens_maps = None 

        kspace_pred = masked_kspace.clone()
        kspace_pred = torch.view_as_real(kspace_pred) 
        ref_kspace = torch.view_as_real(masked_kspace) 
        
        if kspace_pred.dim() < 5:
            # single coil 
            kspace_pred = kspace_pred.unsqueeze(1)
            ref_kspace = ref_kspace.unsqueeze(1)

        mask = mask.unsqueeze(1).unsqueeze(-1)

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, ref_kspace, mask, sens_maps)

        # kspace_pred -= kspace_pred*mask - ref_kspace
        recon = ifftn_(torch.view_as_complex(kspace_pred), dim=[-2, -1]).squeeze(1)
        recon = complex_to_chan(recon, chan_dim=1, num_chan=1)
        if recon.dim() == 5:
            # multi-coil
            recon = _rss(recon, dim=2)
        
        return recon, recon_zf 