import torch.nn as nn
from monai.networks.nets import UNet
from ..networks.modules import GaussianNormalize
from ..utils.ifftn import ifftn_
from ..utils.complex_to_chan import complex_to_chan
from ..utils.multicoil_ops import _rss

class UNetMonaiReconstructor(nn.Module):
    def __init__(self,
                 spatial_dims=2,
                 in_channels=1,
                 out_channels=1,
                 channels=[32, 64, 128, 256],
                 strides=[2, 2, 2],
                 norm=True):
        super().__init__()
        self.unet_monai_recon = UNet(spatial_dims,
                                     in_channels,
                                     out_channels,
                                     channels,
                                     strides)
        self.norm = norm

    def forward(self, kspace_sampled, mask_binarized):
        recon_zf = ifftn_(kspace_sampled)
        recon_zf = complex_to_chan(recon_zf, chan_dim=1, num_chan=1)
        if recon_zf.dim() == 5:
            # multi-coil 
            recon_zf = _rss(recon_zf, dim=2)
        if self.norm:
            gn = GaussianNormalize()
            recon_zf = gn.input(recon_zf)
            recon = recon_zf + self.unet_monai_recon(recon_zf)
            recon = gn.output(recon)
            recon_zf = gn.output(recon_zf)
            # from ..utils.fftn import fftn_
            # kspace_pred = fftn_(recon, dim=(-2,-1)).squeeze(1)
            # kspace_pred -= kspace_pred*mask_binarized - kspace_sampled
            # recon = complex_to_chan(ifftn_(kspace_pred), chan_dim=1, num_chan=1)
        else:
            recon = recon_zf + self.unet_monai_recon(recon_zf) 
        return recon, recon_zf