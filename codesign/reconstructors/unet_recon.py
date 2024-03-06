import torch.nn as nn
from ..networks.unet import UNet
from ..networks.modules import GaussianNormalize
from ..utils.ifftn import ifftn_
from ..utils.complex_to_chan import complex_to_chan
from ..utils.multicoil_ops import _rss

class UNetReconstructor(nn.Module):
    def __init__(self,
                 in_chans=2,
                 out_chans=2,
                 chans=32,
                 num_pool_layers=4,
                 drop_prob=0,
                 norm=True):
        super().__init__()
        self.UNet = UNet(in_chans,
                         out_chans,
                         chans,
                         num_pool_layers,
                         drop_prob)
        self.norm = norm
        
    def forward(self, kspace_sampled, mask_binarized):
        recon_zf = ifftn_(kspace_sampled)
        recon_zf = complex_to_chan(recon_zf)
        if recon_zf.dim() == 5:
            # multi-coil 
            recon_zf = _rss(recon_zf, dim=2)
        if self.norm:
            gn = GaussianNormalize()
            recon_zf = gn.input(recon_zf)
            recon = recon_zf + self.unet_monai_recon(recon_zf)
            recon = gn.output(recon)
            recon_zf = gn.output(recon_zf)
        else:
            recon = recon_zf + self.UNet(recon_zf) 
        return recon, recon_zf