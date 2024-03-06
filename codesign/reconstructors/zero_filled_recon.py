import torch.nn as nn
from ..utils.ifftn import ifftn_
from ..utils.complex_to_chan import complex_to_chan
from codesign.utils.multicoil_ops import _rss

class ZeroFilledReconstructor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, kspace_sampled, mask_binarized):
        recon_zf = ifftn_(kspace_sampled)
        recon_zf = complex_to_chan(recon_zf)

        if recon_zf.dim() == 5:
            # multi-coil 
            recon_zf = _rss(recon_zf, dim=2)
        
        return recon_zf, recon_zf