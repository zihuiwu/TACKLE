from torch import nn
from ..networks.unet import UNet
from ..networks.modules import GaussianNormalize

class UNetSegmenter(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        chans,
        num_pool_layers,
        drop_prob
    ):
        super().__init__()
        self.unet_segmenter = UNet(in_chans,
                                   out_chans,
                                   chans,
                                   num_pool_layers,
                                   drop_prob)
    
    def forward(self, recon):
        gn = GaussianNormalize()
        recon = gn.input(recon)
        segmap = self.unet_segmenter(recon) 
        return segmap