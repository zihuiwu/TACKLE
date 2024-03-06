from torch import nn
from ..networks.unet import UNet
from ..networks.modules import GaussianNormalize

class UNetLocalReconstructor(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        chans,
        num_pool_layers,
        drop_prob
    ):
        super().__init__()
        self.unet_local_reconstructor = UNet(in_chans,
                                             out_chans,
                                             chans,
                                             num_pool_layers,
                                             drop_prob)
    
    def forward(self, recon):
        gn = GaussianNormalize()
        recon = gn.input(recon)
        recon_enhanced = recon + self.unet_monai_local_reconstructor(recon)
        recon_enhanced = gn.output(recon_enhanced)
        return recon_enhanced