from torch import nn
from monai.networks.nets import UNet
from ..networks.modules import GaussianNormalize

class UNetMonaiLocalReconstructor(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        channels,
        strides
    ):
        super().__init__()
        self.unet_monai_local_reconstructor = UNet(spatial_dims,
                                                   in_channels,
                                                   out_channels,
                                                   channels,
                                                   strides)
    
    def forward(self, recon):
        gn = GaussianNormalize()
        recon = gn.input(recon)
        recon_enhanced = recon + self.unet_monai_local_reconstructor(recon)
        recon_enhanced = gn.output(recon_enhanced)
        return recon_enhanced