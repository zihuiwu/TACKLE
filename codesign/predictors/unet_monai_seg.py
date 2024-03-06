from torch import nn
from monai.networks.nets import UNet
from ..networks.modules import GaussianNormalize

class UNetMonaiSegmenter(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        channels,
        strides
    ):
        super().__init__()
        self.unet_monai_segmenter = UNet(spatial_dims,
                                         in_channels,
                                         out_channels,
                                         channels,
                                         strides)
    
    def forward(self, recon):
        gn = GaussianNormalize()
        recon = gn.input(recon)
        segmap = self.unet_monai_segmenter(recon) 
        return segmap