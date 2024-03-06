from torch import nn
from monai.networks.nets import VNet
from ..networks.modules import GaussianNormalize

class VNetMonaiSegmenter(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.vnet_monai_segmenter = VNet(args.predictor.spatial_dims,
                                         args.predictor.in_channels,
                                         args.predictor.out_channels,
                                         args.predictor.act,
                                         args.predictor.dropout_prob,
                                         args.predictor.dropout_dim,
                                         args.predictor.bias)
    
    def forward(self, recon):
        gn = GaussianNormalize()
        recon = gn.input(recon)
        segmap = self.vnet_monai_segmenter(recon) 
        return segmap