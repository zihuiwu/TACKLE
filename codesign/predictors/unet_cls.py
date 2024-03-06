import torch.nn as nn
from ..networks.unet import UNet
from ..networks.modules import GaussianNormalize

class UNetClassifier(nn.Module):
    def __init__(self,
                 in_chans=2,
                 out_chans=2,
                 chans=64,
                 num_pool_layers=4,
                 drop_prob=0):
        super().__init__()
        self.unet = UNet(in_chans,
                         out_chans,
                         chans,
                         num_pool_layers,
                         drop_prob)
        self.classifier = ...
    
    def forward(self, recon):
        gn = GaussianNormalize()
        recon = gn.input(recon)
        features = self.unet(recon)
        label = self.classifier(features)
        return label