import torch.nn as nn
from ..networks.modules import GaussianNormalize
from torchvision.models import resnet101 

class ResNet101Classifier(nn.Module):
    def __init__(
        self,         
        in_chans=1,
        num_classes=2
    ):
        super().__init__()
        self.classifier = resnet101(
            pretrained=False, 
            num_classes=num_classes,
        )
        self.classifier.conv1 = nn.Conv2d(
            in_chans, 
            64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
    
    def forward(self, recon):
        gn = GaussianNormalize()
        recon = gn.input(recon)
        label = self.classifier(recon)
        return label