import torch.nn as nn

class IdentityPredictor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, recon):
        return recon