import torch
import torch.nn as nn

class FullSampler(nn.Module):
    def __init__(
        self, 
        shape=[320, 320]
    ):
        super().__init__()

        # properties
        self.shape = shape
        
        # center_mask
        self.mask_binarized = torch.ones(1, *self.shape)
        self.mask_binarized_vis = self.mask_binarized[0]

    def forward(self, kspace):
        return kspace, self.mask_binarized.to(kspace.device)
