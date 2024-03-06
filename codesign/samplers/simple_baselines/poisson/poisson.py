import torch, os
import torch.nn as nn
import numpy as np

class PoissonSampler(nn.Module):
    def __init__(
        self, 
        shape=[320, 320], 
        subsampling_dim=[-2, -1],
        acceleration=4
    ):
        super().__init__()

        # properties
        self.shape = shape
        self.subsampling_dim = subsampling_dim
        self.acceleration = acceleration
        
        # poisson mask
        try:
            mask = np.load(f'{os.getcwd()}/codesign/samplers/simple_baselines/poisson/masks/shape={tuple(shape)}_{acceleration}x.npy')
            self.mask_binarized = torch.from_numpy(mask).unsqueeze(0)
        except:
            raise NotImplementedError(f'Mask with shape {shape} not found!')
        self.mask_binarized_vis = self.mask_binarized[0]

    def forward(self, kspace):
        if self.mask_binarized.dim() != kspace.dim():
            # multicoil data 
            self.mask_binarized = self.mask_binarized.unsqueeze(1)
        kspace_masked = self.mask_binarized.to(kspace.device) * kspace
        self.mask_binarized = self.mask_binarized.squeeze(1)
        return kspace_masked, self.mask_binarized.to(kspace.device)
