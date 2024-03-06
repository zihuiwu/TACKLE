import torch 
import torch.nn as nn
from codesign.samplers.modules import preselect

class CenterSampler(nn.Module):
    def __init__(
        self, 
        shape=[320, 320], 
        subsampling_dim=[-2, -1],
        acceleration=4, 
        line_constrained=False
    ):
        super().__init__()

        # properties
        self.shape = shape
        self.subsampling_dim = subsampling_dim
        self.acceleration = acceleration
        self.line_constrained = line_constrained
        
        # center_mask
        self.mask_binarized = preselect(
            mask=torch.zeros(1, *self.shape),
            dim=self.subsampling_dim, 
            preselect_num=0, 
            preselect_ratio=self.acceleration, 
            line_constrained=self.line_constrained
        )
        self.mask_binarized_vis = self.mask_binarized[0]

    def forward(self, kspace):
        if self.mask_binarized.dim() != kspace.dim():
            # multicoil data 
            self.mask_binarized = self.mask_binarized.unsqueeze(1)
        kspace_masked = self.mask_binarized.to(kspace.device) * kspace
        self.mask_binarized = self.mask_binarized.squeeze(1)
        return kspace_masked, self.mask_binarized.to(kspace.device)

if __name__ == '__main__':
    import scipy.io as sio
    cs = CenterSampler(shape=[192, 192])
    mask = cs.mask_binarized.numpy()[0]
    sio.savemat('center_192*192_4x.mat', {'mask': mask})
