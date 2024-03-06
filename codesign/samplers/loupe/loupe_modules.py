import torch
import torch.nn as nn
from torch.fft import fftshift


class LineConstrainedProbMask(nn.Module):
    """
    A learnable probablistic mask with the same shape as the kspace measurement.
    The mask is constrinaed to include whole kspace lines in the readout direction
    """
    def __init__(self, shape=[32], slope=5):
        super().__init__()
        self.slope = slope
        self.mask = nn.Parameter(self._slope_random_uniform(shape[-1]))

    def _slope_random_uniform(self, shape, eps=1e-2):
        temp = torch.zeros(shape).uniform_(eps, 1-eps)
        return -torch.log(1./temp-1.) / self.slope

    def forward(self, input):
        mask_prob = torch.sigmoid(self.slope * self.mask).view(1, 1, input.shape[-1]).repeat(1, input.shape[-2], 1)
        return mask_prob 
    

class ProbMask(nn.Module):
    """
    A learnable probablistic mask with the same shape as the kspace measurement.
    This learned mask samples measurements in the whole kspace
    """
    def __init__(self, shape=[320, 320], slope=5, preselect=False, preselect_num=0, preselect_ratio=0):
        """
            shape ([int. int]): Shape of the reconstructed image
            slope (float): Slope for the Loupe probability mask. Larger slopes make the mask converge faster to
                           deterministic state.
        """
        super(ProbMask, self).__init__()

        self.slope = slope
        self.preselect = preselect 
        self.preselect_num_one_side = preselect_num // 2 
        self.preselect_ratio = preselect_ratio

        init_tensor = self._slope_random_uniform(shape)
        self.mask = nn.Parameter(init_tensor, requires_grad=True)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape NHWC

        Returns:
            (torch.Tensor): Output tensor of shape NHWC
        """
        logits = self.mask.view(1, input.shape[-2], input.shape[-1])
        mask_prob =  torch.sigmoid(self.slope * logits)
        return fftshift(mask_prob, dim=[-2, -1])

    def _slope_random_uniform(self, shape, eps=1e-2):
        """
            uniform random sampling mask
        """
        temp = torch.zeros(shape).uniform_(eps, 1-eps)

        # logit with slope factor
        logits = -torch.log(1./temp-1.) / self.slope

        logits = logits.reshape(1, shape[0], shape[1]) 

        if self.preselect_ratio > 0:
            H, W = shape 

            subH, subW = int(H / (self.preselect_ratio**(1/2)) / 2), int(W / (self.preselect_ratio**(1/2)) / 2) 

            logits[:, :subH, :subW] = -torch.inf
            logits[:, :subH, -subW:] = -torch.inf 
            logits[:, -subH:, :subW] = -torch.inf
            logits[:, -subH:, -subW:] = -torch.inf  

        """
        elif self.preselect_num_one_side > 0:
            logits[:, :self.preselect_num_one_side, :self.preselect_num_one_side] = -1e2
            logits[:, :self.preselect_num_one_side, -self.preselect_num_one_side:] = -1e2 
            logits[:, -self.preselect_num_one_side:, :self.preselect_num_one_side] = -1e2 
            logits[:, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = -1e2  
        """
        return logits 