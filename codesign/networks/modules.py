import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """
    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super(ConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class GaussianNormalize:
    def input(self, input: torch.Tensor):
        self.mean = torch.mean(input, tuple(range(1, input.dim())), keepdim=True)
        self.std = torch.std(input, tuple(range(1, input.dim())), keepdim=True)
        return (input - self.mean.detach()) / self.std.detach()
    
    def output(self, output):
        return self.mean.detach() + self.std.detach() * output


class ComplexGaussianNormalize:
    def input(self, input: torch.Tensor):
        self.mean = torch.mean(input.abs(), tuple(range(1, input.dim())), keepdim=True)
        self.std = torch.std(input.abs(), tuple(range(1, input.dim())), keepdim=True)
        return (input - self.mean.detach()) / self.std.detach()
    
    def output(self, output):
        return self.mean.detach() + self.std.detach() * output
