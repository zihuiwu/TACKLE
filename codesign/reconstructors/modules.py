import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """ Self-Attention Layer 
        reference: https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """
    def __init__(self, in_dim, activation):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        B, C, W ,H = x.size()
        query = self.query_conv(x).view(B, -1, W*H) # B X C X N
        key = self.key_conv(x).view(B, -1, W*H) # B X C x N
        energy = torch.bmm(query.transpose(-2,-1), key) # transpose check
        attention = self.softmax(energy) # B X N X N 
        value = self.value_conv(x).view(B, -1, W*H) # B X C X N

        out = torch.bmm(value, attention.transpose(-2,-1))
        out = out.view(B, C, W, H)
        
        out = self.gamma * out + x
        return out, attention
