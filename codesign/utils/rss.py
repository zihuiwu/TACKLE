import torch

def rss(data, dim=1):
    return torch.sqrt((data ** 2).sum(dim))