import torch
import numpy as np

def permute_to_chan(tensor, chosen_dim=-1, target_dim=0):
    dim = np.arange(len(tensor.shape))
    dim[target_dim+1:] = dim[target_dim:-1]
    dim[target_dim] = chosen_dim
    return tensor.permute(*dim)

if __name__ == '__main__':
    tensor = torch.zeros(2,3,4,5)
    dim = permute_to_chan(tensor, chosen_dim=-1, target_dim=0)
    print(dim)