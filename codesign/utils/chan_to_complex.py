import torch
import numpy as np

def chan_to_complex(real_tensor, chan_dim=1):
    assert real_tensor.dtype in [torch.float32, torch.float64], 'The dtype of the input must be torch.float32/64!'
    assert real_tensor.size(1) == 2, 'The second dimension of the input must be 2!'

    dim = np.arange(len(real_tensor.shape))
    dim[chan_dim:-1] = dim[chan_dim+1:]
    dim[-1] = chan_dim
    real_tensor = real_tensor.permute(*dim).contiguous()
    
    return torch.view_as_complex(real_tensor)

if __name__ == '__main__':
    a = torch.zeros(2,2,3,4,5, dtype=torch.float32)
    b = chan_to_complex(a)
    print(a.shape, b.shape)