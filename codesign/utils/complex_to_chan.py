import torch
import numpy as np

def complex_to_chan(complex_tensor, chan_dim=1, num_chan=1):
    assert complex_tensor.dtype in [torch.complex32, torch.complex64, torch.complex128], 'The dtype of the input must be torch.complex32/64/128!'
    assert num_chan in [1, 2], 'Number of channels must be either 1 or 2!'

    if num_chan == 1:
        real_tensor = complex_tensor.abs().unsqueeze(-1)
    elif num_chan == 2:
        real_tensor = torch.view_as_real(complex_tensor)
    dim = np.arange(len(real_tensor.shape))
    dim[chan_dim+1:] = dim[chan_dim:-1]
    dim[chan_dim] = -1
    return real_tensor.permute(*dim)

if __name__ == '__main__':
    a = torch.zeros(2,3,4,5, dtype=torch.complex32)
    b = complex_to_chan(a)
    print(a.shape, b.shape)