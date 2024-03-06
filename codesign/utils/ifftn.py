from torch.fft import fftshift, ifftshift, ifftn

def ifftn_(kspace, dim=None):
    if dim is None:
        dim = tuple(range(1, len(kspace.shape)))
    image = fftshift(ifftn(ifftshift(kspace, dim=dim), dim=dim, norm='ortho'), dim=dim)
    return image