from torch.fft import fftshift, ifftshift, fftn

def fftn_(image, dim=None):
    if dim is None:
        dim = tuple(range(1, len(image.shape)))
    kspace = fftshift(fftn(ifftshift(image, dim=dim), dim=dim, norm='ortho'), dim=dim)
    return kspace