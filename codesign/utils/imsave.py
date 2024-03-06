import torch
import imageio as iio

def imsave(fname, arr):
    assert arr.dtype in [torch.float32, torch.float64, torch.int64]
    try:
        arr.detach().cpu()
    except:
        pass
    if arr.dtype in [torch.float32, torch.float64]:
        arr = ((arr - arr.min()) / (arr.max() - arr.min())* 255).type(torch.uint8)
    elif arr.dtype == torch.int64:
        arr = arr.type(torch.uint8)
    iio.imwrite(fname, arr)