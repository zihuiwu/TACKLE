import torch

def center_crop(data, dim=[-2, -1], shape=[128, 128]):
    assert len(dim) == len(shape), '"dim" and "shape" must have the same length'
    if [data.shape[d] for d in dim] == shape:
        # return data if no cropping is needed
        return data

    index = [slice(None)] * len(data.shape)
    for i, d in enumerate(dim):
        assert 0 < shape[i] <= data.shape[d], f'desired size is larger than the size of data in {d}th dimension'

        d_from = (data.shape[d] - shape[i]) // 2
        d_to = d_from + shape[i]
        index[d] = slice(d_from, d_to)

    return data[index]

if __name__ == '__main__':
    a = torch.zeros(2,2,20,30,40)
    b = center_crop(a, dim=[-3, -1], shape=[10, 20])
    print(a.shape, b.shape)