import torch
import numpy as np
from torch.autograd import Function

def preselect(mask, dim, preselect_num, preselect_ratio, value=1, line_constrained=False):
    if preselect_num > 0 and preselect_ratio > 0:
        raise AssertionError('preselect_num and preselect_ratio cannot both be > 0. Only use one of them')
    
    index = [slice(None)] * len(mask.shape)
    if line_constrained:
        assert isinstance(dim, int), 'there must be exactly one dimension for preselection'

        D = mask.shape[dim]
        if preselect_ratio > 0:
            preD = int(D / preselect_ratio)
            index[dim] = slice((D-preD)//2, (D-preD)//2+preD)
        elif preselect_num > 0:
            index[dim] = slice((D-preselect_num)//2, (D-preselect_num)//2+preselect_num)
    else:
        assert isinstance(dim, list), 'the dimensions for preselection must be a list'
        assert len(dim) == 2, 'there must be exactly two dimensions for preselection'

        H, W = mask.shape[dim[0]], mask.shape[dim[1]]
        if preselect_ratio > 0:
            preH, preW = int(H / np.sqrt(preselect_ratio)), int(W / np.sqrt(preselect_ratio)) 
            index[dim[0]] = slice((H-preH)//2, (H-preH)//2+preH)
            index[dim[1]] = slice((W-preW)//2, (W-preW)//2+preW)
        elif preselect_num > 0:
            index[dim[0]] = slice((H-preselect_num)//2, (H-preselect_num)//2+preselect_num)
            index[dim[1]] = slice((W-preselect_num)//2, (W-preselect_num)//2+preselect_num)
        
    mask[index] = value

    return mask

def RescaleProbMap(batch_x, sparsity):
    """
        Rescale Probability Map
        given a prob map x, rescales it so that it obtains the desired sparsity

        if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
        if mean(x) < sparsity, one can basically do the same thing by rescaling
                                (1-x) appropriately, then taking 1 minus the result.
    """
    batch_size = len(batch_x)
    ret = []
    for i in range(batch_size):
        x = batch_x[i:i+1]
        xbar = torch.mean(x)
        r = sparsity / (xbar)
        beta = (1-sparsity) / (1-xbar)

        # compute adjucement
        le = torch.le(r, 1).float()
        ret.append(le * x * r + (1-le) * (1 - (1 - x) * beta))

    return torch.cat(ret, dim=0)

class NanDebugger(Function):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def forward(ctx, input, step_name):
        ctx.save_for_backward(input, step_name)
        return input.clone() 

    @staticmethod
    def backward(ctx, grad_output):
        if torch.isnan(grad_output).any():
            input, step_name = ctx.saved_tensors
            print(f"nan {step_name}")

            import pdb; pdb.set_trace()
        return grad_output, None 

class ThresholdRandomMaskSigmoidV1(Function):
    def __init__(self):
        """
            Straight through estimator.
            The forward step stochastically binarizes the probability mask.
            The backward step estimate the non differentiable > operator using sigmoid with large slope (10).
        """
        super(ThresholdRandomMaskSigmoidV1, self).__init__()

    @staticmethod
    def forward(ctx, input):
        batch_size = len(input)
        probs = [] 
        results = [] 

        for i in range(batch_size):
            x = input[i:i+1]

            count = 0 
            while True:
                prob = x.new(x.size()).uniform_()
                result = (x > prob).float()

                if torch.isclose(torch.mean(result), torch.mean(x), atol=1e-3):
                    break

                count += 1 

                if count > 1000:
                    print("something wrong with your code ", 
                        torch.mean(prob), torch.mean(result), torch.mean(x))
                    assert 0 

            probs.append(prob)
            results.append(result)

        results = torch.cat(results, dim=0)
        probs = torch.cat(probs, dim=0)
        ctx.save_for_backward(input, probs)

        return results  

    @staticmethod
    def backward(ctx, grad_output):
        slope = 10
        input, prob = ctx.saved_tensors

        # derivative of sigmoid function
        current_grad = slope * torch.exp(-slope * (input - prob)) / torch.pow((torch.exp(-slope*(input-prob))+1), 2)

        return current_grad * grad_output


def MaximumBinarize(input):
    batch_size = len(input)
    results = [] 

    for i in range(batch_size):
        x = input[i:i+1]
        num = torch.sum(x).round().int()

        indices = torch.topk(x.reshape(-1), k=num)[1]

        mask = torch.zeros_like(x).reshape(-1)
        
        mask[indices] = 1

        mask = mask.reshape(*x.shape) 

        results.append(mask)

    results = torch.cat(results, dim=0)

    return results  

def MaximumBinarizeLineConstrained(input):
    batch_size = len(input)
    results = [] 

    for i in range(batch_size):
        x = input[i:i+1]
        num = torch.sum(x / x.shape[-2]).round().int() * x.shape[-2] # make sure to select the entire line by making sure that num is divisible by x.shape[-2]

        indices = torch.topk(x.reshape(-1), k=num)[1]

        mask = torch.zeros_like(x).reshape(-1)
        
        mask[indices] = 1

        mask = mask.reshape(*x.shape) 

        results.append(mask)

    results = torch.cat(results, dim=0)

    return results  