import torch
from torch.nn.functional import one_hot
from codesign.utils.permute_to_chan import permute_to_chan

def seg_argmax_pred(pred, chan_dim=1):
    return permute_to_chan(
        one_hot(
            torch.argmax(pred, dim=chan_dim), 
            num_classes=pred.size(chan_dim)
        ), 
        chosen_dim=-1, 
        target_dim=chan_dim
    )