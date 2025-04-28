import torch
from torch import nn
import math
import warnings
import random
"""
Notation:
    N -> refer to batch size
    C -> refer to feature dimension
    L/P -> refer to length of particle sequence, or # of particles per event
"""


class SequenceTrimmer(nn.Module):

    def __init__(self, enabled = False, target = (0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)

        self.__enabled = enabled
        self.__target = target
        self.__counter = 0


    def forward(self, x: torch.FloatTensor, v = None, mask = None, U = None) -> torch.FloatTensor:

        #x : N x C x P
        #v: N x 4 x P
        #mask: N x 1 x P -- real particle = 1, padded = 0
        #U: N x C' x P x P
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        
        mask = mask.bool()

        if self.__enabled:
            if self.__counter < 5:
                self.__counter += 1
            else:
                if self.__training:
                    q = min(1, random.uniform(*self.__target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim = -1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim = -1, descending = True) # N x 1 x P
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if U is not None:
                        U = torch.gather(U, -2, perm.unsqueeze(-1).expand_as(U))
                        U = torch.gather(U, -1, perm.unsqueeze(-2).expand_as(U))
                else:
                    maxlen = mask.sum(dim = -1).max()

                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]

                    if v is not None:
                        v = v[:, :, :maxlen]
                    if U is not None:
                        U = U[:, :, :maxlen, :maxlen]

            
        return x, v, mask, U



def build_sparse_tensor(U: torch.FloatTensor, idx: torch.FloatTensor, seq_len: int) -> torch.FloatTensor:
    # U -> N x C x (P*P)
    # idx -> N x 2 x (P*P)
    batch_size, num_features, num_pairs = U.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)

    i = torch.cat((
        torch.arange(0, batch_size, device = U.device).repeat_interleave(num_features * num_pairs).unsqueeze(0),
        idx[:, :1, :].expand_as(U).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(U).flatten().unsqueeze(0),
        ),dim = 0)
    return torch.sparse_coo_tensor(
            i, U.flatten(),
            size = (batch_size, num_features, seq_len + 1, seq_len + 1),
            device = U.device).to_dense()[:, :, :seq_len, :seq_len]
    # N x C x (seq_len*seq_len)


def trunc_normal_(tensor: torch.FloatTensor, mean = 0., std = 1., a = -2., b = 2.) -> torch.FloatTensor:
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8

    r"""
    Fills the input Tensor with values drawn from a truncated normal distribution. The values are effictively  drawn from the normal distribution with values outside [a , b] redrawn until they are within the bounds. The method used for generating the random values works best when |a| = |b|.
    
    tensor: an n-dimensional `torch.Tensor`
    mean: the mean of the normal distribution
    std: the standard deviation of the normal distribution 
    a: right bound
    b: left bound


    """
    def norm_cdf(x):
        return (1. + math.erf(x/ math.sqrt(2.))) /2.
        
    if ( mean < a - 2 * std ) or (mean > b + 2 * std):

        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. ""The distribution of values might be incorrect.", stacklevel = 2)

    with torch.no_grad():
        l = norm_cdf((a - mean) /std)
        u = norm_cdf((b - mean) /std)

        tensor.uniform_(2 * l -1 , 2 * u -1 )


        tensor.erfinv_()


        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min = a, max = b)

        return tensor


