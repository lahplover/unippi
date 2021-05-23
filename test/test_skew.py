import torch
import torch.nn.functional as F


def _unskew(x, L):
    # L is the relative attention span
    # (N, S, S) -> (N, S, L)
    N, S, S = x.size()
    x = x.view(N, -1)  # N x SS
    x = F.pad(x, (S, 0))  # N x (SS+S)
    x = x.view(N, S, S + 1)  # N x S x (S+1)
    x = x[:, :, -L:]  # N x S x L
    return x


def _skew(x, pad_value=0):
    # X = N x S x L  -> (N, S, S)
    N, S, L = x.size()
    x = F.pad(x, (S - L + 1, 0), value=pad_value)  # N x S x (S+1)
    x = x.view(N, -1)  # N x (SS+S)
    x = x[:, S:]  # N x SS
    x = x.view(N, S, S)  # N x S x S
    return x


x = torch.arange(32).reshape(2, 4, 4) + 1
a = _unskew(x, 2)
c = _skew(a, 0)

x = torch.arange(32).reshape(2, 4, 4) + 1
a = _unskew(x, 4)
c = _skew(a, 0)

x = torch.arange(16).reshape(2, 4, 2) + 1
b = _skew(x, 0)

x = torch.arange(32).reshape(2, 4, 4) + 1
b = _skew(x, 0)

