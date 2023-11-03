"""
Inspired from SOC code from
https://github.com/singlasahil14/SOC/blob/main/custom_activations.py
"""
import torch
from torch import nn


class AbsoluteIdentity(nn.Module):
    @staticmethod
    def forward(x, axis=1):
        a, b = x.split(x.shape[axis] // 2, axis)
        c, d = torch.abs(a), b
        return torch.cat([c, d], dim=axis)
