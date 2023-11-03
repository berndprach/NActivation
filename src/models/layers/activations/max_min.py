"""
MaxMin activation function, proposed in https://arxiv.org/abs/1811.05381.
"""

import torch
import torch.nn as nn


class MaxMin(nn.Module):
    @staticmethod
    def forward(x):
        in_size = x.size()
        if in_size[1] % 2 != 0:
            raise ValueError("Did not expect odd number of channels!")
        x_rs = x.view(in_size[0], in_size[1] // 2, 2, *in_size[2:])
        # Order [max, min, max, min, ...]
        x_max = torch.max(x_rs, dim=2, keepdim=True)[0]
        x_min = torch.min(x_rs, dim=2, keepdim=True)[0]
        x_max_min = torch.cat((x_max, x_min), dim=2)
        return x_max_min.view(*in_size)
