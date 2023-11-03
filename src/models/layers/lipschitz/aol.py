"""
Almost Orthogonal Lipschitz (AOL) layer.
Proposed in https://arxiv.org/abs/2208.03160
Code adapted from
"1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness", 2023.
"""

from typing import Optional, Callable, Union

from torch import nn, Tensor
import torch
from torch.nn.common_types import _size_2_t
from torch.nn.utils.parametrize import register_parametrization


def aol_conv2d_rescaling(weight: Tensor) -> Tensor:
    """ Expected weight shape: out_channels x in_channels x ks1 x ks_2 """
    _, _, k1, k2 = weight.shape
    weight_tp = weight.transpose(0, 1)
    v = torch.nn.functional.conv2d(
        weight_tp, weight_tp, padding=(k1 - 1, k2 - 1))
    v_scaled = v.abs().sum(dim=(1, 2, 3), keepdim=True).transpose(0, 1)
    return weight / (v_scaled + 1e-6).sqrt()


def aol_linear_rescaling(weight: Tensor) -> Tensor:  # shape: out x in
    wwt = torch.matmul(weight.transpose(0, 1), weight)  # shape: in x in
    ls_bounds_squared = wwt.abs().sum(dim=0, keepdim=True)  # shape: 1 x in
    return weight / (ls_bounds_squared + 1e-6).sqrt()  # shape: out x in


class AOLConv2dRescaling(nn.Module):
    def forward(self, weight: Tensor) -> Tensor:
        return aol_conv2d_rescaling(weight)


class AOLLinearRescaling(nn.Module):
    def forward(self, weight: Tensor) -> Tensor:
        return aol_linear_rescaling(weight)


class AOLConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 initializer: Optional[Callable] = None,
                 padding: Union[_size_2_t, str] = 'same',
                 padding_mode: str = 'circular',
                 **kwargs) -> None:
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=padding, padding_mode=padding_mode, **kwargs)

        if initializer is None:
            initializer = nn.init.dirac_
        initializer(self.weight)

        torch.nn.init.zeros_(self.bias)

        register_parametrization(self, 'weight', AOLConv2dRescaling())


class AOLConv2dOrthogonal(AOLConv2d):
    """ Alias for AOLConv2d with orthogonal initialization. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         initializer=torch.nn.init.orthogonal_,
                         **kwargs)


class AOLLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 initializer: Optional[Callable] = None,
                 **kwargs) -> None:
        super().__init__(in_features, out_features, **kwargs)

        if initializer is None:
            initializer = nn.init.eye_
        initializer(self.weight)

        torch.nn.init.zeros_(self.bias)

        register_parametrization(self, 'weight', AOLLinearRescaling())


class AOLLinearOrthogonal(AOLLinear):
    """ Alias for AOLLinear with orthogonal initialization. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         initializer=torch.nn.init.orthogonal_,
                         **kwargs)
