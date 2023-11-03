from torch import nn

from .absolute_identity import AbsoluteIdentity
from .absolute_value import AbsoluteValue
from .max_min import MaxMin
from .n_activation import NActivation


class Identity(nn.Module):
    @staticmethod
    def forward(x):
        return x
