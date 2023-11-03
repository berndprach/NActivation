from torch import nn

from src.models.layers.activations import (
    AbsoluteIdentity,
    Identity,
    MaxMin,
)
from src.models.layers.activations.absolute_value import AbsoluteValue
from src.models.layers.activations.n_activation import (
    NActivation,
    RandomLogUniformNActivation,
    RandomNormalNActivation,
)
from src.models.layers.basic import (
    ConcatenationPooling2D,
    FirstChannels,
    ZeroChannelConcatenation,
)
from src.models.layers.lipschitz.aol import (
    AOLLinear,
    AOLLinearOrthogonal,
    AOLConv2d,
    AOLConv2dOrthogonal,
)
from src.models.layers.lipschitz.cpl import CPLConv2d
from src.models.layers.lipschitz.soc import SOCConv2d


class Conv2d(nn.Conv2d):
    def __init__(self, *args, padding="same", **kwargs):
        super().__init__(*args, padding=padding, **kwargs)
        # self.name = "Conv2d"
        nn.init.dirac_(self.weight)
