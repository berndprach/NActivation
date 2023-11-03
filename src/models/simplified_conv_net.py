"""
Simplified ConvNet model.
Code adapted from
"1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness", 2023.
"""

import torch

from dataclasses import dataclass
from torch import nn
from typing import Type, Optional, Union, Protocol
from collections import OrderedDict

from .layers import (
    ConcatenationPooling2D,
    FirstChannels,
    ZeroChannelConcatenation,
)


class Activation(Protocol):
    def __init__(self, in_channels: int): ...
    def forward(self, x): ...


@dataclass
class SimplifiedConvNetHyperparameters:
    # Layers:
    activation_cls: Type[Activation]
    conv_cls: Type[nn.Conv2d]
    first_conv_cls: Type[nn.Conv2d] = None
    head_conv_cls: Type[nn.Conv2d] = None

    # Size:
    base_width: int = 16
    nrof_blocks: int = 5
    nrof_layers_per_block: int = 5
    kernel_size: int = 3
    nrof_dense_layers: int = 1

    # Classification head:
    nrof_classes: Optional[int] = 10

    def __post_init__(self):
        if self.first_conv_cls is None:
            self.first_conv_cls = self.conv_cls
        if self.head_conv_cls is None:
            self.head_conv_cls = self.conv_cls


class ConvBlock(nn.Sequential):
    def __init__(self,
                 conv_cls: Type[nn.Conv2d],
                 activation_cls: Type[Activation],
                 in_channels: int,
                 nrof_convolutions: int,
                 kernel_size: int,
                 with_pooling: bool = True,
                 ):
        layers = []
        for _ in range(nrof_convolutions):
            conv = conv_cls(in_channels, in_channels, kernel_size)
            act = activation_cls(in_channels)
            layers.extend([conv, act])

        if with_pooling:
            layers.append(FirstChannels(in_channels // 2))
            layers.append(ConcatenationPooling2D((2, 2)))

        super().__init__(*layers)


class SimplifiedConvNet(nn.Sequential):
    def __init__(self, *args, seed: Union[int, None] = None, **kwargs):
        if seed is not None:
            torch.manual_seed(seed)
        self.hp = SimplifiedConvNetHyperparameters(*args, **kwargs)
        layers = get_layers(self.hp)
        super().__init__(OrderedDict(layers))


def get_layers(hp: SimplifiedConvNetHyperparameters):
    first_conv = hp.first_conv_cls(
        in_channels=hp.base_width,
        out_channels=hp.base_width,
        kernel_size=1
    )

    kernel_sizes = [hp.kernel_size for _ in range(hp.nrof_blocks)]
    kernel_sizes[-1] = 1  # 2x2 blocks do not allow kernel size >= 3.
    conv_blocks = []
    for i in range(hp.nrof_blocks):
        block = ConvBlock(
            hp.conv_cls,
            hp.activation_cls,
            hp.base_width * 2 ** i,
            hp.nrof_layers_per_block,
            kernel_sizes[i]
        )
        conv_blocks.append((f"Block{i + 1}", block))

    final_width = hp.base_width * 2 ** hp.nrof_blocks
    if hp.nrof_dense_layers > 1:
        dense_layers = [("DenseBlock", ConvBlock(
            hp.conv_cls,
            hp.activation_cls,
            final_width,
            hp.nrof_dense_layers - 1,
            1,
            with_pooling=False,
        ))]
    else:
        dense_layers = []

    classification_head = nn.Sequential(
        hp.head_conv_cls(
            in_channels=hp.base_width * 2 ** hp.nrof_blocks,
            out_channels=hp.base_width * 2 ** hp.nrof_blocks,
            kernel_size=1),
        FirstChannels(hp.nrof_classes),
        nn.Flatten()
    )

    layers = [
        ("ZeroConcatenation", ZeroChannelConcatenation(hp.base_width)),
        ("FirstConv", first_conv),
        ("FirstActivation", hp.activation_cls(hp.base_width)),
        *conv_blocks,
        *dense_layers,
        ("ClassificationHead", classification_head),
    ]
    return layers


DEFAULT_MODELS = {
    "ConvNetXS": dict(base_width=16),
    "ConvNetS": dict(base_width=32),
    "ConvNetM": dict(base_width=64),
    "ConvNetL": dict(base_width=128),
    "ConvNet64XS": dict(base_width=8, nrof_blocks=6),
    "ConvNet64S": dict(base_width=16, nrof_blocks=6),
    "ConvNet64M": dict(base_width=32, nrof_blocks=6),
    "ConvNet64L": dict(base_width=64, nrof_blocks=6),
}
