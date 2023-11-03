
import unittest

import torch
from torch import nn, Tensor

from src.models import SimplifiedConvNet
from src.models import util


class PrintConv(nn.Conv2d):
    def __init__(self, *args, padding="same", **kwargs):
        super().__init__(*args, padding=padding, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        print(f"Conv2d({x.shape})")
        return super().forward(x)


class TestSimplifiedConvNet(unittest.TestCase):
    def test_standard_layers(self):
        model = SimplifiedConvNet(
            conv_cls=PrintConv,
            activation_cls=nn.ReLU
        )

        print(model)

        all_layers = util.get_all_layers(model)
        print(model)
        nrof_convolutions = len([layer for layer in all_layers
                                 if isinstance(layer, nn.Conv2d)])
        self.assertEqual(nrof_convolutions, 5 * 5 + 2)

        nrof_activations = len([layer for layer in all_layers
                                if isinstance(layer, nn.ReLU)])
        self.assertEqual(nrof_activations, 5 * 5 + 1)

        test_input = torch.ones((2, 3, 32, 32))
        output = model(test_input)
        self.assertEqual(output.shape, (2, 10))


