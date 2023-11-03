import unittest

import torch
from matplotlib import pyplot as plt
from torch import nn

from src import models


class TestLayersInModel(unittest.TestCase):
    def test_simple_dense(self):
        width = 40
        simple_fc = nn.Sequential(
            models.layers.ZeroChannelConcatenation(concatenate_to=40),
            models.layers.AOLLinear(width, width),
            models.layers.NActivation(width, theta_init_values=(0., 0.)),
            models.layers.AOLLinear(width, width),
            models.layers.NActivation(width, theta_init_values=(0., 0.)),
            models.layers.AOLLinear(width, width),
            models.layers.FirstChannels(nrof_channels=1)
        )

        input_tensor = torch.rand(32, 1)
        output_tensor = simple_fc(input_tensor)
        self.assertEqual(output_tensor.shape, (32, 1))
        self.assertTrue(torch.allclose(output_tensor, input_tensor))

    # def test_simple_dense_with_activation(self):
    #     width = 40
    #     simple_fc = nn.Sequential(
    #         models.layers.ZeroChannelConcatenation(concatenate_to=40),
    #         models.layers.AOLLinear(width, width),
    #         models.layers.NActivation(width, theta_init_values=(-0.5, 0.5)),
    #         models.layers.AOLLinear(width, width),
    #         models.layers.NActivation(width, theta_init_values=(-0.5, 0.5)),
    #         # models.layers.NActivation(width, theta_init_values=(-0., 0.)),
    #         models.layers.AOLLinear(width, width),
    #         models.layers.FirstChannels(nrof_channels=1)
    #     )
    #
    #     input_tensor = torch.linspace(-3., 3., 512)[:, None]
    #     output_tensor = simple_fc(input_tensor)
    #     self.assertEqual(output_tensor.shape, (512, 1))
    #     plt.title("Should be like /\\/\\/!")
    #     plt.plot(input_tensor.detach(), output_tensor.detach())
    #     plt.show()
    #     # Should be a "double N" like
    #     #       /
    #     #  /\/\/
    #     # /
