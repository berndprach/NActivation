
import unittest

import torch
from src.models import layers
from torch import nn


class TestBasic(unittest.TestCase):
    def test_pixel_unshuffle(self):
        layer = nn.PixelUnshuffle(2)
        # input_tensor = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
        ones = torch.ones((4, 4))
        input_tensor = torch.stack([1*ones, 2*ones, 3*ones])[None]
        print(input_tensor.shape)
        output_tensor = layer(input_tensor)

        goal_channels = sum((4*[i] for i in range(1, 4)), start=[])
        print(goal_channels)
        ones = torch.ones((2, 2))
        goal_output = torch.stack([ones*i for i in goal_channels])[None]
        print(goal_output.shape)

        self.assertTrue(torch.allclose(output_tensor, goal_output))

    def test_concatenation_pooling_shape(self):
        layer = layers.ConcatenationPooling2D(window_shape=(2, 2))
        input_tensor = torch.ones((2, 3, 28, 28))
        output_tensor = layer(input_tensor)
        self.assertEqual(output_tensor.shape, (2, 12, 14, 14))

    def test_concatenation_pooling(self):
        layer = layers.ConcatenationPooling2D(window_shape=(2, 2))
        ones = torch.ones((4, 4))
        input_tensor = torch.stack([1*ones, 2*ones, 3*ones])[None]
        print(input_tensor.shape)

        output_tensor = layer(input_tensor)
        print(output_tensor.shape)
        print(output_tensor)

        goal_channels = [i for i in range(1, 4)]*4
        print(goal_channels)
        ones = torch.ones((2, 2))
        goal_output = torch.stack([ones*i for i in goal_channels])[None]
        print(goal_output.shape)
        print(goal_output)

        self.assertTrue(torch.allclose(output_tensor, goal_output))


if __name__ == "__main__":
    unittest.main()
