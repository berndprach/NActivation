import unittest

import torch

from src.models import layers


class TestActivations(unittest.TestCase):
    def test_abs(self):
        layer = layers.AbsoluteValue()
        input_tensor = torch.Tensor([[-1, 0, 1], [-2, 0, 2]])
        output_tensor = layer(input_tensor)
        goal_output = torch.Tensor([[1, 0, 1], [2, 0, 2]])
        self.assertTrue(torch.allclose(output_tensor, goal_output))

    def test_max_min(self):
        layer = layers.MaxMin()
        input_tensor = torch.Tensor([[-1, 0, 1, -2], [-2, 0, 2, 2]])
        output_tensor = layer(input_tensor)
        goal_output = torch.Tensor([[0, -1, 1, -2], [0, -2, 2, 2]])
        self.assertTrue(torch.allclose(output_tensor, goal_output))

    def test_abs_id(self):
        layer = layers.AbsoluteIdentity()
        input_tensor = torch.Tensor([[-1, 0, 1, -2], [-2, 0, 2, 2]])
        output_tensor = layer(input_tensor)
        goal_output = torch.Tensor([[1, 0, 1, -2], [2, 0, 2, 2]])
        self.assertTrue(torch.allclose(output_tensor, goal_output))

    def test_max_min_equivalent_to_abs_id(self):
        abs_id = layers.AbsoluteIdentity()
        max_min = layers.MaxMin()
        torch.manual_seed(1111)
        input_tensor = torch.randn(64, 2)
        rotation = torch.tensor([[1, 1], [-1, 1]])
        rotation = rotation / torch.sqrt(torch.tensor(2.))
        rotation_tp = torch.transpose(rotation, 0, 1)

        rotated_input = rotation_tp[None, :, :] @ input_tensor[:, :, None]
        # rotated_input = torch.mv(rotation_tp[None, :, :], input_tensor)
        rotated_output = abs_id(rotated_input)
        abs_id_output = rotation[None, :, :] @ rotated_output
        abs_id_output = abs_id_output[:, :, 0]

        output_tensor_max_min = max_min(input_tensor)

        self.assertTrue(torch.allclose(abs_id_output,
                                       output_tensor_max_min))
