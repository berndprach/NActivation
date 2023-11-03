import unittest

import torch

from src import models
from src.models import layers
from src.models.util import get_operator_norms_of_input_gradients


class TestGetLargestInputGradient(unittest.TestCase):
    def test_computation_of_jacobian_shape(self):
        # Dummy test to check the shape of the Jacobian.
        input_batch = torch.ones((2, 3, 32, 32))
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 7, 3, padding=1),
            torch.nn.MaxPool2d(32),
            torch.nn.Flatten(),
        )
        output_shape = model(input_batch).shape
        jacobian = torch.autograd.functional.jacobian(model, input_batch)
        print(jacobian.shape)
        # self.assertEqual(jacobian.shape, (2, 7, 2, 3, 32, 32))
        self.assertEqual(jacobian.shape,
                         (*output_shape, *input_batch.shape))

        self.assertTrue(torch.all(jacobian[0, :, 1] == 0))

    def test_jacobian_is_constant_for_linear_layers(self):
        # Dummy sanity check test.
        input_batch = torch.randn((16, 3, 32, 32))
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.Conv2d(8, 7, 3, padding=1),
            torch.nn.AvgPool2d(32),
            torch.nn.Flatten(),
            torch.nn.Linear(7, 7),
        )
        jacobians = []
        for i in range(16):
            single_input = input_batch[i:i+1]
            jacobian = torch.autograd.functional.jacobian(model, single_input)
            self.assertEqual(jacobian.shape, (1, 7, 1, 3, 32, 32))
            jacobians.append(jacobian)

        mean_jacobian = torch.mean(torch.stack(jacobians), dim=0)
        for i in range(16):
            self.assertTrue(torch.allclose(jacobians[i], mean_jacobian))

    def test_get_largest_input_gradient(self):
        bs = 64
        input_batch = torch.randn((bs, 3, 32, 32))
        model = torch.nn.Sequential(
            torch.nn.ReLU(),
            layers.ZeroChannelConcatenation(2*bs),
            layers.FirstChannels(2),
            torch.nn.Flatten(),
            layers.FirstChannels(2),
        )
        input_gradient_norms = get_operator_norms_of_input_gradients(
            model, input_batch)
        # Input gradient norm should be 0 exactly when both outputs are 0.

        self.assertEqual(input_gradient_norms.shape, (bs,))
        print(input_gradient_norms)
        self.assertTrue(torch.all(input_gradient_norms >= 0.0))
        self.assertTrue(torch.all(input_gradient_norms <= 1.))
        self.assertAlmostEqual(torch.min(input_gradient_norms).item(), 0.0)
        self.assertAlmostEqual(torch.max(input_gradient_norms).item(), 1.0)

        self.assertAlmostEqual(torch.mean(input_gradient_norms).item(), 0.75,
                               delta=0.1)

    def test_get_largest_input_gradient_on_Lipschitz_model(self):
        bs = 64
        input_batch = torch.randn((bs, 3, 32, 32))
        activation_clss = [
            torch.nn.ReLU,
            lambda w: layers.MaxMin(),
            layers.NActivation,
            lambda w: layers.NActivation(w, theta_init_values=(0, 0)),
        ]
        for activation_cls in activation_clss:
            model = models.SimplifiedConvNet(
                activation_cls=activation_cls,
                conv_cls=layers.AOLConv2d,
                # conv_cls=layers.AOLConv2dOrthogonal,
            )
            input_gradient_norms = get_operator_norms_of_input_gradients(
                model, input_batch)

            self.assertEqual(input_gradient_norms.shape, (bs,))
            print(input_gradient_norms)
            self.assertTrue(torch.all(input_gradient_norms >= 0.0))
            self.assertTrue(torch.all(input_gradient_norms <= 1.))


