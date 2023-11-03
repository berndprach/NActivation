import os
import unittest

import torch

from src import models
from src.models import layers

DATA_ROOT = os.path.join("data")


class TestGetActivationVariance(unittest.TestCase):
    def test_get_activation_iterator(self):
        input_batch = torch.randn((16, 3, 32, 32))
        model = torch.nn.Sequential(
            layers.AOLConv2d(3, 3, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        activation_iterator = models.util.get_activation_iterator(
            model, input_batch, include_input=False)

        # activations = list(activation_iterator)
        activations = [a for _, a in activation_iterator]
        print(list(act.shape for act in activations))

        self.assertEqual(len(activations), 3)
        self.assertTrue(torch.allclose(activations[0], input_batch))
        self.assertEqual(activations[1].shape, input_batch.shape)
        self.assertTrue(torch.all(activations[1] >= 0))
        self.assertEqual(activations[2].shape, (16, 3*32*32))

    def test_variances_stay_the_same_for_orthogonal_layers(self):
        input_batch = torch.randn((16, 3, 32, 32))
        input_variance = input_batch.var(dim=0).sum()
        model = torch.nn.Sequential(
            layers.AOLConv2d(3, 3, 3, padding=1),
            layers.ZeroChannelConcatenation(8),
            layers.AOLConv2d(8, 8, 3, padding=1),
            layers.FirstChannels(4),
            layers.NActivation(4, theta_init_values=(0., 0.)),
            torch.nn.Flatten(),
        )

        for name, activation in models.util.get_activation_iterator(
                model, input_batch, include_input=True):
            print(f"{name}: {activation.shape}")
            act_var = activation.var(dim=0).sum()
            self.assertTrue(torch.allclose(act_var, input_variance))

    def test_variance_is_decreasing_throughout_the_model(self):
        input_batch = torch.randn((16, 3, 32, 32))
        model = torch.nn.Sequential(
            layers.AOLConv2dOrthogonal(3, 3, 3, padding=1),
            layers.ZeroChannelConcatenation(8),
            layers.CPLConv2d(8, 8, 3, padding=1),
            layers.MaxMin(),
            layers.FirstChannels(4),
            layers.SOCConv2d(4, 4, 3, padding=1),
            layers.NActivation(4),
            torch.nn.Flatten(),
        )
        prev_act_var = input_batch.var(dim=0).sum()
        for ln, activation in models.util.get_activation_iterator(
                model, input_batch, include_input=True):
            print(ln)
            print(activation.shape)
            act_var = activation.var(dim=0).sum()
            print(act_var)
            if act_var > prev_act_var:
                line = "#"*72
                print(f"\n\n{line}\n"
                      f"# Attention, numeric error!\n"
                      f"# {ln}: "
                      f"Var. (={act_var}) > Prev. Var. (={prev_act_var}) \n"
                      f"{line}\n\n")
            self.assertTrue(act_var/prev_act_var <= 1.01)
            prev_act_var = act_var

    # def test_plot_activation_variance(self):
    #     input_tensor = torch.randn((16, 3, 32, 32))
    #     model = torch.nn.Sequential(
    #         layers.AOLConv2dOrthogonal(3, 3, 3, padding=1),
    #         layers.ZeroChannelConcatenation(8),
    #         layers.CPLConv2d(8, 8, 3, padding=1),
    #         layers.MaxMin(),
    #         layers.FirstChannels(4),
    #         layers.SOCConv2d(4, 4, 3, padding=1),
    #         layers.NActivation(4),
    #         torch.nn.Flatten(),
    #     )
    #     _, activation_variances = models.util.get_activation_variance(
    #         model, input_tensor)
    #     models.util.plot_activation_variance(activation_variances)
    #     plt.show()

    # def test_plot_activation_variance_on_CIFAR10(self):
    #     cifar10_loader, _ = get_image_data_loaders(
    #         dataset_name="CIFAR10",
    #         data_dir=DATA_ROOT,
    #         batch_size=16,
    #     )
    #     input_tensor, _ = next(iter(cifar10_loader))
    #     model = models.SimplifiedConvNet(
    #         activation_cls=lambda w: layers.MaxMin(),
    #         conv_cls=layers.AOLConv2d,
    #     )
    #     out = models.util.get_activation_variance(model, input_tensor)
    #     layer_names, activation_variances = out
    #
    #     print(models.util.activation_variance_to_string(
    #         layer_names, activation_variances))
    #
    #     models.util.plot_activation_variance(activation_variances)
    #     plt.show()

