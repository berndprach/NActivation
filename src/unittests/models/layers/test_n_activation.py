import unittest

import torch

from src.datasets.n_function_dataset import NFunctionDataset
from src.models.layers import NActivation
from src.models.layers.activations.n_activation import (
    ThetaInitializer,
    RandomThetaInitializer,
)


class TestNActivation(unittest.TestCase):
    def test_theta_initialization(self):
        init_values = (-0.1, 0.1)
        initializer = ThetaInitializer(init_values)
        theta = initializer((3, 2), device="cpu")
        print(theta)
        goal_theta = torch.Tensor([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]])
        self.assertTrue(torch.allclose(theta, goal_theta))

        init_values = (-1., 1., -2., 2.)
        initializer = ThetaInitializer(init_values)
        theta = initializer((4, 2), device="cpu")
        print(theta)
        goal_theta = torch.Tensor([[-1., 1.], [-2., 2.], [-1., 1.], [-2., 2.]])
        self.assertTrue(torch.allclose(theta, goal_theta))

        init_values = (-2., 1)
        initializer = ThetaInitializer(init_values)
        theta = initializer((4, 2), device="cpu")
        print(theta)
        goal_tensor = torch.Tensor([[-2., 1], [-2., 1], [-2., 1], [-2., 1]])
        self.assertTrue(torch.allclose(theta, goal_tensor))

        init_values = (0., 0., -1., 0.)
        initializer = ThetaInitializer(init_values)
        theta = initializer((4, 2), device="cpu")
        print(theta)
        goal_tensor = torch.Tensor([[0., 0.], [-1., 0.], [0., 0.], [-1., 0.]])
        self.assertTrue(torch.allclose(theta, goal_tensor))

    def test_log_uniform_theta_initialization(self):
        log_interval = (-5., 0.)
        initializer = RandomThetaInitializer(log_interval)
        theta = initializer((5, 2), device="cpu")
        print(theta)
        theta_abs = torch.abs(theta)
        lower_bound, upper_bound = (10 ** log_val for log_val in log_interval)
        self.assertTrue(torch.all(torch.less(theta_abs, upper_bound)))
        self.assertTrue(torch.all(torch.greater(theta_abs, lower_bound)))

    def test_n_activation_linear(self):
        n_act = NActivation(1, theta_init_values=(-1., 1.))
        input_tensor = torch.tensor([-3., -2., -1., 0., 1., 2., 3.])[None, :]
        output_tensor = n_act(input_tensor)
        print(output_tensor)
        goal_output = torch.tensor([-1, 0., 1., 0., -1., 0., 1.])[None, :]
        self.assertEqual(output_tensor.shape, input_tensor.shape)
        self.assertTrue(torch.allclose(output_tensor, goal_output))

    def test_n_activation_for_4d_input(self):
        n_act = NActivation(3, theta_init_values=(-1., -1.))
        input_tensor = torch.randn(2, 3, 8, 8)
        output_tensor = n_act(input_tensor)
        # print(output_tensor)
        self.assertEqual(output_tensor.shape, input_tensor.shape)
        self.assertTrue(torch.allclose(output_tensor, input_tensor + 2.))

        n_act = NActivation(16, theta_init_values=(-1., 1.))
        input_tensor = torch.rand(8, 16, 2, 2)
        output_tensor = n_act(input_tensor)
        # print(output_tensor)
        self.assertEqual(output_tensor.shape, input_tensor.shape)
        self.assertTrue(torch.allclose(output_tensor, -input_tensor))

    # def test_plot_n_activation(self):
    #     n_act = NActivation(1, theta_init_values=(-1., 1.))
    #     input_tensor = torch.linspace(-3., 3., 100)[:, None]
    #     output_tensor = n_act(input_tensor)
    #     plt.plot(input_tensor[:, 0].detach(), output_tensor[:, 0].detach())
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     plt.show()

    # def test_plot_derivative(self):
    #     n_act = NActivation(1, theta_init_values=(-1., 1.))
    #     input_tensor = torch.linspace(-2., 2., 100)[:, None]
    #     input_tensor.requires_grad = True
    #     output_tensor = n_act(input_tensor)
    #     out_sum = output_tensor.sum()
    #     out_sum.backward()
    #
    #     plt.title("Input gradient")
    #     plt.plot(input_tensor[:, 0].detach(),
    #              input_tensor.grad[:, 0].detach())
    #     plt.show()
    #
    #     # Gradient wrt theta:
    #     theta_grad = n_act.theta.grad
    #     print(theta_grad)

    def test_n_activation_with_unordered_theta(self):
        input_tensor_2d = torch.randn(32, 16)
        input_tensor_4d = torch.randn(8, 16, 2, 2)

        n_act1 = NActivation(16, theta_init_values=(0., -1., 2., 0.))
        n_act2 = NActivation(16, theta_init_values=(-1., 0., 0., 2.))

        output_tensor1_2d = n_act1(input_tensor_2d)
        output_tensor2_2d = n_act2(input_tensor_2d)

        self.assertTrue(output_tensor1_2d.shape == output_tensor2_2d.shape)
        self.assertTrue(torch.allclose(output_tensor1_2d, output_tensor2_2d))

        output_tensor1_4d = n_act1(input_tensor_4d)
        output_tensor2_4d = n_act2(input_tensor_4d)

        self.assertTrue(output_tensor1_4d.shape == output_tensor2_4d.shape)
        self.assertTrue(torch.allclose(output_tensor1_4d, output_tensor2_4d))

    def test_fitting(self):
        print("\n\nTesting fitting with the NActivation.\n")
        dataset = NFunctionDataset(1000)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        model = torch.nn.Sequential(NActivation(1, theta_init_values=(0., 0.)))

        # print(next(model.parameters()).shape)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(10):
            for x, y in loader:
                optimizer.zero_grad()
                y_pred = model(x)
                y_pred = y_pred.squeeze()
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch}: {loss.item()}")
        self.assertAlmostEqual(loss.item(), 0.)

    # def test_plot_random_initialized_n_activation(self):
    #     # n_act = RandomLogUniformNActivation(16)
    #     n_act = RandomNormalNActivation(16)
    #
    #     input_tensor = torch.linspace(-1., 1., 100)[:, None, None, None]
    #     output_tensor = n_act(input_tensor)
    #
    #     # Plot in subplots:
    #     fig, axs = plt.subplots(4, 4)
    #     plt.suptitle("Randomly initialized NActivation")
    #
    #     axs_flat = axs.flatten()
    #     for i in range(16):
    #         ax = axs_flat[i]
    #         ax.plot(
    #             input_tensor[:, 0, 0, 0].detach(),
    #             output_tensor[:, i, 0, 0].detach(),
    #             alpha=0.5,
    #         )
    #
    #         ax.axhline(0, color="black", alpha=0.5)
    #         ax.axvline(0, color="black", alpha=0.5)
    #
    #         ax.set_aspect('equal', adjustable='box')
    #
    #     plt.show()

    def test_ellipsis_works_as_expected(self):
        test_tensor = torch.rand(8, 2)
        for _ in range(2):
            test_tensor = test_tensor[..., None]

        print(test_tensor.shape)
        self.assertEqual(test_tensor.shape, (8, 2, 1, 1))

    def test_lr_factor(self):
        input_tensor = torch.randn(8, 16, 2, 2)
        theta_init_values = (-1., 0.1, -0.1, 0.1)
        lr_factor = 10.

        n_act_1 = NActivation(
            16,
            theta_init_values=theta_init_values,
            lr_factor=1.,
        )
        outputs_1 = n_act_1(input_tensor)
        loss_1 = outputs_1.sum()
        loss_1.backward()
        grad_1 = n_act_1.theta.grad
        # print(grad_1)
        ratio_1 = grad_1 / n_act_1.theta
        print(ratio_1)

        n_act_2 = NActivation(
            16,
            theta_init_values=theta_init_values,
            lr_factor=lr_factor,
        )
        outputs_2 = n_act_2(input_tensor)
        loss_2 = outputs_2.sum()
        loss_2.backward()
        grad_2 = n_act_2.theta.grad
        # print(grad_2)
        ratio_2 = grad_2 / n_act_2.theta
        print(ratio_2)

        self.assertTrue(torch.allclose(outputs_1, outputs_2, atol=1e-3))

        # self.assertTrue(torch.allclose(grad_1, grad_2 * 10.))
        self.assertTrue(torch.allclose(ratio_1 * lr_factor, ratio_2))
