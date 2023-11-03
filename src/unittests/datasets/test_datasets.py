import os
import sys
import unittest

from src.datasets import CIFAR10, CIFAR100, TinyImageNet, NFunctionDataset
from src.datasets.n_function_dataset import n_function

DATA_ROOT = os.path.join("data")


class TestDatasets(unittest.TestCase):
    """
    Expected to be run from parent folder of src (so data_root is correct).
    """
    def test_cifar_10(self):
        train_dataset = CIFAR10(
            root=DATA_ROOT,
            train=True,
            download=True)
        self.assertEqual(len(train_dataset), 50000)
        print(train_dataset)

        test_dataset = CIFAR10(
            root=DATA_ROOT,
            train=False,
            download=True)
        self.assertEqual(len(test_dataset), 10000)
        print(test_dataset)

    def test_cifar_100(self):
        train_dataset = CIFAR100(
            root=DATA_ROOT,
            train=True,
            download=True)
        self.assertEqual(len(train_dataset), 50000)
        print(train_dataset)

        test_dataset = CIFAR100(
            root=DATA_ROOT,
            train=False,
            download=True)
        self.assertEqual(len(test_dataset), 10000)
        print(test_dataset)

    def test_tiny_image_net(self):
        if sys.platform.startswith("win"):
            print("Skipping TinyImageNet test on Windows.")
            return

        train_dataset = TinyImageNet(
            root=DATA_ROOT,
            train=True,
            download=True)
        self.assertEqual(len(train_dataset), 100000)
        print(train_dataset)

        test_dataset = TinyImageNet(
            root=DATA_ROOT,
            train=False,
            download=True)
        self.assertEqual(len(test_dataset), 50 * 200)
        print(test_dataset)

    def test_n_function(self):
        input_values = [-2., -1., 0., 1., 2., 3., 4.]
        goal_outputs = [-1., 0., 0., 0., 1., 2., 3.]
        output_values = [n_function(x) for x in input_values]

        for output, goal in zip(output_values, goal_outputs):
            self.assertAlmostEqual(output, goal)

    def test_n_function_dataset(self):
        train_dataset = NFunctionDataset(ts_size=1000)
        self.assertEqual(len(train_dataset), 1000)
        print(train_dataset)

    # def test_plot_n_function(self):
    #     # input_tensor = np.arange(-1., 1., 0.001)
    #     input_tensor = np.arange(-2., 2., 0.001)
    #     output_tensor = [n_function(x) for x in input_tensor]
    #     # Plot:
    #     plt.rcParams.update({'font.size': 22})
    #     plt.title("N Function")
    #     plt.plot(input_tensor, output_tensor)
    #     plt.axis("scaled")
    #     plt.axhline(y=0, color="black")
    #     plt.axvline(x=0, color="black")
    #     plt.xlabel("$x$")
    #     plt.ylabel("$\mathcal{N}(x)$")
    #     plt.show()


if __name__ == '__main__':
    unittest.main()

