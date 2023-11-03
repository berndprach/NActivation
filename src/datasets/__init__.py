
from torchvision.datasets import CIFAR10, CIFAR100

from src.datasets.tiny_image_net import TinyImageNet
from src.datasets.n_function_dataset import NFunctionDataset

from src.datasets.get_data_loaders import get_image_data_loaders

from src.datasets import transforms


DEFAULT_TRANSFORMATIONS = {
    (CIFAR10, "train"): transforms.CIFAR10_TRAIN,
    (CIFAR10, "test"): transforms.CIFAR10_TEST,
    (CIFAR100, "train"): transforms.CIFAR100_TRAIN,
    (CIFAR100, "test"): transforms.CIFAR100_TEST,
    (TinyImageNet, "train"): transforms.TINY_IN_TRAIN,
    (TinyImageNet, "test"): transforms.TINY_IN_TEST,
}
