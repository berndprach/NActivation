
from torchvision import transforms

CIFAR10_MEAN = [0.49139968, 0.48215841, 0.44653091]
CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
IMAGENET_MEAN = [0.485, 0.456, 0.406]

CIFAR10_TRAIN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, [1., 1., 1.]),
    transforms.RandomCrop(32, 4),
    transforms.RandomHorizontalFlip(),
])

CIFAR10_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, [1., 1., 1.]),
])

CIFAR100_TRAIN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, [1., 1., 1.]),
    transforms.RandomCrop(32, 4),
    transforms.RandomHorizontalFlip(),
])

CIFAR100_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, [1., 1., 1.]),
])

TINY_IN_TRAIN = transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, [1., 1., 1.]),
    # transforms.RandomCrop(64, 4),
    # transforms.RandomHorizontalFlip(),
])

TINY_IN_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, [1., 1., 1.]),
])

