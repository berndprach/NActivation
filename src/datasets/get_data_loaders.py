from typing import Protocol, Optional

import numpy as np
import torch.utils.data
import torchvision.transforms
from torch.utils.data import SubsetRandomSampler

from src import datasets

DataLoader = torch.utils.data.DataLoader
Transform = torchvision.transforms.Compose


class Dataset(Protocol):
    def __init__(self,
                 data_dir: str,
                 train: bool,
                 download: bool,
                 transform: Transform) -> None: ...

    def __len__(self) -> int: ...


def get_image_data_loaders(dataset_name: str,
                           data_dir: str,
                           batch_size: int,
                           use_test_set: bool = False,
                           val_proportion: float = 0.1,
                           shuffle_in_train_loader: bool = True,
                           num_workers: int = 0,
                           ) -> (DataLoader, DataLoader):
    dataset_cls: type[Dataset] = getattr(datasets, dataset_name)

    train_transform = datasets.DEFAULT_TRANSFORMATIONS[(dataset_cls, "train")]
    test_transform = datasets.DEFAULT_TRANSFORMATIONS[(dataset_cls, "test")]

    train_dataset = dataset_cls(
        data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    use_validation_set = not use_test_set
    test_dataset = dataset_cls(
        data_dir,
        train=use_validation_set,
        download=True,
        transform=test_transform
    )
    if use_test_set:
        train_kwarg = {"shuffle": shuffle_in_train_loader}
        test_kwarg = {"shuffle": False}
    else:
        assert len(train_dataset) == len(test_dataset)
        train_sampler, test_sampler = get_subset_samplers(
            total_size=len(train_dataset),
            val_proportion=val_proportion,
        )
        train_kwarg = {"sampler": train_sampler}
        test_kwarg = {"sampler": test_sampler}

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **train_kwarg,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **test_kwarg,
    )
    return train_loader, test_loader


def get_subset_samplers(total_size, val_proportion, shuffle=True):
    indices = list(range(total_size))
    split = int(np.floor(val_proportion * total_size))

    if shuffle:
        np.random.seed(1111)
        np.random.shuffle(indices)

    valid_idx, train_idx = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, valid_sampler
