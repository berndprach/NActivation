import math
import os
import unittest

import torch

from src import datasets

DATA_ROOT = os.path.join("data")
CIFAR10_TRAIN_VAL_SIZE = 50_000
CIFAR10_TEST_SIZE = 10_000


class TestGetDataLoaders(unittest.TestCase):
    def test_cifar10_get_train_val_loaders(self):
        batch_size = 256
        train_loader, val_loader = datasets.get_image_data_loaders(
            dataset_name="CIFAR10",
            data_dir=DATA_ROOT,
            batch_size=batch_size,
            val_proportion=0.1,
            shuffle_in_train_loader=True,
            num_workers=4,
            use_test_set=False,
        )

        decimal_train_length = CIFAR10_TRAIN_VAL_SIZE * 0.9 / batch_size
        expected_train_length = int(math.ceil(decimal_train_length))
        self.assertEqual(len(train_loader), expected_train_length)

        decimal_val_length = CIFAR10_TRAIN_VAL_SIZE * 0.1 / batch_size
        expected_val_length = int(math.ceil(decimal_val_length))
        self.assertEqual(len(val_loader), expected_val_length)

        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        self.assertEqual(train_batch[0].shape, (256, 3, 32, 32))
        self.assertEqual(train_batch[1].shape, (256,))

        self.assertEqual(val_batch[0].shape, (256, 3, 32, 32))
        self.assertEqual(val_batch[1].shape, (256,))

    def test_cifar10_get_train_test_loaders(self):
        batch_size = 256
        full_train_loader, test_loader = datasets.get_image_data_loaders(
            dataset_name="CIFAR10",
            data_dir=DATA_ROOT,
            batch_size=batch_size,
            val_proportion=0.1,
            shuffle_in_train_loader=True,
            num_workers=4,
            use_test_set=True,
        )

        decimal_full_train_length = CIFAR10_TRAIN_VAL_SIZE / batch_size
        expected_train_length = int(math.ceil(decimal_full_train_length))
        self.assertEqual(len(full_train_loader), expected_train_length)

        decimal_test_length = CIFAR10_TEST_SIZE / batch_size
        expected_test_length = int(math.ceil(decimal_test_length))
        self.assertEqual(len(test_loader), expected_test_length)

        train_batch = next(iter(full_train_loader))
        test_batch = next(iter(test_loader))

        self.assertEqual(train_batch[0].shape, (256, 3, 32, 32))
        self.assertEqual(train_batch[1].shape, (256,))

        self.assertEqual(test_batch[0].shape, (256, 3, 32, 32))
        self.assertEqual(test_batch[1].shape, (256,))

    def test_n_function_train(self):
        batch_size = 256
        ts_size = 1000
        dataset = datasets.NFunctionDataset(ts_size=ts_size)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        decimal_test_length = ts_size / batch_size
        expected_test_length = int(math.ceil(decimal_test_length))
        self.assertEqual(len(data_loader), expected_test_length)

        x_batch, y_batch = next(iter(data_loader))

        self.assertEqual(list(x_batch.shape), [batch_size, 1])
        self.assertEqual(list(y_batch.shape), [batch_size])

