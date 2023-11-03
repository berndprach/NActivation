
import numpy as np
import torch

from torch.utils.data import TensorDataset


def n_function(x):
    if x <= -1 / 2:
        return x + 1
    elif x <= 1 / 2:
        return -x
    else:
        return x - 1


class NFunctionDataset(TensorDataset):
    def __init__(self, ts_size: int):

        x_train = np.random.uniform(-3, 3, size=[ts_size, 1])
        y_train = np.array([n_function(x) for x in x_train[:, 0]])

        super().__init__(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).float(),
        )


