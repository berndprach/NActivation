
import torch
import torch.nn as nn

from typing import Tuple, Callable


def n_activation(x: torch.Tensor, theta: torch.Tensor):
    # x.shape e.g. [bs, c, h, w] or [bs, c].
    # theta.shape [c, 2].

    theta_sorted, _ = torch.sort(theta, dim=1)
    for _ in range(len(x.shape) - 2):
        theta_sorted = theta_sorted[..., None]

    line1 = x - 2 * theta_sorted[:, 0]
    line2 = -x
    line3 = x - 2 * theta_sorted[:, 1]

    piece1 = line1
    piece2 = torch.where(
        torch.less(x, theta_sorted[:, 0]),
        piece1,
        line2,
    )
    piece3 = torch.where(
        torch.less(x, theta_sorted[:, 1]),
        piece2,
        line3,
    )

    result = piece3
    return result


class BaseNActivation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 initializer: Callable,
                 trainable: bool = True,
                 lr_factor: float = 1.,  # Changes grad/theta ratio.
                 ):
        super().__init__()

        self.sqrt_lr_factor = lr_factor ** 0.5

        theta_init_values = initializer(shape=(in_channels, 2), device=None)
        theta_init_values = theta_init_values / self.sqrt_lr_factor
        self.theta = nn.Parameter(theta_init_values, requires_grad=trainable)

    def forward(self, x: torch.Tensor):
        # x.shape e.g. [bs, c, h, w] or [bs, c].
        theta = self.theta * self.sqrt_lr_factor
        return n_activation(x, theta)


class NActivation(BaseNActivation):
    def __init__(self,
                 *args,
                 theta_init_values: Tuple = (-1., 0., 0., 0.),
                 **kwargs,
                 ):

        initializer = ThetaInitializer(theta_init_values)
        super().__init__(*args, initializer=initializer, **kwargs)


class RandomLogUniformNActivation(BaseNActivation):
    def __init__(self,
                 *args,
                 log_interval: Tuple[float, float] = (-5., 0.),  # log10
                 base: float = 10.,
                 **kwargs,
                 ):

        initializer = RandomThetaInitializer(log_interval, base)
        super().__init__(*args, initializer=initializer, **kwargs)


class RandomNormalNActivation(BaseNActivation):
    def __init__(self,
                 *args,
                 init_mean: float = 0.,
                 init_std: float = 1.,
                 **kwargs,
                 ):
        initializer = GaussianInitializer(init_mean, init_std)
        super().__init__(*args, initializer=initializer, **kwargs)


class ThetaInitializer:
    def __init__(self, values: Tuple):
        self.values = values

    def __call__(self, shape, device):
        assert shape[1] == 2, "The second dimension must be 2."

        theta_values = torch.ones(shape, device=device)
        theta_values = theta_values.reshape(-1, len(self.values))
        theta_values = theta_values * torch.tensor(self.values, device=device)
        theta_values = theta_values.reshape(shape)
        return theta_values


class RandomThetaInitializer:
    def __init__(self,
                 log_interval: Tuple[float, float] = (-5., 0.),
                 base: float = 10.
                 ):
        self.log_interval = log_interval
        self.base = base

    def __call__(self, shape, device):
        uniform01 = torch.rand(shape, device=device)
        l, h = self.log_interval
        log_theta_init = l + (h - l) * uniform01
        unsigned_theta_init = self.base ** log_theta_init
        signs = torch.tensor([-1., 1.], device=device)
        theta_values = unsigned_theta_init * signs
        return theta_values


class GaussianInitializer:
    def __init__(self, mean: float = 0., std: float = 1.):
        self.mean = mean
        self.std = std

    def __call__(self, shape, device):
        rand_gaussian = torch.randn(shape, device=device)
        theta_init = rand_gaussian * self.std + self.mean
        return theta_init
