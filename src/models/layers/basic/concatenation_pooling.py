from typing import Tuple

from torch import nn, Tensor


class ConcatenationPooling2D(nn.Module):
    """
    This layer is similar to pytorch's PixelUnshuffle,
    but it concatenates the channels instead of interleaving them.
    More specifically, the output at a given position is
    output[: :, i, j] = [
        input[:, 0, 2*i, 2*j], ... input[:, c-1, 2*i, 2*j],
        input[:, 0, 2*i, 2*j+1], ... input[:, c-1, 2*i, 2*j+1],
        ... ]
    """
    def __init__(self, window_shape: Tuple[int, int]):
        super().__init__()
        self.window_shape = window_shape

    def forward(self, x: Tensor) -> Tensor:
        # x.shape = (batch_size, channels, height, width)

        bs, c, h, w = x.shape
        pool_h, pool_w = self.window_shape
        new_h, new_w = h // pool_h, w // pool_w

        x = x.view(bs, c, new_h, pool_h, new_w, pool_w)
        x = x.permute(0, 3, 5, 1, 2, 4)  # bs, pool_h, pool_w, c, new_h, new_w
        return x.reshape(bs, c * pool_h * pool_w, new_h, new_w)

