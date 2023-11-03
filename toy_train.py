import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import datasets
from src.models import layers

TS_SIZE = 1_000
BATCH_SIZE = 100
WIDTH = 40
LR = 0.01
MOMENTUM = 0.9
NROF_EPOCHS = 1_000

OUTPUT_FOLDER = os.path.join("outputs", "toy")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

HISTORY_FILE_NAME = "histories.txt"
FUNCTION_VALUES_FILE_NAME = "function_values.txt"
FINAL_LOSSES_FILE_NAME = "final_losses.txt"

PLOTTING_INTERVAL = np.linspace(-3, 3, 1000)


def train_toy_experiment(activation_cls, dense_cls):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = datasets.NFunctionDataset(ts_size=TS_SIZE)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = nn.Sequential(
        layers.ZeroChannelConcatenation(concatenate_to=WIDTH),
        dense_cls(WIDTH, WIDTH),
        activation_cls(WIDTH),
        dense_cls(WIDTH, WIDTH),
        activation_cls(WIDTH),
        dense_cls(WIDTH, WIDTH),
        layers.FirstChannels(nrof_channels=1)
    )
    model.to(device)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
    )

    loss_history = []
    model.train()
    for epoch in range(1, 1 + NROF_EPOCHS):
        batch_losses = []

        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)

            output = output.squeeze()
            loss = loss_function(output, y)

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        loss_history.append(epoch_loss)

        if is_power_of_two(epoch):
            print(f"Epoch: {epoch: d}, loss: {epoch_loss}")

    activation_name = activation_cls.__name__
    dense_name = dense_cls.__name__
    names = [activation_name, dense_name]

    save_result_to_file(loss_history, HISTORY_FILE_NAME, *names)

    final_loss = loss_history[-1]
    save_result_to_file(final_loss, FINAL_LOSSES_FILE_NAME, *names)

    xs_line = torch.tensor(PLOTTING_INTERVAL, dtype=torch.float32)[:, None]
    xs_line = xs_line.to(device)
    outputs = model(xs_line)
    outputs_np = outputs.cpu().detach().numpy()
    function_values = list(outputs_np[:, 0])
    save_result_to_file(function_values, FUNCTION_VALUES_FILE_NAME, *names)


def save_result_to_file(result, file_name, activation_name, dense_name):
    file_path = os.path.join(OUTPUT_FOLDER, file_name)
    with open(file_path, "a") as f:
        f.write(f"{activation_name}; {dense_name}; {result}\n")


class MaxMin(layers.MaxMin):
    def __init__(self, width):
        super().__init__()

    def forward(self, x):
        return super().forward(x)


class AbsoluteValue(layers.AbsoluteValue):
    def __init__(self, width):
        super().__init__()

    def forward(self, x):
        return super().forward(x)


def main(run_nr: int):
    settings = {
        0: (torch.nn.ReLU, torch.nn.Linear),
        1: (MaxMin, layers.AOLLinearOrthogonal),
        2: (AbsoluteValue, layers.AOLLinearOrthogonal),
        3: (layers.NActivation, layers.AOLLinearOrthogonal),
    }
    activation_cls, dense_cls = settings[run_nr]

    print(f"Using activation: {activation_cls.__name__}")
    print(f"Using dense: {dense_cls.__name__}")

    train_toy_experiment(activation_cls, dense_cls)


def is_power_of_two(n):
    return (n & (n - 1) == 0) and n != 0


if __name__ == "__main__":
    chosen_run_nr = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Training run: {chosen_run_nr}")

    main(chosen_run_nr)
