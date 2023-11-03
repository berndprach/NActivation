import os
import unittest

import torch
from torch import nn

from src.train_model import train_model
from src.default_settings import DefaultSettings

DATA_ROOT = os.path.join("data")


class TestTrainModel(unittest.TestCase):
    def test_train_model(self):
        if not torch.cuda.is_available():
            print("Skipping train_model test on cpu.")
            return

        settings = DefaultSettings()
        settings.data_dir = DATA_ROOT
        settings.nrof_epochs = 2
        train_model(settings)

    def test_plot_learning_rate_from_scheduler(self):
        dummy_model = nn.Sequential(nn.Linear(32, 32))
        weight_decay = 10 ** -3
        momentum = 0.9
        learning_rate = 10 ** -1

        optimizer = torch.optim.SGD(
            dummy_model.parameters(),
            lr=0.,  # Learning rate is determined by the scheduler (below).
            weight_decay=weight_decay,
            momentum=momentum,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=200,
            steps_per_epoch=1
        )

        observed_learning_rates = []
        for epoch in range(200):
            learning_rate = scheduler.get_last_lr()[0]
            # print(epoch, learning_rate)
            observed_learning_rates.append(learning_rate)
            optimizer.step()
            scheduler.step()

        import matplotlib.pyplot as plt
        plt.title("Learning rates throughout training")
        plt.plot(observed_learning_rates)
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.grid()

        filename = "output_test_plot_learning_rate_from_scheduler.png"
        filepath = os.path.join("src", "unittests", filename)

        try:
            plt.savefig(filepath)
            # plt.show()
            plt.close("all")
        except FileNotFoundError:
            print(f"Could not save learning rate plot to {filepath}.")
