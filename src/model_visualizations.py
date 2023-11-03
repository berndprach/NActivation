import os
from abc import abstractmethod, ABC
from typing import Dict, Set, Optional, Union

from matplotlib import pyplot as plt

from src.models.util import (
    get_activation_variance,
    activation_variance_to_string,
    plot_activation_variance,
)


class Visualization(ABC):
    log_dir: str = None

    def __init__(self):
        self.visualization_epochs: Union[str, Set[int]] = "all"

    def __call__(self, epoch: int):
        if self.log_dir is None:
            raise ValueError("Visualization.log_dir not set!")

        if self.is_visualization_epoch(epoch):
            self.execute(epoch)

    @abstractmethod
    def execute(self, epoch: int):
        ...

    def is_visualization_epoch(self, epoch: int):
        if self.visualization_epochs is "all":
            return True
        return epoch in self.visualization_epochs

    def get_name(self):
        if hasattr(self, "name"):
            return self.name
        return self.__class__.__name__

    def at_epochs(self, epochs: Set[int]):
        self.visualization_epochs = epochs
        return self

    @staticmethod
    def set_log_dir(log_dir: str):
        Visualization.log_dir = log_dir


class SaveActivationVariances(Visualization):
    def __init__(self, model, input_batch):
        super().__init__()
        self.model = model
        self.input_batch = input_batch
        self.text_path = os.path.join(self.log_dir, "activation_variance.txt")
        self.plot_path = os.path.join(self.log_dir, "activation_variance.png")

    def execute(self, epoch):
        layer_names, activation_variances = get_activation_variance(
            self.model,
            self.input_batch
        )

        act_text = activation_variance_to_string(
            layer_names, activation_variances)
        with open(self.text_path, "w") as f:
            f.write(act_text)

        plot_activation_variance(activation_variances)
        plt.title(f"Activation variance epoch {epoch}")
        plt.savefig(self.plot_path)
        plt.close("all")
