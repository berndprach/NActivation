import torch

from dataclasses import dataclass
from typing import Iterable, Callable, List, Any

from src.run_logging import BatchTracker
from src.metrics import Metric


@dataclass
class Trainer:
    model: torch.nn.Module
    train_loader: Iterable
    val_loader: Iterable
    loss_function: Callable
    optimizer: Any
    training_metrics: List[Metric]
    device: torch.device

    def train_epoch(self):
        batch_tracker = BatchTracker(self.training_metrics)
        self.model.train()

        for i, (images, labels) in enumerate(self.train_loader):
            # Move tensors to the configured device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Evaluate the metrics
            batch_tracker.update(outputs, labels)

        train_metrics = batch_tracker.get_average_results(prefix="Train_")
        return train_metrics

    def evaluate(self):
        val_metrics = get_validation_metrics(
            self.model, self.val_loader, self.training_metrics, self.device)
        return val_metrics


@torch.no_grad()
# Decorator caches output from register_parametrization()
@torch.nn.utils.parametrize.cached()
def get_validation_metrics(model, val_loader, used_metrics, device):
    batch_tracker = BatchTracker(used_metrics)
    model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            batch_tracker.update(outputs, labels)

    return batch_tracker.get_average_results(prefix="Val_")
