from typing import List

import torch

from src.metrics import Metric


class BatchTracker:
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
        self.batch_results = None

        self.reset()

    def reset(self):
        self.batch_results = {metric.get_name(): [] for metric in self.metrics}

    def update(self, output_batch, label_batch):
        with torch.no_grad():
            for metric in self.metrics:
                batch_aggregated = metric.aggregated(output_batch, label_batch)
                self.batch_results[metric.get_name()].append(batch_aggregated)

    def get_average_results(self, prefix=""):
        return {f"{prefix}{k}": (sum(v) / len(v)).item()
                for k, v in self.batch_results.items()}


