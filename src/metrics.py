
from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor


class Metric(ABC):
    print_as_percentage: bool = False
    aggregation: Callable[[Tensor], Tensor] = torch.mean

    @abstractmethod
    def __call__(self,
                 prediction_batch: Tensor,
                 label_batch: Tensor,  # labels not one-hot
                 ) -> Tensor:  # not aggregated
        raise NotImplementedError

    def aggregated(self, prediction_batch, label_batch) -> Tensor:
        return self.aggregation(self(prediction_batch, label_batch))

    def get_name(self):
        if hasattr(self, "name"):
            return self.name
        return self.__class__.__name__


class Accuracy(Metric):
    print_as_percentage = True

    def __call__(self, prediction_batch, label_batch):
        return (prediction_batch.argmax(dim=1) == label_batch).float()


class BatchVariance(Metric):
    def __call__(self, prediction_batch, label_batch) -> Tensor:
        return prediction_batch.var(dim=0, correction=0).sum()


class OffsetCrossEntropyFromScores(Metric):
    def __init__(self, offset=0., temperature=1.):
        super().__init__()
        self.offset = offset
        self.temperature = temperature
        self.name = f"{self.offset:.2g}-OffsetXent"

    def __call__(self, score_batch, label_batch):
        label_batch_oh = torch.nn.functional.one_hot(
            label_batch.to(torch.int64),
            num_classes=score_batch.shape[-1]
        )
        offset_scores = score_batch - self.offset * label_batch_oh
        offset_scores /= self.temperature
        return torch.nn.functional.cross_entropy(
            offset_scores,
            label_batch.to(torch.int64),
        ) * self.temperature


class Margin(Metric):
    def __call__(self, score_batch, label_batch):
        label_batch_oh = torch.nn.functional.one_hot(
            label_batch.to(torch.int64),
            num_classes=score_batch.shape[-1]
        )
        true_score = (score_batch * label_batch_oh).sum(dim=-1)
        best_other = (score_batch - label_batch_oh * 1e6).max(dim=-1)[0]
        return true_score - best_other


class CertifiedRobustAccuracyFromScores(Metric):
    def __init__(self, maximal_perturbation, rescaling_factor=2. ** (1 / 2)):
        self.name = f"CRA{maximal_perturbation:.2f}"
        self.threshold = maximal_perturbation * rescaling_factor
        super().__init__()

    def __call__(self, score_batch, label_batch):
        label_batch_oh = torch.nn.functional.one_hot(
            label_batch.to(torch.int64),
            num_classes=score_batch.shape[-1]
        )
        penalized_scores = score_batch - self.threshold * label_batch_oh
        return (penalized_scores.argmax(dim=1) == label_batch).float()


CRAFromScores = CertifiedRobustAccuracyFromScores



