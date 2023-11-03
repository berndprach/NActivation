
import unittest

import torch

from src import metrics


SCORES = torch.Tensor([
    [0.1, 0.9],
    [0.1, 0.9],
    [0.6, 0.4],
])
LABELS = torch.Tensor([0, 1, 1])


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        metric = metrics.Accuracy()
        value = metric.aggregated(SCORES, LABELS)
        print(value)
        self.assertAlmostEqual(float(value), 1 / 3)

    def test_batch_variance(self):
        metric = metrics.BatchVariance()
        value = metric(SCORES, LABELS)
        print(value)
        goal1 = (0.1**2 + 0.1**2 + 0.6**2) / 3 - (0.1 + 0.1 + 0.6)**2 / 9
        goal2 = (0.9**2 + 0.9**2 + 0.4**2) / 3 - (0.9 + 0.9 + 0.4)**2 / 9
        # self.assertAlmostEqual(float(value), goal1/2 + goal2/2)
        self.assertAlmostEqual(float(value), goal1 + goal2)

    def test_offset_cross_entropy(self):
        # Huge Offset:
        offset = 100
        metric = metrics.OffsetCrossEntropyFromScores(offset=offset)
        value = metric(SCORES, LABELS)
        print(value)
        self.assertAlmostEqual(float(value), offset, places=0)

        # Huge negative Offset:
        metric = metrics.OffsetCrossEntropyFromScores(offset=-100)
        value = metric(SCORES, LABELS)
        print(value)
        self.assertAlmostEqual(float(value), 0.)

        # Huge Temperature:
        metric = metrics.OffsetCrossEntropyFromScores(temperature=1/100)
        value = metric(SCORES, LABELS)
        expected_margins = [
            0.1 - 0.9,
            0.9 - 0.1,
            0.4 - 0.6,
        ]
        expected_hinges = [max(-m, 0.) for m in expected_margins]
        expected_loss = sum(expected_hinges) / 3

        print(value)
        self.assertAlmostEqual(float(value), expected_loss)

    def test_margin(self):
        metric = metrics.Margin()
        value = metric.aggregated(SCORES, LABELS)
        print(value)
        self.assertAlmostEqual(float(value), (-0.8 + 0.8 -0.2) / 3)

    def test_cra_from_scores(self):
        metric = metrics.CRAFromScores(1.)
        value = metric.aggregated(SCORES, LABELS)
        self.assertAlmostEqual(float(value), 0.)

        metric = metrics.CRAFromScores(0.1)
        value = metric.aggregated(SCORES, LABELS)
        self.assertAlmostEqual(float(value), 1 / 3)

        metric = metrics.CRAFromScores(-0.2)
        value = metric.aggregated(SCORES, LABELS)
        self.assertAlmostEqual(float(value), 2 / 3)


if __name__ == '__main__':
    unittest.main()
