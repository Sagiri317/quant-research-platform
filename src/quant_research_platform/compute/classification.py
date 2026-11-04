"""Fixed-threshold direction targets and equal-row three-class probability scores."""

from dataclasses import dataclass, field

import numpy as np


def direction_classes(returns, threshold):
    values = np.asarray(returns)
    return np.where(values < -threshold, 0, np.where(values > threshold, 2, 1))


def classification_metrics(returns, probabilities, threshold):
    result = ClassificationMetrics(threshold)
    result.add(returns, probabilities)
    return result.summary()


@dataclass
class ClassificationMetrics:
    threshold: float
    count: int = 0
    log_loss: float = 0.0
    brier: float = 0.0
    confusion: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype="int64")
    )

    def add(self, returns, probabilities):
        actual = direction_classes(returns, self.threshold)
        predicted = probabilities.argmax(axis=1)
        self.count += len(actual)
        # Bound only the log calculation, without changing the saved probabilities.
        self.log_loss -= np.log(
            np.maximum(
                probabilities[np.arange(len(actual)), actual], np.finfo(float).tiny
            )
        ).sum()
        self.brier += ((probabilities - np.eye(3)[actual]) ** 2).sum()
        self.confusion += np.bincount(3 * actual + predicted, minlength=9).reshape(3, 3)

    def summary(self):
        if not self.count:
            return dict.fromkeys(("log_loss", "brier", "accuracy", "macro_f1"))
        support = self.confusion.sum(axis=0) + self.confusion.sum(axis=1)
        f1 = np.divide(
            2 * self.confusion.diagonal(), support, out=np.zeros(3), where=support > 0
        )
        return {
            "log_loss": float(self.log_loss / self.count),
            "brier": float(self.brier / self.count),
            "accuracy": float(self.confusion.trace() / self.count),
            "macro_f1": float(f1.mean()),
        }
