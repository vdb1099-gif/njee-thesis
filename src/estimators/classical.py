"""Classical estimators using observed joint configurations."""

import numpy as np

from src.estimators.base import joint_counts


def _entropy_from_counts(counts: np.ndarray) -> float:
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


class PlugInEstimator:
    """Empirical joint entropy: -sum(p_hat * ln(p_hat)), in nats."""

    def estimate(self, samples: np.ndarray) -> float:
        return _entropy_from_counts(joint_counts(samples))


class MillerMadowEstimator:
    """Plug-in plus (K_observed - 1)/(2*n), in nats.

    K_observed counts observed joint states, not the theoretical alphabet.
    No clipping is applied: the correction can exceed the true entropy or
    the log of the alphabet size at finite sample sizes.
    """

    def estimate(self, samples: np.ndarray) -> float:
        counts = joint_counts(samples)
        return _entropy_from_counts(counts) + float(
            (counts.size - 1) / (2 * counts.sum())
        )
