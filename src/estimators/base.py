"""Common contract for joint discrete entropy estimators."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EntropyEstimator(Protocol):
    """Estimate joint entropy in nats from samples shaped (n,) or (n, d).

    Rows are observations; columns are components in their supplied order.
    The result is one joint entropy, not a sum/mean of marginal entropies.
    Samples must be a NumPy ndarray; lists and masked arrays are rejected.
    """

    def estimate(self, samples: np.ndarray) -> float: ...


def joint_counts(samples: np.ndarray) -> np.ndarray:
    """Count observed rows without enumerating the full joint alphabet.

    Discrete labels must be integers, booleans, or finite integer-valued
    floats in a NumPy ndarray. Negative and non-contiguous labels are supported.
    Lists are rejected to avoid implicit lossy dtype coercion. Masked arrays
    (even with an empty mask), missing values, continuous measurements and
    empty datasets are rejected. Callers must preserve labels when constructing
    their arrays; precision already lost before this call cannot be recovered.
    """
    if np.ma.isMaskedArray(samples):
        raise ValueError("samples must not be a MaskedArray; masked inputs are unsupported.")
    if not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a NumPy ndarray with an explicit, label-preserving dtype.")
    values = np.asarray(samples)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("samples must have non-empty shape (n,) or (n, d).")
    if values.dtype.kind not in "biuf":
        raise ValueError("samples must contain numeric discrete labels.")
    if values.dtype.kind == "f" and (
        not np.isfinite(values).all() or not (values == np.floor(values)).all()
    ):
        raise ValueError("samples must contain finite integer-valued labels.")
    return np.unique(values, axis=0, return_counts=True)[1]
