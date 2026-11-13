"""Ablation-only data controls; no change to validated generators or split APIs."""
from dataclasses import replace
import hashlib
import math

import numpy as np

from src.utils.categorical import CategoricalBatch
from src.utils.reproducibility import validate_seed


def nested_training_subset(train, fraction, *, seed):
    """One value-independent permutation, nested prefixes, original row identities."""
    validate_seed(seed)
    if not isinstance(train, CategoricalBatch) or train.role != 'train':
        raise ValueError('A training batch is required.')
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Training fraction must lie in (0,1].')
    count = math.floor(len(train.values) * fraction)
    if count < 1:
        raise ValueError('Empty training subset.')
    positions = np.sort(np.random.default_rng(seed).permutation(len(train.values))[:count])
    return replace(train, values=train.values[positions], row_indices=train.row_indices[positions])


def discretize_uniform(values, states):
    """Fixed analytic equal-width bins of independent Uniform[0,1) coordinates.

    Boundaries j/K belong to the bin on their right. No fitted thresholds, no
    merging of unobserved categories. Each K defines a different estimand.
    """
    if type(states) is not int or states not in (2, 3, 5):
        raise ValueError('Only binary, ternary and five-state bins are frozen.')
    if (not isinstance(values, np.ndarray) or values.ndim != 2 or values.dtype.kind != 'f'
            or not values.size or not np.isfinite(values).all() or np.any(values < 0) or np.any(values >= 1)):
        raise ValueError('Expected finite nonempty float matrix in [0,1).')
    return np.floor(values * states).astype(np.int64)


def discretized_entropy(d, states):
    if type(d) is not int or d < 1 or type(states) is not int or states not in (2, 3, 5):
        raise ValueError('Invalid dimension or alphabet.')
    return d * math.log(states)


def array_digest(values):
    """Hash shape, dtype and canonical row-major bytes, not just numerical values."""
    payload = f'{values.shape}:{values.dtype.str}:'.encode() + values.tobytes(order='C')
    return hashlib.sha256(payload).hexdigest()
