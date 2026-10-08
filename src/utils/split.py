"""Reproducible IID splits of observation indices, independent of sample values."""

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


ALGORITHM = "iid-pcg64-largest-remainder-v1"
_ROLES = ("train", "validation", "test")


def _allocation(
    n_samples: int, fractions: tuple[float, float, float], seed: int,
) -> tuple[int, int, int]:
    if type(n_samples) is not int or not 0 < n_samples <= np.iinfo(np.int64).max:
        raise ValueError("n_samples must be a positive Python int within int64 range.")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a Python int in [0, 4294967295].")
    if not isinstance(fractions, tuple) or len(fractions) != 3:
        raise ValueError("fractions must be a tuple of three positive finite numbers.")
    if any(type(f) not in (int, float) or f <= 0 or f > 1 or not math.isfinite(f)
           for f in fractions):
        raise ValueError("fractions must contain three positive finite int/float values.")
    if not math.isclose(math.fsum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("fractions must sum to 1 within absolute tolerance 1e-12.")
    # Exact decimal arithmetic avoids binary rounding breaking allocation ties.
    # The accepted near-one sum is corrected only for quota arithmetic; requested
    # fractions are retained verbatim. Invalid fractions are never normalized.
    weights = tuple(Fraction(str(f)) for f in fractions)
    total = sum(weights)
    quotas = tuple(n_samples * weight / total for weight in weights)
    counts = [q.numerator // q.denominator for q in quotas]
    remainder_order = sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in remainder_order[:n_samples - sum(counts)]:
        counts[i] += 1
    if any(count == 0 for count in counts):
        raise ValueError("fractions and n_samples produce an empty train, validation or test split.")
    return tuple(counts)


@dataclass(frozen=True, eq=False)
class SplitIndices:
    """Validated, sorted indices with metadata; arrays are copied and read-only.

    Construction and loading validate coverage and allocation, rather than merely
    trusting persisted metadata. Index identity refers to the original row order.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    n_samples: int
    fractions: tuple[float, float, float]
    seed: int
    algorithm: str = ALGORITHM

    def __post_init__(self) -> None:
        counts = _allocation(self.n_samples, self.fractions, self.seed)
        if self.algorithm != ALGORITHM:
            raise ValueError(f"Unsupported split algorithm: {self.algorithm!r}.")
        arrays = []
        for role, count in zip(_ROLES, counts):
            values = getattr(self, role)
            if np.ma.isMaskedArray(values) or not isinstance(values, np.ndarray):
                raise ValueError(f"{role} indices must be an unmasked NumPy ndarray.")
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError(f"{role} indices must be a one-dimensional integer array.")
            if len(values) != count:
                raise ValueError(f"{role} index count does not match requested allocation.")
            if np.any(values < 0) or np.any(values >= self.n_samples):
                raise ValueError(f"{role} indices must be within [0, n_samples).")
            if np.any(values[1:] <= values[:-1]):
                raise ValueError(f"{role} indices must be sorted and unique.")
            arrays.append(values.astype(np.int64, copy=True))
        combined = np.concatenate(arrays)
        if len(np.unique(combined)) != self.n_samples:
            raise ValueError("Split indices must be disjoint and cover all original samples.")
        for role, values in zip(_ROLES, arrays):
            values.flags.writeable = False
            object.__setattr__(self, role, values)


def make_iid_split(
    n_samples: int,
    *,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int,
) -> SplitIndices:
    """Allocate by largest remainder, then partition one local PCG64 permutation.

    Ties favor train, validation, test, in that order. Returned indices are
    sorted within each role. No data values or global RNG state are consulted.
    """
    counts = _allocation(n_samples, fractions, seed)
    indices = np.random.Generator(np.random.PCG64(seed)).permutation(n_samples)
    train_end = counts[0]
    validation_end = train_end + counts[1]
    return SplitIndices(
        np.sort(indices[:train_end]),
        np.sort(indices[train_end:validation_end]),
        np.sort(indices[validation_end:]),
        n_samples, fractions, seed,
    )


def save_split(split: SplitIndices, path: str | Path) -> None:
    """Save indices to an .npz path and metadata to its .json sidecar.

    Existing files are overwritten; parent directories must already exist.
    This is a two-file format, not an atomic experiment manifest.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Split path must have an .npz suffix.")
    # Revalidate in case a caller deliberately bypassed array write protection.
    checked = SplitIndices(**{field: getattr(split, field) for field in (
        *_ROLES, "n_samples", "fractions", "seed", "algorithm",
    )})
    metadata = {"n_samples": checked.n_samples, "fractions": checked.fractions,
                "seed": checked.seed, "algorithm": checked.algorithm}
    np.savez(path, **{role: getattr(checked, role) for role in _ROLES})
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def load_split(path: str | Path) -> SplitIndices:
    """Load without pickle and validate metadata, counts, ordering and coverage.

    Missing files raise FileNotFoundError; malformed contents raise ValueError.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Split path must have an .npz suffix.")
    try:
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or set(metadata) != {
            "n_samples", "fractions", "seed", "algorithm",
        }:
            raise ValueError("Invalid split metadata keys.")
        if not isinstance(metadata["fractions"], list):
            raise ValueError("Persisted fractions must be a JSON array.")
        metadata["fractions"] = tuple(metadata["fractions"])
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 3 or set(archive.files) != set(_ROLES):
                raise ValueError("Split archive must contain exactly train, validation and test.")
            return SplitIndices(**{role: archive[role] for role in _ROLES}, **metadata)
    except FileNotFoundError:
        raise
    except (ValueError, TypeError, OSError, EOFError, BadZipFile) as exc:
        raise ValueError(f"Invalid persisted split '{path}': {exc}") from exc
