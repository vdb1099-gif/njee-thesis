"""Categorical observations and train-only encodings for future NJEE models.

Identity metadata never enters encoded features. Declared alphabets are domain
knowledge supplied by the caller before validation/test inspection; this module
cannot authenticate that provenance or caller-assigned dataset/split identities.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from src.utils.split import SplitIndices


VariableId = str | int
_ROLES = ("train", "validation", "test")


def _values(values: np.ndarray) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        raise ValueError("MaskedArray categorical data are unsupported.")
    if not isinstance(values, np.ndarray):
        raise TypeError("Categorical data must be a NumPy ndarray preserving label dtype.")
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("Categorical data must have nonempty shape (n,) or (n, d).")
    if values.dtype.kind not in "biuf":
        raise ValueError("Categorical data must contain numeric discrete labels.")
    if values.dtype.kind == "f" and (
        not np.isfinite(values).all() or not (values == np.floor(values)).all()
    ):
        raise ValueError("Categorical floats must be finite and exactly integral.")
    result = values.copy()
    result.flags.writeable = False
    return result


def _identity(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _ids(ids: tuple[VariableId, ...]) -> None:
    if not isinstance(ids, tuple) or any(
        type(v) not in (str, int) or (isinstance(v, str) and not v.strip()) for v in ids
    ):
        raise ValueError("Variable identifiers must be a tuple of nonempty strings or Python ints.")
    if len(set(ids)) != len(ids):
        raise ValueError("Variable identifiers must be unique.")


def validate_variable_order(
    order: tuple[VariableId, ...], variable_ids: tuple[VariableId, ...],
) -> tuple[VariableId, ...]:
    """Validate a complete permutation of a supplied target-variable group.

    The group may be a subset of dataset columns, leaving separate context
    variables available for future C-NJEE. Identifiers are never renamed.
    """
    _ids(variable_ids)
    _ids(order)
    if not variable_ids or set(order) != set(variable_ids):
        raise ValueError("Variable order must contain every target identifier exactly once.")
    return order


@dataclass(frozen=True, eq=False)
class CategoricalDataset:
    """Owned copy of raw values in original row and column order."""

    values: np.ndarray
    dataset_id: str
    variable_ids: tuple[VariableId, ...]

    def __post_init__(self) -> None:
        _identity(self.dataset_id, "dataset_id")
        _ids(self.variable_ids)
        values = _values(self.values)
        if len(self.variable_ids) != values.shape[1]:
            raise ValueError("variable_ids must match the number of columns.")
        object.__setattr__(self, "values", values)

    def batch(self, split: SplitIndices, role: str, *, split_id: str) -> "CategoricalBatch":
        """Select one role without examining other partitions' category values."""
        if role not in _ROLES:
            raise ValueError("role must be train, validation or test.")
        if not isinstance(split, SplitIndices):
            raise TypeError("split must be SplitIndices.")
        # Validate again if a caller bypassed the split arrays' write protection.
        checked = SplitIndices(split.train, split.validation, split.test,
                               split.n_samples, split.fractions, split.seed, split.algorithm)
        if checked.n_samples != len(self.values):
            raise ValueError("Split sample count does not match the dataset.")
        indices = getattr(checked, role)
        return CategoricalBatch(self.values[indices], self.dataset_id, self.variable_ids,
                                indices, role, split_id, checked.n_samples)


@dataclass(frozen=True, eq=False)
class CategoricalBatch:
    """Raw partition with original row IDs; direct construction trusts role labels."""

    values: np.ndarray
    dataset_id: str
    variable_ids: tuple[VariableId, ...]
    row_indices: np.ndarray
    role: str
    split_id: str
    n_samples: int

    def __post_init__(self) -> None:
        _identity(self.dataset_id, "dataset_id")
        _identity(self.split_id, "split_id")
        _ids(self.variable_ids)
        if self.role not in _ROLES:
            raise ValueError("role must be train, validation or test.")
        if type(self.n_samples) is not int or not 0 < self.n_samples <= np.iinfo(np.int64).max:
            raise ValueError("n_samples must be a positive Python int within int64 range.")
        values = _values(self.values)
        if len(self.variable_ids) != values.shape[1]:
            raise ValueError("variable_ids must match the number of columns.")
        rows = self.row_indices
        if (np.ma.isMaskedArray(rows) or not isinstance(rows, np.ndarray)
                or rows.ndim != 1 or rows.dtype.kind not in "iu"):
            raise ValueError("row_indices must be an unmasked one-dimensional integer ndarray.")
        if len(rows) != len(values) or np.any(rows < 0) or np.any(rows >= self.n_samples):
            raise ValueError("row_indices must match batch rows and stay within dataset bounds.")
        if np.unique(rows).size != len(rows):
            raise ValueError("row_indices must be unique.")
        rows = rows.astype(np.int64, copy=True)
        rows.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "row_indices", rows)


class CategoricalEncoder:
    """Frozen mappings fitted from one training batch; no refit or partial fit.

    Labels are keyed by exact Python integers, including integral float values.
    Inverse mappings retain the fitted alphabet dtype. Limits are explicit safety
    budgets, not scientific category-merging policies. No categories are dropped.
    """

    @classmethod
    def fit(
        cls, train: CategoricalBatch, *, policy: str = "train_observed",
        alphabets: Mapping[VariableId, np.ndarray] | None = None,
        max_cardinality: int = 100_000,
    ) -> "CategoricalEncoder":
        """Create an encoder using training values only.

        Declared alphabets must be nonempty 1D numeric arrays, one per variable.
        Both policies sort categories numerically for deterministic class IDs.
        Declared duplicates are errors; absent declared categories remain valid.
        """
        if not isinstance(train, CategoricalBatch):
            raise TypeError("fit requires a CategoricalBatch.")
        if train.role != "train":
            raise ValueError("Categorical mappings may only be fitted on the train role.")
        if policy not in ("declared", "train_observed"):
            raise ValueError("policy must be declared or train_observed.")
        if type(max_cardinality) is not int or max_cardinality <= 0:
            raise ValueError("max_cardinality must be a positive Python int.")
        if policy == "declared":
            if not isinstance(alphabets, Mapping):
                raise ValueError("declared policy requires an alphabet mapping.")
            _ids(tuple(alphabets))
            if set(alphabets) != set(train.variable_ids):
                raise ValueError("Declared alphabets must match all variable identifiers.")
        elif alphabets is not None:
            raise ValueError("train_observed policy does not accept declared alphabets.")
        categories, mappings = {}, {}
        for column, variable in enumerate(train.variable_ids):
            if policy == "declared":
                raw = alphabets[variable]
                if not isinstance(raw, np.ndarray) or raw.ndim != 1:
                    raise ValueError(f"Declared alphabet for {variable!r} must be a 1D ndarray.")
                raw = _values(raw)[:, 0]
                labels = np.unique(raw)
                if len(labels) != len(raw):
                    raise ValueError(f"Declared alphabet for {variable!r} contains duplicates.")
            else:
                labels = np.unique(train.values[:, column])
            if len(labels) > max_cardinality:
                raise ValueError(f"Variable {variable!r} exceeds max_cardinality={max_cardinality}.")
            labels.flags.writeable = False
            categories[variable] = labels
            mappings[variable] = MappingProxyType({int(label): i for i, label in enumerate(labels)})
        result = cls()
        result._dataset_id = train.dataset_id
        result._split_id = train.split_id
        result._n_samples = train.n_samples
        result._variable_ids = train.variable_ids
        result._train_rows = frozenset(map(int, train.row_indices))
        result._policy = policy
        result._categories = MappingProxyType(categories)
        result._mappings = MappingProxyType(mappings)
        result._cardinalities = MappingProxyType({v: len(c) for v, c in categories.items()})
        # Reject training values outside the independently declared support.
        for variable in train.variable_ids:
            result.target_indices(train, variable)
        return result

    @property
    def policy(self) -> str:
        return self._policy

    @property
    def variable_ids(self) -> tuple[VariableId, ...]:
        return self._variable_ids

    @property
    def categories(self) -> Mapping[VariableId, np.ndarray]:
        # Detached copies prevent callers from modifying fitted inverse maps.
        return MappingProxyType({v: c.copy() for v, c in self._categories.items()})

    @property
    def mappings(self) -> Mapping[VariableId, Mapping[int, int]]:
        return self._mappings

    @property
    def cardinalities(self) -> Mapping[VariableId, int]:
        return self._cardinalities

    def _check(self, batch: CategoricalBatch) -> None:
        if not isinstance(batch, CategoricalBatch):
            raise TypeError("Encoding requires a CategoricalBatch.")
        if (batch.dataset_id != self._dataset_id or batch.split_id != self._split_id
                or batch.n_samples != self._n_samples or batch.variable_ids != self.variable_ids):
            raise ValueError("Batch dataset/split identity, size or column order differs from training.")
        rows = frozenset(map(int, batch.row_indices))
        if batch.role == "train" and not rows <= self._train_rows:
            raise ValueError("Train batch contains rows outside the fitted training partition.")
        if batch.role != "train" and rows & self._train_rows:
            raise ValueError("Validation/test rows overlap the fitted training partition.")

    def _variable(self, variable: VariableId) -> None:
        _ids((variable,))
        if variable not in self._mappings:
            raise ValueError(f"Unknown variable identifier {variable!r}.")

    def target_indices(self, batch: CategoricalBatch, variable: VariableId) -> np.ndarray:
        """Return (n,) int64 target classes; report unknown label, role and variable."""
        self._check(batch)
        self._variable(variable)
        column = batch.values[:, self.variable_ids.index(variable)]
        mapping = self._mappings[variable]
        encoded = np.empty(len(column), dtype=np.int64)
        for row, label in enumerate(column):
            key = int(label)
            if key not in mapping:
                raise ValueError(f"Unknown category {key!r} for variable {variable!r} in {batch.role}.")
            encoded[row] = mapping[key]
        return encoded

    def inverse_target(self, indices: np.ndarray, variable: VariableId) -> np.ndarray:
        """Decode 1D class indices into labels using the alphabet's original dtype."""
        self._variable(variable)
        if (np.ma.isMaskedArray(indices) or not isinstance(indices, np.ndarray)
                or indices.ndim != 1 or indices.dtype.kind not in "iu"):
            raise ValueError("Class indices must be an unmasked one-dimensional integer ndarray.")
        if np.any(indices < 0) or np.any(indices >= self.cardinalities[variable]):
            raise ValueError("Class indices are outside the variable alphabet.")
        return self._categories[variable][indices.astype(np.int64)]

    def conditioning_features(
        self, batch: CategoricalBatch, conditioning: tuple[VariableId, ...], *,
        target: VariableId, max_bytes: int = 256 * 1024**2,
    ) -> np.ndarray:
        """Return dense float32 one-hot blocks in explicit conditioning order.

        Target exclusion is mandatory. Empty conditioning returns shape (n, 0).
        The memory budget covers the dense output; mapping/data storage is separate.
        """
        self._check(batch)
        self._variable(target)
        _ids(conditioning)
        for variable in conditioning:
            self._variable(variable)
        if target in conditioning:
            raise ValueError("Target variable must be excluded from conditioning features.")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive Python int.")
        width = sum(self.cardinalities[v] for v in conditioning)
        required = len(batch.values) * width * np.dtype(np.float32).itemsize
        if required > max_bytes:
            raise ValueError(f"One-hot features require {required} bytes, exceeding max_bytes={max_bytes}.")
        features = np.zeros((len(batch.values), width), dtype=np.float32)
        offset = 0
        rows = np.arange(len(batch.values))
        for variable in conditioning:
            encoded = self.target_indices(batch, variable)
            features[rows, offset + encoded] = 1.0
            offset += self.cardinalities[variable]
        return features
