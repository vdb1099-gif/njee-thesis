"""Standalone neural YAML schema; the classical baseline schema is unchanged.

All fields below are required: omitted/unsupported/invalid settings never fall
back to defaults. Configuration loading neither seeds RNGs nor initializes CUDA.
"""

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any

import yaml

from src.utils.categorical import VariableId, validate_variable_order
from src.utils.reproducibility import validate_device, validate_seed
from src.utils.split import _allocation


@dataclass(frozen=True)
class NeuralConfig:
    """Minimal fixed baseline choices with tunable scalar training settings.

    One or two ReLU hidden layers, Adam, train_plugin marginal, and zero workers.
    split_seed and training_seed designate independent RNG roles; numerical equality
    is permitted. Validation/test shuffle and optimization are not config options.
    """

    split_seed: int
    split_fractions: tuple[float, float, float]
    training_seed: int
    component_order: tuple[VariableId, ...]
    alphabet_policy: str
    marginal_method: str
    hidden_sizes: tuple[int, ...]
    activation: str
    optimizer: str
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float
    device: str
    reproducibility_mode: str
    num_workers: int

    def __post_init__(self) -> None:
        for name in ("split_seed", "training_seed"):
            try:
                validate_seed(getattr(self, name))
            except ValueError as exc:
                raise ValueError(f"{name}={getattr(self, name)!r}: {exc}") from exc
        fractions = self.split_fractions
        if (not isinstance(fractions, tuple) or len(fractions) != 3
                or any(type(f) not in (int, float) or f <= 0 or f > 1 or not math.isfinite(f)
                       for f in fractions)
                or not math.isclose(math.fsum(fractions), 1., rel_tol=0., abs_tol=1e-12)):
            raise ValueError(f"split_fractions={fractions!r}: expected three positive finite "
                             "fractions summing to 1 within 1e-12.")
        # Without a schema, validate identifiers and duplicates, not assumed positions.
        validate_variable_order(self.component_order, self.component_order)
        if (not isinstance(self.hidden_sizes, tuple) or not 1 <= len(self.hidden_sizes) <= 2
                or any(type(size) is not int or size <= 0 for size in self.hidden_sizes)):
            raise ValueError(f"hidden_sizes={self.hidden_sizes!r}: expected one or two positive integers.")
        supported = {"alphabet_policy": ("declared", "train_observed"),
                     "marginal_method": ("train_plugin",), "activation": ("relu",),
                     "optimizer": ("adam",), "reproducibility_mode": ("seeded", "strict")}
        for name, options in supported.items():
            value = getattr(self, name)
            if not isinstance(value, str) or value not in options:
                raise ValueError(f"{name}={value!r}: supported values are {options}.")
        for name in ("batch_size", "max_epochs", "patience"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}={value!r}: expected a positive Python int.")
        for name in ("learning_rate", "min_delta"):
            value = getattr(self, name)
            try:
                valid = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"{name}={value!r}: expected a finite "
                                 f"{'positive' if name == 'learning_rate' else 'nonnegative'} number.")
        validate_device(self.device)
        if type(self.num_workers) is not int or self.num_workers != 0:
            raise ValueError(f"num_workers={self.num_workers!r}: only integer zero is supported.")

    def validate_dataset(
        self, *, variable_ids: tuple[VariableId, ...], n_samples: int | None = None,
    ) -> None:
        """Validate complete target order and, when known, nonempty split allocation."""
        validate_variable_order(self.component_order, variable_ids)
        if n_samples is not None:
            _allocation(n_samples, self.split_fractions, self.split_seed)


def validate_neural_config(
    config: dict[str, Any], *, variable_ids: tuple[VariableId, ...] | None = None,
    n_samples: int | None = None,
) -> NeuralConfig:
    """Validate a standalone flat mapping and return an immutable configuration.

    YAML lists for tuple fields are converted without numeric coercion. Supplying
    variable_ids enables permutation checks against the actual target schema.
    """
    if not isinstance(config, dict):
        raise ValueError("Neural configuration must be a mapping.")
    expected = {field.name for field in fields(NeuralConfig)}
    if set(config) != expected:
        raise ValueError(f"Neural configuration keys: missing={expected - set(config)!r}, "
                         f"unsupported={set(config) - expected!r}.")
    values = config.copy()
    for name in ("split_fractions", "component_order", "hidden_sizes"):
        if isinstance(values[name], list):
            values[name] = tuple(values[name])
    result = NeuralConfig(**values)
    if variable_ids is not None:
        result.validate_dataset(variable_ids=variable_ids, n_samples=n_samples)
    elif n_samples is not None:
        _allocation(n_samples, result.split_fractions, result.split_seed)
    return result


def load_neural_config(
    path: str | Path, *, variable_ids: tuple[VariableId, ...] | None = None,
    n_samples: int | None = None,
) -> NeuralConfig:
    """Load standalone neural YAML; invalid contents report the source path."""
    path = Path(path)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        return validate_neural_config(config, variable_ids=variable_ids, n_samples=n_samples)
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"Invalid neural configuration '{path}': {exc}") from exc
