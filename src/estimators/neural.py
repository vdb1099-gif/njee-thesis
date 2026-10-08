"""Baseline NJEE: training plug-in marginal plus held-out conditional CE in nats."""

from dataclasses import dataclass
import math
import random
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from src.estimators.classical import PlugInEstimator
from src.estimators.neural_training import train_conditional_term
from src.utils.categorical import CategoricalBatch, CategoricalEncoder, VariableId
from src.utils.neural_config import NeuralConfig
from src.utils.reproducibility import reproducible_execution


@dataclass(frozen=True)
class TermDiagnostic:
    """One chain contribution; non-applicable training fields are explicitly None."""

    position: int
    target: VariableId
    conditioning: tuple[VariableId, ...]
    method: str
    contribution_nats: float
    evaluation_role: str
    n_evaluation_observations: int
    output_cardinality: int
    initialization_seed: int | None
    shuffle_seed: int | None
    epochs_trained: int | None
    best_epoch: int | None
    best_validation_ce_nats: float | None
    final_training_ce_nats: float | None
    training_runtime_seconds: float
    evaluation_runtime_seconds: float
    stop_reason: str
    warnings: tuple[str, ...]
    epoch_training_ce_nats: tuple[float, ...]
    epoch_validation_ce_nats: tuple[float, ...]
    environment: Mapping | None


@dataclass(frozen=True)
class NeuralEntropyReport:
    """Read-only numerical result and diagnostics; runtimes may vary across calls."""

    entropy_nats: float
    component_order: tuple[VariableId, ...]
    terms: tuple[TermDiagnostic, ...]
    reproducibility_mode: str
    dataset_id: str
    split_id: str


def _batch(batch: CategoricalBatch, role: str) -> None:
    if not isinstance(batch, CategoricalBatch):
        raise TypeError(f"{role} must be a CategoricalBatch.")
    if batch.role != role:
        raise ValueError(f"Expected {role} role.")


def _test_ce(model, features: np.ndarray, targets: np.ndarray, *,
             cardinality: int, config: NeuralConfig) -> float:
    """Visit each row once, without shuffle, optimizer or RNG consumption.

    The trained checkpoint must already be in eval mode. Refuse a mutable
    training-mode model rather than silently changing its mode during estimate.
    Ordered slicing handles incomplete final batches and uses no DataLoader RNG.
    """
    if model.training:
        raise RuntimeError("Held-out evaluation requires an eval-mode checkpoint.")
    totals = []
    with torch.inference_mode():
        for start in range(0, len(targets), config.batch_size):
            x = torch.from_numpy(features[start:start + config.batch_size]).to(config.device)
            y = torch.from_numpy(targets[start:start + config.batch_size]).to(config.device)
            logits = model(x)
            if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
                    or logits.shape != (len(y), cardinality)
                    or not logits.is_floating_point() or not torch.isfinite(logits).all()):
                raise RuntimeError("Invalid or nonfinite held-out logits.")
            loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
            if not torch.isfinite(loss).all():
                raise RuntimeError("Nonfinite held-out cross-entropy.")
            totals.append(loss.double().sum().item())
    value = math.fsum(totals) / len(targets)
    if not math.isfinite(value):
        raise RuntimeError("Nonfinite mean held-out cross-entropy.")
    return value


class NeuralJointEntropyEstimator:
    """Fit once on TRAIN/VALIDATION, then estimate on independent TEST.

    estimate returns NeuralEntropyReport, not the classical ndarray/float API.
    One schema is fitted on training only. Declared alphabets are caller-supplied
    domain knowledge; their provenance cannot be authenticated here. Refit is
    rejected. A failed fit publishes no fitted state and can be retried.
    Calls to estimate do not store results or alter checkpoints, encodings or RNG
    streams. Timing fields vary; numerical contributions repeat in one environment.
    This API does not enforce one-time scientific use of test results.
    """

    def __init__(self, config: NeuralConfig, *,
                 alphabets: Mapping[VariableId, np.ndarray] | None = None) -> None:
        if not isinstance(config, NeuralConfig):
            raise TypeError("config must be a validated NeuralConfig.")
        if config.alphabet_policy == "declared":
            if not isinstance(alphabets, Mapping):
                raise ValueError("Declared policy requires alphabets.")
            if any(not isinstance(a, np.ndarray) for a in alphabets.values()):
                raise ValueError("Declared alphabets must be NumPy arrays.")
            self._alphabets = {v: a.copy() for v, a in alphabets.items()}
        else:
            if alphabets is not None:
                raise ValueError("train_observed policy does not accept alphabets.")
            self._alphabets = None
        self._config = config
        self._fitted = False

    @property
    def config(self) -> NeuralConfig:
        return self._config

    def fit(self, train: CategoricalBatch, validation: CategoricalBatch):
        """Train all non-marginal terms without accepting or accessing test data."""
        if self._fitted:
            raise RuntimeError("Estimator is already fitted; construct a new estimator to refit.")
        _batch(train, "train")
        _batch(validation, "validation")
        self.config.validate_dataset(variable_ids=train.variable_ids, n_samples=train.n_samples)
        encoder = CategoricalEncoder.fit(train, policy=self.config.alphabet_policy,
                                         alphabets=self._alphabets)
        # Check every validation variable even for d=1 or singleton targets.
        for variable in self.config.component_order:
            encoder.target_indices(validation, variable)
        if np.intersect1d(train.row_indices, validation.row_indices).size:
            raise ValueError("Training and validation partitions overlap.")
        start = perf_counter()
        first = self.config.component_order[0]
        marginal = PlugInEstimator().estimate(train.values[:, train.variable_ids.index(first)])
        marginal_runtime = perf_counter() - start
        results = tuple(train_conditional_term(
            train, validation, encoder=encoder, config=self.config, target=target,
            conditioning=self.config.component_order[:position],
        ) for position, target in enumerate(self.config.component_order) if position)
        self._encoder = encoder
        self._train_rows = frozenset(map(int, train.row_indices))
        self._validation_rows = frozenset(map(int, validation.row_indices))
        self._n_samples = train.n_samples
        self._dataset_id, self._split_id = train.dataset_id, train.split_id
        self._n_train = len(train.values)
        self._marginal, self._marginal_runtime = marginal, marginal_runtime
        self._results = results
        self._fitted = True
        return self

    def estimate(self, test: CategoricalBatch) -> NeuralEntropyReport:
        """Evaluate frozen checkpoints; sum TRAIN marginal and TEST conditional CE."""
        if not self._fitted:
            raise RuntimeError("Estimator must be fitted before estimate.")
        _batch(test, "test")
        # Public encoder checks identities, column order, bounds and train overlap.
        targets = {v: self._encoder.target_indices(test, v) for v in self.config.component_order}
        rows = frozenset(map(int, test.row_indices))
        if rows & self._validation_rows:
            raise ValueError("Test and validation partitions overlap.")
        if len(rows | self._train_rows | self._validation_rows) != self._n_samples:
            raise ValueError("Partitions must cover all original dataset rows.")
        first = self.config.component_order[0]
        terms = [TermDiagnostic(
            1, first, (), "train_plugin", self._marginal, "train", self._n_train,
            self._encoder.cardinalities[first], None, None, None, None, None, None,
            self._marginal_runtime, 0., "train_plugin", (), (), (), None,
        )]
        if not self._results:
            return NeuralEntropyReport(self._marginal, self.config.component_order, tuple(terms),
                                       self.config.reproducibility_mode, self._dataset_id, self._split_id)
        # Reuse strict device/backend controls while restoring all seeded RNGs.
        # No reseeding/consumption escapes evaluation, including on exceptions.
        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        try:
            with torch.random.fork_rng(devices=devices):
                with reproducible_execution(self.config.training_seed,
                                            mode=self.config.reproducibility_mode,
                                            device=self.config.device):
                    for position, result in enumerate(self._results, start=2):
                        start = perf_counter()
                        features = self._encoder.conditioning_features(
                            test, result.conditioning, target=result.target)
                        value = _test_ce(result.model, features, targets[result.target],
                                         cardinality=result.output_cardinality, config=self.config)
                        terms.append(TermDiagnostic(
                            position, result.target, result.conditioning, "neural_test_ce", value,
                            "test", len(test.values), result.output_cardinality,
                            result.initialization_seed, result.shuffle_seed, result.epochs_trained,
                            result.best_epoch, result.best_validation_ce_nats,
                            result.final_training_ce_nats, result.runtime_seconds,
                            perf_counter() - start, result.stop_reason, result.warnings,
                            result.epoch_training_ce_nats, result.epoch_validation_ce_nats,
                            MappingProxyType(dict(result.environment)),
                        ))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        total = math.fsum(term.contribution_nats for term in terms)
        if not math.isfinite(total):
            raise RuntimeError("Nonfinite joint entropy estimate.")
        return NeuralEntropyReport(total, self.config.component_order, tuple(terms),
                                   self.config.reproducibility_mode, self._dataset_id, self._split_id)
