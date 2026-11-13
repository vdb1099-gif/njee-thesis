"""C-NJEE: one held-out conditional classifier per X term, including the first."""
from dataclasses import dataclass
import math
import random
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from src.estimators.neural import NeuralJointEntropyEstimator, TermDiagnostic, _batch, _test_ce
from src.estimators.neural_training import train_conditional_term
from src.utils.categorical import CategoricalBatch, CategoricalEncoder, VariableId, validate_variable_order
from src.utils.neural_config import NeuralConfig
from src.utils.reproducibility import reproducible_execution


@dataclass(frozen=True)
class ConditionalNeuralEntropyReport:
    """Immutable conditional estimate in nats; all contributions are TEST CE."""
    entropy_nats: float
    component_order: tuple[VariableId, ...]
    conditioning_order: tuple[VariableId, ...]
    terms: tuple[TermDiagnostic, ...]
    reproducibility_mode: str
    dataset_id: str
    split_id: str


class ConditionalNeuralJointEntropyEstimator(NeuralJointEntropyEstimator):
    """Estimate H(X|Y) with ordered Y and config.component_order specifying X.

    Reuses the base constructor's config/alphabet copying and fitted-state rules;
    neither base fit nor its unconditional marginal is used. NeuralConfig's
    marginal_method field is NJEE compatibility metadata, unused for C-NJEE.
    Y must be nonempty. Unused columns are allowed; alphabets still cover the
    complete dataset schema. Declared support is caller-provided domain knowledge.
    Numerical results repeat in one environment; timing fields may differ.
    Test evaluation does not enforce one-time scientific use by the caller.
    """

    def __init__(self, config: NeuralConfig, *, conditioning_order: tuple[VariableId, ...],
                 alphabets: Mapping[VariableId, np.ndarray] | None = None) -> None:
        super().__init__(config, alphabets=alphabets)
        validate_variable_order(conditioning_order, conditioning_order)
        if not conditioning_order:
            raise ValueError("C-NJEE requires a nonempty conditioning-variable order.")
        if set(conditioning_order) & set(config.component_order):
            raise ValueError("X and Y variable identities must be disjoint.")
        self._conditioning_order = conditioning_order

    @property
    def conditioning_order(self):
        return self._conditioning_order

    def fit(self, train: CategoricalBatch, validation: CategoricalBatch):
        """Fit train-only mappings and d_X classifiers, using validation selection."""
        if self._fitted:
            raise RuntimeError("Estimator is already fitted; construct a new estimator to refit.")
        _batch(train, "train")
        _batch(validation, "validation")
        group = tuple(v for v in train.variable_ids if v in self.config.component_order)
        self.config.validate_dataset(variable_ids=group, n_samples=train.n_samples)
        if not set(self.conditioning_order) <= set(train.variable_ids):
            raise ValueError("Unknown conditioning variable identifier.")
        encoder = CategoricalEncoder.fit(train, policy=self.config.alphabet_policy,
                                         alphabets=self._alphabets)
        for variable in self.conditioning_order + self.config.component_order:
            encoder.target_indices(validation, variable)
        if np.intersect1d(train.row_indices, validation.row_indices).size:
            raise ValueError("Training and validation partitions overlap.")
        results = tuple(train_conditional_term(
            train, validation, encoder=encoder, config=self.config, target=target,
            conditioning=self.conditioning_order + self.config.component_order[:position],
            context=self.conditioning_order,
        ) for position, target in enumerate(self.config.component_order))
        # Publish fitted state only once every classifier has completed.
        self._encoder, self._results = encoder, results
        self._train_rows = frozenset(map(int, train.row_indices))
        self._validation_rows = frozenset(map(int, validation.row_indices))
        self._n_samples = train.n_samples
        self._dataset_id, self._split_id = train.dataset_id, train.split_id
        self._fitted = True
        return self

    def estimate(self, test: CategoricalBatch) -> ConditionalNeuralEntropyReport:
        """Sum test CE of frozen models for H(X_m | Y, X_<m); no marginal term."""
        if not self._fitted:
            raise RuntimeError("Estimator must be fitted before estimate.")
        _batch(test, "test")
        # Validate all used variables before any model evaluation.
        encoded = {v: self._encoder.target_indices(test, v)
                   for v in self.conditioning_order + self.config.component_order}
        rows = frozenset(map(int, test.row_indices))
        if rows & self._validation_rows:
            raise ValueError("Test and validation partitions overlap.")
        if len(rows | self._train_rows | self._validation_rows) != self._n_samples:
            raise ValueError("Partitions must cover all original dataset rows.")
        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        terms = []
        try:
            with torch.random.fork_rng(devices=devices):
                with reproducible_execution(self.config.training_seed,
                                            mode=self.config.reproducibility_mode,
                                            device=self.config.device):
                    for position, result in enumerate(self._results, 1):
                        start = perf_counter()
                        features = self._encoder.conditioning_features(test, result.conditioning,
                                                                        target=result.target)
                        value = _test_ce(result.model, features, encoded[result.target],
                                         cardinality=result.output_cardinality, config=self.config)
                        terms.append(TermDiagnostic(
                            position, result.target, result.conditioning, "neural_test_ce", value,
                            "test", len(test.values), result.output_cardinality,
                            result.initialization_seed, result.shuffle_seed, result.epochs_trained,
                            result.best_epoch, result.best_validation_ce_nats,
                            result.final_training_ce_nats, result.runtime_seconds,
                            perf_counter()-start, result.stop_reason, result.warnings,
                            result.epoch_training_ce_nats, result.epoch_validation_ce_nats,
                            MappingProxyType(dict(result.environment)),
                        ))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        total = math.fsum(t.contribution_nats for t in terms)
        if not math.isfinite(total):
            raise RuntimeError("Nonfinite conditional entropy estimate.")
        return ConditionalNeuralEntropyReport(total, self.config.component_order, self.conditioning_order,
                                              tuple(terms), self.config.reproducibility_mode,
                                              self._dataset_id, self._split_id)
