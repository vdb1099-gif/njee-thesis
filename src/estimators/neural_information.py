"""Directional MI/CMI compositions of validated NJEE and C-NJEE (nats).

No symmetrization or clipping. MI retains NJEE's TRAIN plug-in first marginal.
Distinct deterministic streams prevent accidental reuse, not prove independence.
Caller-assigned identities and declared support provenance cannot be authenticated.
"""
from dataclasses import dataclass, replace
import json
import math
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from src.estimators.neural import NeuralJointEntropyEstimator, NeuralEntropyReport, _batch
from src.estimators.conditional_neural import (
    ConditionalNeuralJointEntropyEstimator, ConditionalNeuralEntropyReport,
)
from src.utils.categorical import CategoricalBatch, CategoricalEncoder, VariableId, validate_variable_order
from src.utils.neural_config import NeuralConfig
from src.utils.reproducibility import derive_term_seed, capture_environment


@dataclass(frozen=True)
class InformationComponent:
    name: str
    config: NeuralConfig
    conditioning_order: tuple[VariableId, ...]
    entropy: NeuralEntropyReport | ConditionalNeuralEntropyReport
    fit_runtime_seconds: float
    evaluation_runtime_seconds: float


@dataclass(frozen=True)
class NeuralInformationReport:
    """Signed difference; nested entropy reports retain every term and environment.

    Effective initialization/shuffle seeds are in each entropy report's terms;
    component training seeds are in config. caller_environment records backend
    flags outside the scoped component evaluations; term environments record active
    training settings. Timing fields vary across evaluations.
    """
    information_nats: float
    kind: str
    x_order: tuple[VariableId, ...]
    y_order: tuple[VariableId, ...]
    z_order: tuple[VariableId, ...]
    dataset_id: str
    split_id: str
    base_config: NeuralConfig
    seed_derivation: str
    caller_environment: Mapping
    components: tuple[InformationComponent, InformationComponent]


SEED_DERIVATION = "njee-information-components-v1"


def _project(batch, order):
    return replace(batch, values=batch.values[:, [batch.variable_ids.index(v) for v in order]],
                   variable_ids=order)


class NeuralMutualInformationEstimator:
    """Fit H(X) and H(X|Y) independently on identical partition rows.

    config.component_order specifies X; y_order specifies Y. Unused columns are
    permitted. Alphabets cover the complete input schema. Fit is atomic and
    single-use; estimate is read-only and may be repeated. Original schema/order
    and all partition metadata are checked before projection. No split is made.
    """
    def __init__(self, config: NeuralConfig, *, y_order: tuple[VariableId, ...],
                 alphabets: Mapping[VariableId, np.ndarray] | None = None):
        self._initialize(config, y_order, (), alphabets)

    def _initialize(self, config, y_order, z_order, alphabets):
        # Reuse the existing constructor's validation and defensive alphabet copy.
        validated = NeuralJointEntropyEstimator(config, alphabets=alphabets)
        validate_variable_order(y_order, y_order)
        if z_order:
            validate_variable_order(z_order, z_order)
        groups = config.component_order + y_order + z_order
        if len(set(groups)) != len(groups):
            raise ValueError("X, Y and Z blocks must be disjoint.")
        self._config, self._alphabets = config, validated._alphabets
        self._y_order, self._z_order = y_order, z_order
        self._fitted = False

    @property
    def config(self):
        return self._config

    @property
    def y_order(self):
        return self._y_order

    @property
    def z_order(self):
        return self._z_order

    def _specifications(self):
        if self.z_order:
            return (("h_x_given_z", self.z_order),
                    ("h_x_given_yz", self.y_order + self.z_order))
        return (("h_x", ()), ("h_x_given_y", self.y_order))

    def _plans(self):
        """Hash compact JSON [version, component name, X, Y, Z] through
        derive_term_seed(base training seed, identifier, purpose="initialization").
        Existing trainers then derive their per-target initialization/shuffle seeds.
        Reject finite-domain collisions across all effective component streams.
        """
        plans, seeds = [], set()
        for name, context in self._specifications():
            identifier = json.dumps([SEED_DERIVATION, name, self.config.component_order,
                                     self.y_order, self.z_order],
                                    ensure_ascii=False, separators=(",", ":"))
            seed = derive_term_seed(self.config.training_seed, identifier, purpose="initialization")
            config = replace(self.config, training_seed=seed)
            streams = [seed]
            for i, target in enumerate(config.component_order):
                if not context and i == 0:
                    continue
                tid = json.dumps(["njee-conditional-term-v1", target,
                                  list(context + config.component_order[:i])],
                                 ensure_ascii=False, separators=(",", ":"))
                streams.extend(derive_term_seed(seed, tid, purpose=p)
                               for p in ("initialization", "shuffle"))
            if len(set(streams)) != len(streams) or seeds.intersection(streams):
                raise ValueError("Effective component seed collision; no models fitted.")
            seeds.update(streams)
            plans.append((name, context, config))
        return plans

    def fit(self, train: CategoricalBatch, validation: CategoricalBatch):
        if self._fitted:
            raise RuntimeError("Estimator is already fitted.")
        _batch(train, "train")
        _batch(validation, "validation")
        used = self.config.component_order + self.y_order + self.z_order
        if not set(used) <= set(train.variable_ids):
            raise ValueError("Unknown variable in X/Y/Z blocks.")
        self.config.validate_dataset(variable_ids=self.config.component_order, n_samples=train.n_samples)
        encoder = CategoricalEncoder.fit(train, policy=self.config.alphabet_policy, alphabets=self._alphabets)
        # Validate original schema, identities and categories before fitting either component.
        for v in train.variable_ids:
            encoder.target_indices(validation, v)
        fitted = []
        for name, context, config in self._plans():
            order = config.component_order + context
            alphabets = None if self._alphabets is None else {v: self._alphabets[v] for v in order}
            if context:
                model = ConditionalNeuralJointEntropyEstimator(config, conditioning_order=context,
                                                               alphabets=alphabets)
            else:
                model = NeuralJointEntropyEstimator(config, alphabets=alphabets)
            start = perf_counter()
            model.fit(_project(train, order), _project(validation, order))
            fitted.append((name, context, config, order, model, perf_counter() - start))
        self._encoder, self._components = encoder, tuple(fitted)
        self._validation_rows = frozenset(map(int, validation.row_indices))
        self._train_rows = frozenset(map(int, train.row_indices))
        self._n_samples = train.n_samples
        self._dataset_id, self._split_id = train.dataset_id, train.split_id
        self._fitted = True
        return self

    def estimate(self, test: CategoricalBatch) -> NeuralInformationReport:
        if not self._fitted:
            raise RuntimeError("Estimator must be fitted before estimate.")
        _batch(test, "test")
        for v in self._encoder.variable_ids:
            self._encoder.target_indices(test, v)
        rows = frozenset(map(int, test.row_indices))
        if rows & self._validation_rows or len(rows | self._validation_rows | self._train_rows) != self._n_samples:
            raise ValueError("Test overlaps validation or partitions do not cover the dataset.")
        components = []
        for name, context, config, order, model, runtime in self._components:
            start = perf_counter()
            report = model.estimate(_project(test, order))
            if (report.dataset_id != self._dataset_id or report.split_id != self._split_id
                    or report.component_order != config.component_order
                    or (context and report.conditioning_order != context)):
                raise RuntimeError("Component report provenance differs from the requested composition.")
            if not math.isfinite(report.entropy_nats):
                raise RuntimeError("Nonfinite component entropy.")
            components.append(InformationComponent(name, config, context, report, runtime,
                                                   perf_counter() - start))
        difference = components[0].entropy.entropy_nats - components[1].entropy.entropy_nats
        if not math.isfinite(difference):
            raise RuntimeError("Nonfinite information difference.")
        return NeuralInformationReport(difference, "cmi" if self.z_order else "mi",
                                       self.config.component_order, self.y_order, self.z_order,
                                       self._dataset_id, self._split_id, self.config,
                                       SEED_DERIVATION,
                                       MappingProxyType(capture_environment(
                                           mode=self.config.reproducibility_mode, device=self.config.device)),
                                       tuple(components))


class NeuralConditionalMutualInformationEstimator(NeuralMutualInformationEstimator):
    """Directional H(X|Z) - H(X|Y,Z); feature context is Z then Y,Z respectively."""
    def __init__(self, config: NeuralConfig, *, y_order: tuple[VariableId, ...],
                 z_order: tuple[VariableId, ...],
                 alphabets: Mapping[VariableId, np.ndarray] | None = None):
        validate_variable_order(z_order, z_order)
        self._initialize(config, y_order, z_order, alphabets)
