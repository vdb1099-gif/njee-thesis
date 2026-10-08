"""Deterministic algebra, held-out isolation and tiny CPU NJEE integration."""

from dataclasses import FrozenInstanceError, replace
import math
import random

import numpy as np
import pytest
import torch

from src.estimators import neural
from src.estimators.classical import PlugInEstimator
from src.estimators.neural_training import TermTrainingResult
from src.models.mlp import ConditionalMLP
from src.utils.categorical import CategoricalDataset, CategoricalEncoder
from src.utils.neural_config import NeuralConfig
from src.utils.split import make_iid_split


@pytest.fixture(autouse=True)
def preserve_state():
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    torch.set_num_threads(threads)


def scenario(d=3, n=23, policy="declared", singleton=False):
    values = np.array([[(i + j) % 2 for j in range(d)] for i in range(n)], dtype=np.int64)
    if singleton:
        values[:, -1] = 0
    dataset = CategoricalDataset(values, "data", tuple(range(d)))
    split = make_iid_split(n, seed=42)
    batches = tuple(dataset.batch(split, role, split_id="split")
                    for role in ("train", "validation", "test"))
    config = NeuralConfig(42, (0.6, 0.2, 0.2), 202, tuple(range(d)), policy,
                          "train_plugin", (4,), "relu", "adam", .01, 2, 2, 2, 0.,
                          "cpu", "strict", 0)
    alphabets = {v: np.array([0, 1]) for v in range(d)} if policy == "declared" else None
    return batches, config, alphabets


def controlled_trainer(calls):
    def train(train, validation, *, encoder, config, target, conditioning):
        assert train.role == "train" and validation.role == "validation"
        width = sum(encoder.cardinalities[v] for v in conditioning)
        model = ConditionalMLP(width, encoder.cardinalities[target], hidden_sizes=(4,))
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            if model.output_cardinality == 2:
                model.network[-1].bias.copy_(torch.tensor([0., math.log(3)]))
        model.eval().requires_grad_(False)
        model.register_buffer("evaluation_marker", torch.tensor([7.]))
        calls.append((target, conditioning, model, encoder))
        return TermTrainingResult(model, target, conditioning, model.output_cardinality,
                                  str(target), 100 + len(calls), 200 + len(calls), 3, 2,
                                  99., 88., .1, "early_stopping", ("controlled",),
                                  (3., 2., 1.), (100., 99., 100.), {"reproducibility_mode": "strict"})
    return train


def test_chain_order_schema_diagnostics_and_train_marginal(monkeypatch):
    (train, validation, test), config, alphabets = scenario()
    config = replace(config, component_order=(2, 0, 1))
    calls = []
    monkeypatch.setattr(neural, "train_conditional_term", controlled_trainer(calls))
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets)
    assert estimator.fit(train, validation) is estimator
    assert [(t, c) for t, c, _, _ in calls] == [(0, (2,)), (1, (2, 0))]
    assert calls[0][2] is not calls[1][2]
    assert calls[0][3] is calls[1][3]
    report = estimator.estimate(test)
    marginal = PlugInEstimator().estimate(train.values[:, 2])
    contributions = [marginal]
    for target in (0, 1):
        labels = test.values[:, target]
        contributions.append(sum(math.log(4) if y == 0 else math.log(4/3) for y in labels) / len(labels))
    assert [t.contribution_nats for t in report.terms] == pytest.approx(contributions, abs=1e-7)
    assert report.entropy_nats == math.fsum(t.contribution_nats for t in report.terms)
    assert report.component_order == (2, 0, 1)
    assert [t.position for t in report.terms] == [1, 2, 3]
    first, term = report.terms[:2]
    assert first.method == "train_plugin" and first.evaluation_role == "train"
    assert first.n_evaluation_observations == len(train.values)
    assert first.initialization_seed is first.shuffle_seed is first.epochs_trained is None
    assert first.best_epoch is first.best_validation_ce_nats is first.environment is None
    assert term.method == "neural_test_ce" and term.evaluation_role == "test"
    assert term.n_evaluation_observations == len(test.values)
    assert term.best_validation_ce_nats == 99. and term.contribution_nats != 99.
    assert term.epochs_trained == 3 and term.best_epoch == 2
    assert term.final_training_ce_nats == 88. and term.training_runtime_seconds == .1
    assert term.initialization_seed == 101 and term.shuffle_seed == 201
    assert term.stop_reason == "early_stopping" and term.warnings == ("controlled",)
    assert term.epoch_validation_ce_nats == (100., 99., 100.)
    with pytest.raises(FrozenInstanceError):
        report.entropy_nats = 0.
    with pytest.raises(FrozenInstanceError):
        term.contribution_nats = 0.
    with pytest.raises(TypeError):
        term.environment["reproducibility_mode"] = "seeded"


def test_d1_uses_training_plugin_no_classifier_or_test_entropy(monkeypatch):
    (train, validation, test), config, alphabets = scenario(d=1)
    changed = replace(test, values=np.zeros_like(test.values))
    def forbidden(*args, **kwargs):
        raise AssertionError("No neural training/evaluation for d=1")
    monkeypatch.setattr(neural, "train_conditional_term", forbidden)
    monkeypatch.setattr(neural, "_test_ce", forbidden)
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets).fit(train, validation)
    report = estimator.estimate(changed)
    assert report.entropy_nats == PlugInEstimator().estimate(train.values)
    assert report.entropy_nats > PlugInEstimator().estimate(changed.values)
    assert len(report.terms) == 1


@pytest.mark.parametrize("batch_size", [1, 2, 100])
def test_test_ce_coverage_no_updates_and_repeatability(monkeypatch, batch_size):
    (train, validation, test), config, alphabets = scenario(d=2)
    config = replace(config, batch_size=batch_size)
    calls = []
    monkeypatch.setattr(neural, "train_conditional_term", controlled_trainer(calls))
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets).fit(train, validation)
    model = calls[0][2]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mapping = dict(estimator._encoder.mappings)
    seen = []
    def hook(module, args):
        assert not module.training and not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        seen.append(args[0].clone())
    model.register_forward_pre_hook(hook)
    def forbidden(*args, **kwargs):
        raise AssertionError("No optimizer, fitting or training during estimate")
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(CategoricalEncoder, "fit", forbidden)
    monkeypatch.setattr(neural, "train_conditional_term", forbidden)
    monkeypatch.setattr(PlugInEstimator, "estimate", forbidden)
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    flags = torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark
    first = estimator.estimate(test)
    features = torch.from_numpy(estimator._encoder.conditioning_features(test, (0,), target=1))
    assert torch.equal(torch.cat(seen), features)
    assert len(seen[-1]) == len(test.values) % batch_size or len(seen[-1]) == batch_size
    seen.clear()
    second = estimator.estimate(test)
    assert first.entropy_nats == second.entropy_nats
    assert len(calls) == 1 and torch.equal(torch.cat(seen), features)
    assert random.getstate() == rng[0]
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
    assert np.random.get_state()[2:] == rng[1][2:]
    assert torch.equal(torch.get_rng_state(), rng[2])
    assert flags == (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
    assert dict(estimator._encoder.mappings) == mapping
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value)
    assert all(p.grad is None for p in model.parameters())


def test_no_test_access_in_fit_and_test_changes_only_heldout_terms(monkeypatch):
    (train, validation, test), config, alphabets = scenario(d=2)
    calls = []
    monkeypatch.setattr(neural, "train_conditional_term", controlled_trainer(calls))
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets)
    with pytest.raises(TypeError):
        estimator.fit(train, validation, test=test)
    estimator.fit(train, validation)
    low = estimator.estimate(replace(test, values=np.ones_like(test.values)))
    high = estimator.estimate(replace(test, values=np.zeros_like(test.values)))
    assert low.terms[0] == high.terms[0]
    assert low.entropy_nats < high.entropy_nats and len(calls) == 1


@pytest.mark.parametrize("change", [{"dataset_id": "wrong"}, {"split_id": "wrong"},
    {"n_samples": 24}, {"variable_ids": (1, 0)}, {"role": "validation"}])
def test_bad_test_identity_and_role(change):
    (train, validation, test), config, alphabets = scenario(d=2)
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets).fit(train, validation)
    with pytest.raises(ValueError):
        estimator.estimate(replace(test, **change))


@pytest.mark.parametrize("partition", ["train", "validation"])
def test_overlapping_test_partitions(partition):
    (train, validation, test), config, alphabets = scenario(d=2)
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets).fit(train, validation)
    rows = test.row_indices.copy()
    rows[0] = (train if partition == "train" else validation).row_indices[0]
    with pytest.raises(ValueError, match="overlap"):
        estimator.estimate(replace(test, row_indices=rows))


def test_incomplete_test_partition_and_lifecycle():
    (train, validation, test), config, alphabets = scenario(d=2)
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets)
    with pytest.raises(RuntimeError, match="before estimate"):
        estimator.estimate(test)
    with pytest.raises(TypeError):
        estimator.fit(None, validation)
    with pytest.raises(ValueError):
        estimator.fit(train, replace(validation, dataset_id="bad"))
    assert not estimator._fitted
    estimator.fit(train, validation)
    with pytest.raises(RuntimeError, match="already fitted"):
        estimator.fit(train, validation)
    with pytest.raises(ValueError, match="cover"):
        estimator.estimate(replace(test, values=test.values[:-1], row_indices=test.row_indices[:-1]))
    with pytest.raises(AttributeError):
        estimator.config = replace(config, component_order=(1, 0))


@pytest.mark.parametrize("order", [(0,), (0, 2), ("0", "1")])
def test_invalid_permutation(order):
    (train, validation, _), config, alphabets = scenario(d=2)
    estimator = neural.NeuralJointEntropyEstimator(replace(config, component_order=order), alphabets=alphabets)
    with pytest.raises(ValueError, match="every target"):
        estimator.fit(train, validation)


@pytest.mark.parametrize("role", ["train", "validation"])
def test_invalid_fit_roles_and_overlap(role):
    (train, validation, _), config, alphabets = scenario(d=2)
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets)
    if role == "train":
        with pytest.raises(ValueError, match="role"):
            estimator.fit(replace(train, role="test"), validation)
    else:
        rows = validation.row_indices.copy()
        rows[0] = train.row_indices[0]
        with pytest.raises(ValueError, match="overlap"):
            estimator.fit(train, replace(validation, row_indices=rows))


def test_declared_constant_and_true_singleton():
    (train, validation, test), config, alphabets = scenario(d=2, singleton=True)
    binary = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets).fit(train, validation).estimate(test)
    assert binary.terms[1].output_cardinality == 2 and binary.terms[1].epochs_trained > 0
    assert any("absent" in w for w in binary.terms[1].warnings)
    alphabets[1] = np.array([0])
    single = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets).fit(train, validation).estimate(test)
    assert single.terms[1].output_cardinality == 1
    assert single.terms[1].epochs_trained == 0 and single.terms[1].best_epoch is None
    assert single.terms[1].contribution_nats == 0.


def test_declared_unobserved_category_valid_in_validation_and_test():
    (train, validation, test), config, alphabets = scenario(d=2, singleton=True)
    validation_values, test_values = validation.values.copy(), test.values.copy()
    validation_values[:, 1] = 1
    test_values[:, 1] = 1
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets).fit(
        train, replace(validation, values=validation_values))
    report = estimator.estimate(replace(test, values=test_values))
    assert report.terms[1].output_cardinality == 2
    assert math.isfinite(report.terms[1].contribution_nats)
    assert any("absent from training" in w for w in report.terms[1].warnings)


@pytest.mark.parametrize("change", [{"dataset_id": "wrong"}, {"split_id": "wrong"},
    {"n_samples": 24}, {"variable_ids": (1, 0)}, {"role": "test"}])
def test_validation_identity_fails_before_any_training(monkeypatch, change):
    (train, validation, _), config, alphabets = scenario(d=2)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid validation must fail before training")
    monkeypatch.setattr(neural, "train_conditional_term", forbidden)
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets)
    with pytest.raises(ValueError):
        estimator.fit(train, replace(validation, **change))
    assert not estimator._fitted


@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("role", ["validation", "test"])
def test_train_observed_unknown_categories_without_expansion(d, role):
    (train, validation, test), config, _ = scenario(d=d, policy="train_observed", singleton=True)
    changed = (validation if role == "validation" else test).values.copy()
    changed[0, -1] = 9
    estimator = neural.NeuralJointEntropyEstimator(config)
    if role == "validation":
        with pytest.raises(ValueError, match="Unknown category"):
            estimator.fit(train, replace(validation, values=changed))
    else:
        estimator.fit(train, validation)
        before = dict(estimator._encoder.mappings)
        with pytest.raises(ValueError, match="Unknown category"):
            estimator.estimate(replace(test, values=changed))
        assert dict(estimator._encoder.mappings) == before


@pytest.mark.parametrize("failure", ["nan", "shape", "dtype", "loss", "training_mode"])
def test_invalid_outputs_and_losses_fail_restore_rng(monkeypatch, failure):
    (train, validation, test), config, alphabets = scenario(d=2)
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets).fit(train, validation)
    model = estimator._results[0].model
    if failure == "training_mode":
        model.train()
    elif failure == "loss":
        monkeypatch.setattr(torch.nn.functional, "cross_entropy",
                            lambda logits, y, **kw: torch.full((len(y),), float("nan")))
    else:
        def bad_forward(x):
            if failure == "nan":
                return torch.full((len(x), 2), float("nan"))
            if failure == "shape":
                return torch.zeros(len(x), 3)
            return torch.zeros(len(x), 2, dtype=torch.int64)
        monkeypatch.setattr(model, "forward", bad_forward)
    state = torch.get_rng_state()
    flags = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError):
        estimator.estimate(test)
    assert torch.equal(state, torch.get_rng_state())
    assert flags == torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("mode", ["seeded", "strict"])
def test_tiny_end_to_end_cpu_and_small_test(mode):
    (train, validation, test), config, alphabets = scenario(d=3, n=5)
    config = replace(config, reproducibility_mode=mode)
    estimator = neural.NeuralJointEntropyEstimator(config, alphabets=alphabets).fit(train, validation)
    report = estimator.estimate(test)
    assert len(test.values) == 1 and len(report.terms) == 3
    assert math.isfinite(report.entropy_nats) and report.entropy_nats >= 0
    assert report.entropy_nats == math.fsum(t.contribution_nats for t in report.terms)
    assert all(t.environment["reproducibility_mode"] == mode for t in report.terms[1:])
    assert estimator.estimate(test).entropy_nats == report.entropy_nats


def test_constructor_rejects_invalid_config_and_alphabets():
    _, config, alphabets = scenario(d=2)
    with pytest.raises(TypeError):
        neural.NeuralJointEntropyEstimator(None)
    with pytest.raises(ValueError):
        neural.NeuralJointEntropyEstimator(config)
    with pytest.raises(ValueError):
        neural.NeuralJointEntropyEstimator(config, alphabets={0: [0, 1]})
    with pytest.raises(ValueError):
        neural.NeuralJointEntropyEstimator(replace(config, alphabet_policy="train_observed"), alphabets=alphabets)
