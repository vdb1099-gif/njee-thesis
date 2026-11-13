"""Exact feature/gradient contracts and tiny CPU correctness fits, not convergence."""
from dataclasses import replace
import math
import random

import numpy as np
import pytest
import torch

from experiments.ablation_protocol import config_for, cells
from src.estimators import ablation
from src.models.ablation_shared import AblationMultiHead, fixed_slot_features
from src.utils.ablation_data import nested_training_subset, discretize_uniform, discretized_entropy
from src.utils.categorical import CategoricalDataset, CategoricalEncoder
from src.utils.split import make_iid_split


@pytest.fixture(autouse=True)
def state():
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
    torch.set_num_threads(threads)


def batches(d=3):
    values = np.array([[(i + j) % 2 for j in range(d)] for i in range(40)])
    data = CategoricalDataset(values, 'data', tuple(range(d)))
    split = make_iid_split(40, seed=42)
    return tuple(data.batch(split, r, split_id='split') for r in ('train', 'validation', 'test'))


def test_nested_training_only_subsets_and_invalid_values():
    train, val, test = batches()
    a = nested_training_subset(train, .25, seed=42)
    b = nested_training_subset(train, .5, seed=42)
    assert set(a.row_indices) <= set(b.row_indices) <= set(train.row_indices)
    assert len(a.values) == 6 and a.dataset_id == train.dataset_id and a.n_samples == 40
    np.testing.assert_array_equal(a.values, nested_training_subset(train, .25, seed=42).values)
    for bad in (True, 0, -.1, 2, float('nan')):
        with pytest.raises(ValueError): nested_training_subset(train, bad, seed=42)
    with pytest.raises(ValueError): nested_training_subset(val, .5, seed=42)
    assert not set(b.row_indices) & (set(val.row_indices) | set(test.row_indices))


@pytest.mark.parametrize('k', [2, 3, 5])
def test_analytic_bins_boundaries_and_truth(k):
    values = np.array([[0., 1 / k, np.nextafter(1., 0.)]])
    np.testing.assert_array_equal(discretize_uniform(values, k), [[0, 1, k - 1]])
    assert discretized_entropy(6, k) == pytest.approx(6 * math.log(k))
    # The declared K survives a training sample with unobserved categories.
    data = CategoricalDataset(np.zeros((5, 2), dtype=int), 'd', (0, 1))
    train = data.batch(make_iid_split(5, seed=0), 'train', split_id='s')
    e = CategoricalEncoder.fit(train, policy='declared', alphabets={v: np.arange(k) for v in (0, 1)})
    assert e.cardinalities[0] == k


@pytest.mark.parametrize('bad', [np.array([[1.]]), np.array([[float('nan')]]), np.array([0.]), np.array([[0]])])
def test_invalid_discretization(bad):
    with pytest.raises(ValueError): discretize_uniform(bad, 3)


def test_fixed_slots_no_target_or_future_leakage_and_term_identity():
    train, _, _ = batches()
    encoder = CategoricalEncoder.fit(train, policy='declared', alphabets={i: np.arange(2) for i in range(3)})
    order = (2, 0, 1)
    features = fixed_slot_features(encoder, train, order, 1)
    assert features.shape == (24, 8)
    assert not features[:, :4].any()  # target 0 and future 1 slots are absent
    assert np.all(features[:, 4:6].sum(axis=1) == 1) and np.all(features[:, 6] == 1)
    altered = replace(train, values=train.values.copy() ^ np.array([1, 1, 0]))
    np.testing.assert_array_equal(features, fixed_slot_features(encoder, altered, order, 1))


def test_sharing_gradient_semantics_capacity_and_logits():
    shared = AblationMultiHead(3, 2, hidden_sizes=(4,), shared=True)
    separate = AblationMultiHead(3, 2, hidden_sizes=(4,), shared=False)
    assert len(shared.trunks) == 1 and len(separate.trunks) == 2
    assert sum(p.numel() for p in shared.parameters()) < sum(p.numel() for p in separate.parameters())
    x = torch.ones(2, 8)
    loss = torch.nn.functional.cross_entropy(shared(x, 1), torch.tensor([0, 1]))
    loss.backward()
    assert any(p.grad is not None for p in shared.trunks[0].parameters())
    assert all(p.grad is None for p in shared.heads[0].parameters())
    separate(x, 1).sum().backward()
    assert all(p.grad is None for p in separate.trunks[0].parameters())
    with pytest.raises(TypeError): shared(x.double(), 0)
    with pytest.raises(ValueError): shared(torch.ones(1, 7), 0)
    with pytest.raises(ValueError): shared(x, 2)


@pytest.mark.parametrize('variant', ['native', 'sharing_control', 'shared'])
@pytest.mark.parametrize('early', [True, False])
def test_tiny_integration_reconstruction_and_checkpoints(variant, early):
    train, val, test = batches()
    config = replace(config_for(cells()[0], 53), component_order=(2, 0, 1), hidden_sizes=(4,),
                     max_epochs=1, batch_size=8)
    report, checkpoints = ablation.estimate_entropy(train, val, test, config=config,
        alphabets={i: np.arange(2) for i in range(3)}, variant=variant, early_stopping=early)
    assert report['entropy_nats'] == math.fsum([report['marginal_train_nats'], *[t['test_ce_nats'] for t in report['terms']]])
    assert report['optimizer_updates'] == (6 if variant == 'native' else 3)
    assert report['term_batch_evaluations'] == 6
    assert len(checkpoints) == (2 if variant == 'native' else 1)
    assert all(t['selected_epoch'] == 1 for t in report['terms'])
    again, second = ablation.estimate_entropy(train, val, test, config=config,
        alphabets={i: np.arange(2) for i in range(3)}, variant=variant, early_stopping=early)
    assert report['entropy_nats'] == again['entropy_nats']
    for key, arrays in checkpoints.items():
        for name in arrays: np.testing.assert_array_equal(arrays[name], second[key][name])


def test_off_last_checkpoint_vs_validation_selection(monkeypatch):
    # Controlled validation CE independent of TRAIN optimizer losses. No test data.
    config = replace(config_for(cells()[0], 42), max_epochs=4, patience=1)
    original = torch.nn.functional.cross_entropy
    validation_calls = []
    def controlled(logits, targets, **kwargs):
        if not torch.is_grad_enabled():
            validation_calls.append(1)
            return torch.tensor(float(len(validation_calls)))
        return original(logits, targets, **kwargs)
    monkeypatch.setattr(torch.nn.functional, 'cross_entropy', controlled)
    x, y = [np.ones((4, 2), dtype=np.float32)], [np.array([0, 1, 0, 1])]
    from src.models.mlp import ConditionalMLP
    import copy
    torch.manual_seed(42)
    template = ConditionalMLP(2, 2, hidden_sizes=(4,))
    on_model, off_model, first_model = (copy.deepcopy(template) for _ in range(3))
    on = ablation.fit_arrays(on_model, x, y, x, y,
        config=config, shuffle_seed=1, early_stopping=True)
    validation_calls.clear()
    off = ablation.fit_arrays(off_model, x, y, x, y,
        config=config, shuffle_seed=1, early_stopping=False)
    validation_calls.clear()
    ablation.fit_arrays(first_model, x, y, x, y, config=replace(config, max_epochs=1),
                        shuffle_seed=1, early_stopping=False)
    assert on['epochs'] == 2 and on['selected_epoch'] == 1
    assert off['epochs'] == 4 and off['selected_epoch'] == 4 and off['best_validation_epoch'] == 1
    assert all(torch.equal(a, b) for a, b in zip(on_model.parameters(), first_model.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(off_model.parameters(), first_model.parameters()))


def test_partition_overlap_and_identity_rejected():
    train, val, test = batches()
    config = replace(config_for(cells()[0], 0), component_order=(0, 1, 2))
    with pytest.raises(ValueError, match='Overlapping'):
        ablation.estimate_entropy(train, val, replace(test, row_indices=train.row_indices[:len(test.values)]),
            config=config, alphabets={i: np.arange(2) for i in range(3)})


@pytest.mark.parametrize('sizes', [(8,), (16,), (32,), (16, 16), (32, 32)])
def test_declared_shared_architecture_levels(sizes):
    model = AblationMultiHead(6, 2, hidden_sizes=sizes)
    assert model(torch.zeros(1, 17), 4).shape == (1, 2)
    assert sum(isinstance(m, torch.nn.ReLU) for m in model.trunks[0]) == len(sizes)


def test_native_reference_matches_validated_njee():
    from src.estimators.neural import NeuralJointEntropyEstimator
    train, val, test = batches()
    config = replace(config_for(cells()[0], 97), component_order=(2, 0, 1), hidden_sizes=(4,),
                     max_epochs=1, batch_size=8)
    alphabets = {i: np.arange(2) for i in range(3)}
    validated = NeuralJointEntropyEstimator(config, alphabets=alphabets).fit(train, val).estimate(test)
    experimental, _ = ablation.estimate_entropy(train, val, test, config=config, alphabets=alphabets)
    assert validated.entropy_nats == experimental['entropy_nats']
    assert [t.contribution_nats for t in validated.terms[1:]] == [t['test_ce_nats'] for t in experimental['terms']]
