"""Single-term training invariants, controlled checkpoints and leakage checks."""

from dataclasses import replace
import math
import random

import numpy as np
import pytest
import torch

from src.estimators import neural_training as training
from src.models.mlp import ConditionalMLP
from src.utils.categorical import CategoricalDataset, CategoricalEncoder
from src.utils.neural_config import NeuralConfig
from src.utils.reproducibility import derive_term_seed
from src.utils.split import make_iid_split


@pytest.fixture(autouse=True)
def preserve_rng_and_fast_cpu():
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.set_num_threads(threads)


def scenario(n=23, d=2, singleton=False):
    values = np.array([[i % 2] * d for i in range(n)], dtype=np.int64)
    if singleton:
        values[:, -1] = 0
    data = CategoricalDataset(values, "dataset", tuple(range(d)))
    split = make_iid_split(n, seed=42)
    train = data.batch(split, "train", split_id="split")
    validation = data.batch(split, "validation", split_id="split")
    alphabets = {v: np.array([0, 1]) for v in range(d)}
    if singleton:
        alphabets[d - 1] = np.array([0])
    encoder = CategoricalEncoder.fit(train, policy="declared", alphabets=alphabets)
    config = NeuralConfig(42, (0.6, 0.2, 0.2), 202, tuple(range(d)), "declared",
                          "train_plugin", (4,), "relu", "adam", 0.01, 4, 3, 2, 0.,
                          "cpu", "strict", 0)
    return train, validation, encoder, config


def fit(train, validation, encoder, config, *, target=1, conditioning=(0,)):
    return training.train_conditional_term(train, validation, encoder=encoder,
                                           config=config, target=target, conditioning=conditioning)


def clone(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_real_adam_updates_train_only_and_validation_no_grad(monkeypatch):
    train, validation, encoder, config = scenario()
    initial = {}
    steps = []
    original_model, original_step, original_evaluate = ConditionalMLP, torch.optim.Adam.step, training._evaluate

    def model_factory(*args, **kwargs):
        model = original_model(*args, **kwargs)
        initial.update(clone(model))
        return model

    def step(optimizer, *args, **kwargs):
        assert torch.is_grad_enabled()
        steps.append(1)
        return original_step(optimizer, *args, **kwargs)

    def evaluate(model, loader, device):
        before, step_count = clone(model), len(steps)
        calls = []
        hook = model.register_forward_pre_hook(lambda module, args: calls.append(
            (module.training, torch.is_grad_enabled())))
        value = original_evaluate(model, loader, device)
        hook.remove()
        assert calls and all(call == (False, False) for call in calls)
        assert len(steps) == step_count
        for key in before:
            assert torch.equal(before[key], model.state_dict()[key])
        return value

    monkeypatch.setattr(training, "ConditionalMLP", model_factory)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    monkeypatch.setattr(training, "_evaluate", evaluate)
    result = fit(train, validation, encoder, config)
    assert len(steps) == math.ceil(len(train.values) / config.batch_size) * result.epochs_trained
    assert any(not torch.equal(initial[k], result.model.state_dict()[k]) for k in initial)
    assert not result.model.training
    assert all(not p.requires_grad and p.grad is None for p in result.model.parameters())
    assert result.environment["reproducibility_mode"] == "strict"


@pytest.mark.parametrize("batch_size,n", [(1, 5), (4, 23), (100, 5)])
def test_batch_size_one_small_dataset_and_incomplete_batches(batch_size, n):
    train, validation, encoder, config = scenario(n=n)
    result = fit(train, validation, encoder, replace(config, batch_size=batch_size, max_epochs=1))
    assert result.epochs_trained == result.best_epoch == 1
    assert math.isfinite(result.final_training_ce_nats)
    assert result.stop_reason == "max_epochs"
    assert result.output_cardinality == 2 and result.runtime_seconds >= 0


def test_loss_aggregation_is_per_observation_for_incomplete_batches(monkeypatch):
    train, validation, encoder, config = scenario(n=23)

    def fixed_model(*args, **kwargs):
        model = ConditionalMLP(*args, **kwargs)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.network[-1].bias.copy_(torch.tensor([0., math.log(3)]))
        return model

    monkeypatch.setattr(training, "ConditionalMLP", fixed_model)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *args, **kwargs: None)
    result = fit(train, validation, encoder, replace(config, max_epochs=1))

    def expected(batch):
        labels = encoder.target_indices(batch, 1)
        return sum(math.log(4) if y == 0 else math.log(4/3) for y in labels) / len(labels)

    assert result.epoch_training_ce_nats[0] == pytest.approx(expected(train), abs=1e-7)
    assert result.best_validation_ce_nats == pytest.approx(expected(validation), abs=1e-7)
    assert result.final_training_ce_nats == pytest.approx(expected(train), abs=1e-7)


@pytest.mark.parametrize("sequence,delta,patience,best_epoch,epochs,reason", [
    ([3., 2., 2., 2.], 0., 2, 2, 4, "early_stopping"),
    ([2., 2., 2.], 0., 1, 1, 2, "early_stopping"),
    ([2., 1.875, 1.75, 1.75], 0.125, 2, 3, 4, "max_epochs"),
    ([2., 1.875, 1.875], 0.125, 2, 1, 3, "early_stopping"),
    ([4., 3., 2., 1.], 0., 1, 4, 4, "max_epochs"),
])
def test_checkpoint_patience_min_delta_ties_and_restoration(
    monkeypatch, sequence, delta, patience, best_epoch, epochs, reason,
):
    train, validation, encoder, config = scenario()
    states = []

    def controlled_evaluate(model, loader, device):
        if len(states) < epochs:
            states.append(clone(model))
            return sequence[len(states) - 1]
        # Final training evaluation must already see the independently saved best.
        for key in states[best_epoch - 1]:
            assert torch.equal(states[best_epoch - 1][key], model.state_dict()[key])
        return 0.123

    monkeypatch.setattr(training, "_evaluate", controlled_evaluate)
    result = fit(train, validation, encoder, replace(config, min_delta=delta, patience=patience,
                                                     max_epochs=len(sequence)))
    assert result.best_epoch == best_epoch and result.epochs_trained == epochs
    assert result.best_validation_ce_nats == sequence[best_epoch - 1]
    assert result.stop_reason == reason
    assert result.final_training_ce_nats == 0.123
    for key in states[best_epoch - 1]:
        assert torch.equal(result.model.state_dict()[key], states[best_epoch - 1][key])
    if best_epoch < epochs:
        assert any(not torch.equal(states[-1][key], states[best_epoch - 1][key]) for key in states[-1])


def test_repeatable_results_initialization_and_shuffle(monkeypatch):
    train, validation, encoder, config = scenario()
    orders = []
    original_loader = training.DataLoader

    def traced_loader(*args, **kwargs):
        loader = original_loader(*args, **kwargs)
        if kwargs["shuffle"]:
            # Trace batch-sampler indices without consuming the actual generator.
            state = kwargs["generator"].get_state()
            orders.append(list(loader.batch_sampler))
            kwargs["generator"].set_state(state)
        else:
            assert list(loader.sampler) == list(range(len(loader.dataset)))
        assert kwargs["drop_last"] is False and kwargs["num_workers"] == 0
        return loader

    monkeypatch.setattr(training, "DataLoader", traced_loader)
    first, second = fit(train, validation, encoder, config), fit(train, validation, encoder, config)
    assert orders[0] == orders[1]
    assert sorted(sum(orders[0], [])) == list(range(len(train.values)))
    assert first.initialization_seed == derive_term_seed(202, first.term_id, purpose="initialization")
    assert first.shuffle_seed == derive_term_seed(202, first.term_id, purpose="shuffle")
    assert first.initialization_seed != first.shuffle_seed
    assert first.epoch_training_ce_nats == second.epoch_training_ce_nats
    assert first.epoch_validation_ce_nats == second.epoch_validation_ce_nats
    assert first.final_training_ce_nats == second.final_training_ce_nats
    assert first.best_epoch == second.best_epoch
    for key in first.model.state_dict():
        assert torch.equal(first.model.state_dict()[key], second.model.state_dict()[key])


def test_term_seed_distinguishes_conditioning_order_and_typed_targets():
    train, validation, encoder, config = scenario(d=3)
    first = fit(train, validation, encoder, replace(config, max_epochs=1), target=2, conditioning=(0, 1))
    second = fit(train, validation, encoder, replace(config, component_order=(1, 0, 2), max_epochs=1),
                 target=2, conditioning=(1, 0))
    assert first.term_id != second.term_id and first.initialization_seed != second.initialization_seed
    renamed_train = replace(train, variable_ids=(0, 1, "2"))
    renamed_validation = replace(validation, variable_ids=(0, 1, "2"))
    renamed_encoder = CategoricalEncoder.fit(renamed_train, policy="declared", alphabets={
        0: np.array([0, 1]), 1: np.array([0, 1]), "2": np.array([0, 1]),
    })
    third = fit(renamed_train, renamed_validation, renamed_encoder,
                replace(config, component_order=(0, 1, "2"), max_epochs=1), target="2", conditioning=(0, 1))
    assert first.term_id != third.term_id and first.initialization_seed != third.initialization_seed


def test_no_test_access_no_encoder_refit_or_mapping_mutation(monkeypatch):
    train, validation, encoder, config = scenario()
    mappings = dict(encoder.mappings)
    original_features, original_targets = encoder.conditioning_features, encoder.target_indices
    calls = []

    def features(batch, conditioning, **kwargs):
        calls.append(batch.role)
        assert batch.role in ("train", "validation")
        assert kwargs["target"] not in conditioning
        result = original_features(batch, conditioning, **kwargs)
        assert result.shape[1] == 2  # No metadata/row IDs among feature blocks.
        return result

    def targets(batch, variable):
        assert batch.role in ("train", "validation")
        return original_targets(batch, variable)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Encoder refitting is forbidden")

    monkeypatch.setattr(encoder, "conditioning_features", features)
    monkeypatch.setattr(encoder, "target_indices", targets)
    monkeypatch.setattr(CategoricalEncoder, "fit", forbidden_fit)
    fit(train, validation, encoder, config)
    assert calls == ["train", "validation"] and dict(encoder.mappings) == mappings
    class ForbiddenTest:
        def __getattribute__(self, name):
            raise AssertionError("Test data accessed")
    with pytest.raises(TypeError):
        training.train_conditional_term(train, validation, encoder=encoder, config=config,
                                         target=1, conditioning=(0,), test=ForbiddenTest())


def test_changed_test_values_cannot_influence_fit():
    train, validation, encoder, config = scenario()
    first = fit(train, validation, encoder, config)
    # Reconstruct the same dataset with adversarial categories exclusively in TEST.
    values = np.array([[i % 2] * 2 for i in range(train.n_samples)], dtype=np.int64)
    split = make_iid_split(train.n_samples, seed=config.split_seed)
    values[split.test] = 2**53 + 1
    changed = CategoricalDataset(values, train.dataset_id, train.variable_ids)
    changed_train = changed.batch(split, "train", split_id=train.split_id)
    changed_validation = changed.batch(split, "validation", split_id=train.split_id)
    np.testing.assert_array_equal(changed_train.values, train.values)
    np.testing.assert_array_equal(changed_validation.values, validation.values)
    second = fit(changed_train, changed_validation, encoder, config)
    assert first.epoch_validation_ce_nats == second.epoch_validation_ce_nats
    for key in first.model.state_dict():
        assert torch.equal(first.model.state_dict()[key], second.model.state_dict()[key])


def test_initialization_is_immediately_seeded_and_model_has_no_metadata(monkeypatch):
    train, validation, encoder, config = scenario()
    initial, seen = {}, []

    def model_factory(*args, **kwargs):
        seen.append((args, kwargs, torch.initial_seed()))
        model = ConditionalMLP(*args, **kwargs)
        initial.update(clone(model))
        return model

    monkeypatch.setattr(training, "ConditionalMLP", model_factory)
    result = fit(train, validation, encoder, config)
    assert seen == [((2, 2), {"hidden_sizes": (4,)}, result.initialization_seed)]
    from src.utils.reproducibility import reproducible_execution
    with reproducible_execution(result.initialization_seed, mode="strict"):
        reference = ConditionalMLP(2, 2, hidden_sizes=(4,))
    for key in initial:
        assert torch.equal(initial[key], reference.state_dict()[key])


def test_invalid_api_inputs_and_completed_model_cannot_optimize():
    train, validation, encoder, config = scenario()
    with pytest.raises(TypeError):
        fit(None, validation, encoder, config)
    with pytest.raises(TypeError):
        fit(train, validation, encoder, None)
    with pytest.raises(TypeError):
        fit(train, validation, None, config)
    with pytest.raises(ValueError, match="roles"):
        fit(replace(train, role="validation"), validation, encoder, config)
    result = fit(train, validation, encoder, config)
    features = torch.from_numpy(encoder.conditioning_features(train, (0,), target=1))
    loss = torch.nn.functional.cross_entropy(result.model(features),
                                              torch.from_numpy(encoder.target_indices(train, 1)))
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()


@pytest.mark.parametrize("conditioning,target", [((1,), 1), ((), 1), ((0, 0), 1),
    ((9,), 1), ([0], 1), ((True,), 1), ((0,), 9), ((0,), True), ((0,), 0)])
def test_invalid_conditioning_or_target(conditioning, target):
    train, validation, encoder, config = scenario()
    with pytest.raises(ValueError):
        fit(train, validation, encoder, config, target=target, conditioning=conditioning)


@pytest.mark.parametrize("change", [{"split_id": "other"}, {"dataset_id": "other"},
    {"variable_ids": (1, 0)}, {"role": "test"}])
def test_invalid_validation_identity_or_role(change):
    train, validation, encoder, config = scenario()
    with pytest.raises(ValueError):
        fit(train, replace(validation, **change), encoder, config)


def test_train_validation_overlap_and_unfitted_encoder():
    train, validation, encoder, config = scenario()
    rows = validation.row_indices.copy()
    rows[0] = train.row_indices[0]
    with pytest.raises(ValueError, match="overlap"):
        fit(train, replace(validation, row_indices=rows), encoder, config)
    with pytest.raises(ValueError, match="fitted"):
        fit(train, validation, CategoricalEncoder(), config)
    with pytest.raises(ValueError, match="policy"):
        fit(train, validation, encoder, replace(config, alphabet_policy="train_observed"))


def test_singleton_no_optimizer_and_declared_constant_not_collapsed(monkeypatch):
    train, validation, encoder, config = scenario(singleton=True)

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("Singleton must not optimize")

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim, "Adam", forbidden_optimizer)
        result = fit(train, validation, encoder, config)
    assert result.epochs_trained == 0 and result.best_epoch is None
    assert result.best_validation_ce_nats == result.final_training_ce_nats == 0.
    assert result.stop_reason == "singleton_target" and result.output_cardinality == 1
    assert result.model(torch.tensor([[1., 0.]])).shape == (1, 1)
    binary_encoder = CategoricalEncoder.fit(train, policy="declared",
                                            alphabets={0: np.array([0, 1]), 1: np.array([0, 1])})
    binary = fit(train, validation, binary_encoder, config)
    assert binary.output_cardinality == 2 and binary.epochs_trained > 0
    assert any("absent from training" in warning for warning in binary.warnings)


def test_unknown_validation_category_fails_without_expanding_mapping():
    train, validation, _, config = scenario(singleton=True)
    encoder = CategoricalEncoder.fit(train)
    values = validation.values.copy()
    values[0, 1] = 9
    with pytest.raises(ValueError, match="category 9.*variable 1.*validation"):
        fit(train, replace(validation, values=values), encoder,
            replace(config, alphabet_policy="train_observed"))
    assert encoder.cardinalities[1] == 1


@pytest.mark.parametrize("failure", ["loss", "gradient", "parameter", "validation"])
def test_numerical_failures_raise_without_success_result(monkeypatch, failure):
    train, validation, encoder, config = scenario()
    if failure == "loss":
        original = torch.nn.functional.cross_entropy
        monkeypatch.setattr(torch.nn.functional, "cross_entropy",
                            lambda *args, **kwargs: original(*args, **kwargs) * float("nan"))
    elif failure == "gradient":
        def model_factory(*args, **kwargs):
            model = ConditionalMLP(*args, **kwargs)
            next(model.parameters()).register_hook(lambda grad: grad * float("nan"))
            return model
        monkeypatch.setattr(training, "ConditionalMLP", model_factory)
    elif failure == "parameter":
        def broken_step(optimizer, *args, **kwargs):
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("nan"))
        monkeypatch.setattr(torch.optim.Adam, "step", broken_step)
    else:
        monkeypatch.setattr(training, "_evaluate", lambda *args: float("nan"))
    before = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError, match="Nonfinite"):
        fit(train, validation, encoder, config)
    assert torch.are_deterministic_algorithms_enabled() == before
