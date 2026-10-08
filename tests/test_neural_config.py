"""Strict standalone neural configuration validation without training."""

from dataclasses import FrozenInstanceError, asdict, replace

import pytest
import yaml

from src.utils.config import load_config
from src.utils.neural_config import NeuralConfig, load_neural_config, validate_neural_config


def valid_config():
    return dict(split_seed=101, split_fractions=[0.6, 0.2, 0.2], training_seed=202,
                component_order=[0, 1], alphabet_policy="declared", marginal_method="train_plugin",
                hidden_sizes=[64, 64], activation="relu", optimizer="adam", learning_rate=0.001,
                batch_size=64, max_epochs=200, patience=20, min_delta=0.0, device="cpu",
                reproducibility_mode="strict", num_workers=0)


def test_valid_configuration_and_no_mutation():
    raw = valid_config()
    config = validate_neural_config(raw, variable_ids=(0, 1), n_samples=100)
    assert isinstance(config, NeuralConfig)
    assert config.split_fractions == (0.6, 0.2, 0.2)
    assert config.component_order == (0, 1) and config.hidden_sizes == (64, 64)
    assert raw["component_order"] == [0, 1]
    with pytest.raises(FrozenInstanceError):
        config.training_seed = 3


@pytest.mark.parametrize("field", ["split_seed", "training_seed"])
@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_valid_seed_boundaries(field, seed):
    raw = valid_config()
    raw[field] = seed
    assert getattr(validate_neural_config(raw), field) == seed


@pytest.mark.parametrize("field", ["split_seed", "training_seed"])
@pytest.mark.parametrize("seed", [-1, 2**32, True, False, 1.0, "1", None])
def test_invalid_seed_boundaries(field, seed):
    raw = valid_config()
    raw[field] = seed
    with pytest.raises(ValueError, match=field):
        validate_neural_config(raw)


@pytest.mark.parametrize("field,value", [
    ("split_fractions", [0.5, 0.2, 0.2]), ("split_fractions", [0, 0.5, 0.5]),
    ("split_fractions", [True, 0.2, 0.2]), ("split_fractions", [float("nan"), 0.2, 0.2]),
    ("split_fractions", [float("inf"), 0.2, 0.2]), ("split_fractions", "bad"),
    ("split_fractions", [0.5, 0.5]), ("hidden_sizes", []), ("hidden_sizes", [1, 2, 3]),
    ("hidden_sizes", [0]), ("hidden_sizes", [True]), ("hidden_sizes", [64.0]),
    ("activation", "tanh"), ("optimizer", "sgd"), ("marginal_method", "test_plugin"),
    ("alphabet_policy", "unknown"), ("reproducibility_mode", "automatic"),
    ("learning_rate", 0), ("learning_rate", -1), ("learning_rate", float("inf")),
    ("learning_rate", float("nan")), ("learning_rate", True), ("learning_rate", "0.01"),
    ("learning_rate", 10**400), ("min_delta", -1), ("min_delta", float("nan")),
    ("min_delta", True), ("batch_size", 0), ("batch_size", True), ("batch_size", 2.0),
    ("max_epochs", -1), ("max_epochs", "10"), ("patience", 0), ("patience", False),
    ("num_workers", 1), ("num_workers", -1), ("num_workers", False), ("num_workers", 0.0),
    ("device", "auto"), ("device", "mps"), ("device", "cuda:-1"), ("device", "cuda:01"),
])
def test_invalid_fields(field, value):
    raw = valid_config()
    raw[field] = value
    with pytest.raises(ValueError):
        validate_neural_config(raw)


@pytest.mark.parametrize("order", [[], [0, 0], [True, 1], [0, 2], [0], [None, 1]])
def test_invalid_permutations(order):
    raw = valid_config()
    raw["component_order"] = order
    with pytest.raises(ValueError):
        validate_neural_config(raw, variable_ids=(0, 1))


def test_dataset_checks_reuse_existing_split_and_variable_contracts():
    raw = valid_config()
    raw["component_order"] = ["Y", "X"]
    config = validate_neural_config(raw, variable_ids=("X", "Y"), n_samples=10)
    assert config.component_order == ("Y", "X")
    with pytest.raises(ValueError, match="empty"):
        config.validate_dataset(variable_ids=("X", "Y"), n_samples=3)
    with pytest.raises(ValueError, match="n_samples"):
        validate_neural_config(raw, n_samples=True)


def test_optional_schema_and_supported_choices():
    raw = valid_config()
    raw.update(component_order=["only"], hidden_sizes=[8], alphabet_policy="train_observed",
               reproducibility_mode="seeded", device="cuda:0", min_delta=1, learning_rate=1)
    assert validate_neural_config(raw).device == "cuda:0"  # No device initialization here.
    raw["split_fractions"] = [0.6, 0.2, 0.2000000000001]
    validate_neural_config(raw)


@pytest.mark.parametrize("field", list(valid_config()))
def test_every_field_required(field):
    raw = valid_config()
    del raw[field]
    with pytest.raises(ValueError, match="missing"):
        validate_neural_config(raw)


@pytest.mark.parametrize("raw", [None, [], {}, {**valid_config(), "shuffle_validation": True}])
def test_invalid_mapping_or_unknown_options(raw):
    with pytest.raises(ValueError):
        validate_neural_config(raw)


def test_yaml_round_trip_and_classical_schema_unchanged(tmp_path):
    path = tmp_path / "neural.yaml"
    path.write_text(yaml.safe_dump(valid_config()), encoding="utf-8")
    config = load_neural_config(path, variable_ids=(0, 1), n_samples=100)
    path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
    assert load_neural_config(path) == config
    assert load_config("configs/baseline.yaml")["data"]["generator"] == "binary_chain"
    with pytest.raises(ValueError):
        replace(config, optimizer="sgd")


@pytest.mark.parametrize("contents", ["", "[]", "x: [", "training_seed: -1"])
def test_yaml_errors_include_path(tmp_path, contents):
    path = tmp_path / "bad.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="bad.yaml"):
        load_neural_config(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_neural_config(tmp_path / "missing.yaml")


def test_config_validation_does_not_inspect_cuda_or_seed_rngs(monkeypatch):
    import torch
    from src.utils import reproducibility

    def forbidden(*args, **kwargs):
        raise AssertionError("Configuration must not initialize hardware or seed RNGs")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(reproducibility, "set_seed", forbidden)
    raw = valid_config()
    raw["device"] = "cuda:0"
    validate_neural_config(raw, variable_ids=(0, 1), n_samples=100)
