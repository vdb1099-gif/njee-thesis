import pytest

from src.simulation.factory import (
    generate_dataset,
    get_ground_truth,
)


def test_factory_independent_bernoulli():
    config = {
        "seed": 42,
        "data": {
            "generator": "independent_bernoulli",
            "n": 100,
            "d": 4,
            "params": {
                "p": 0.5,
            },
        },
    }

    X = generate_dataset(config)
    H = get_ground_truth(config)

    assert X.shape == (100, 4)
    assert H > 0


def test_factory_binary_chain():
    config = {
        "seed": 42,
        "data": {
            "generator": "binary_chain",
            "n": 100,
            "d": 4,
            "params": {
                "epsilon": 0.15,
            },
        },
    }

    X = generate_dataset(config)
    H = get_ground_truth(config)

    assert X.shape == (100, 4)
    assert H > 0


def test_unknown_generator():
    config = {
        "seed": 42,
        "data": {
            "generator": "wrong_generator",
            "n": 100,
            "d": 4,
            "params": {},
        },
    }

    with pytest.raises(ValueError):
        generate_dataset(config)