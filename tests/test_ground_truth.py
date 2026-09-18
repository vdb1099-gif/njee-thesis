import numpy as np

from src.simulation.ground_truth import (
    binary_entropy,
    independent_bernoulli_entropy,
    binary_chain_entropy,
)


def test_binary_entropy_deterministic():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0


def test_binary_entropy_half():
    assert np.isclose(
        binary_entropy(0.5),
        np.log(2),
    )


def test_independent_bernoulli_entropy():
    H = independent_bernoulli_entropy(
        p=0.5,
        d=8,
    )

    assert np.isclose(
        H,
        8 * np.log(2),
    )


def test_binary_chain_entropy_epsilon_zero():
    H = binary_chain_entropy(
        epsilon=0.0,
        d=8,
    )

    assert np.isclose(
        H,
        np.log(2),
    )


def test_binary_chain_entropy_epsilon_half():
    H = binary_chain_entropy(
        epsilon=0.5,
        d=8,
    )

    assert np.isclose(
        H,
        8 * np.log(2),
    )