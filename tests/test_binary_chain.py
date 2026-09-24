import numpy as np
import pytest

from src.simulation.binary_chain import generate_binary_chain


def test_shape():
    X = generate_binary_chain(
        n=100,
        d=8,
        epsilon=0.15,
        seed=42,
    )

    assert X.shape == (100, 8)


def test_binary_values():
    X = generate_binary_chain(
        n=100,
        d=8,
        epsilon=0.15,
        seed=42,
    )

    assert set(np.unique(X)).issubset({0, 1})


def test_reproducibility():
    X1 = generate_binary_chain(
        n=100,
        d=8,
        epsilon=0.15,
        seed=42,
    )

    X2 = generate_binary_chain(
        n=100,
        d=8,
        epsilon=0.15,
        seed=42,
    )

    assert np.array_equal(X1, X2)


def test_different_seeds():
    X1 = generate_binary_chain(
        n=100,
        d=8,
        epsilon=0.15,
        seed=42,
    )

    X2 = generate_binary_chain(
        n=100,
        d=8,
        epsilon=0.15,
        seed=123,
    )

    assert not np.array_equal(X1, X2)


def test_empirical_flip_rate():
    epsilon = 0.15

    X = generate_binary_chain(
        n=10_000,
        d=8,
        epsilon=epsilon,
        seed=42,
    )

    flip_rate = (X[:, 1:] != X[:, :-1]).mean()

    assert abs(flip_rate - epsilon) < 0.02


def test_epsilon_zero():
    X = generate_binary_chain(
        n=100,
        d=8,
        epsilon=0.0,
        seed=42,
    )

    assert np.all(X == X[:, [0]])


def test_epsilon_one():
    X = generate_binary_chain(
        n=100,
        d=8,
        epsilon=1.0,
        seed=42,
    )

    assert np.all(X[:, 1:] != X[:, :-1])



@pytest.mark.parametrize(
    "n",
    [0, -1, -10],
)
def test_invalid_n(n):
    with pytest.raises(ValueError):
        generate_binary_chain(
            n=n,
            d=4,
            epsilon=0.15,
            seed=42,
        )


@pytest.mark.parametrize(
    "d",
    [0, -1, -10],
)
def test_invalid_d(d):
    with pytest.raises(ValueError):
        generate_binary_chain(
            n=100,
            d=d,
            epsilon=0.15,
            seed=42,
        )


@pytest.mark.parametrize(
    "epsilon",
    [-0.1, 1.1, -1, 2],
)
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        generate_binary_chain(
            n=100,
            d=4,
            epsilon=epsilon,
            seed=42,
        )


def test_d_equals_one():
    X = generate_binary_chain(
        n=100,
        d=1,
        epsilon=0.15,
        seed=42,
    )

    assert X.shape == (100, 1)