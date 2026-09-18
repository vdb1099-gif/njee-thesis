import numpy as np

from src.simulation.bernoulli import generate_independent_bernoulli


def test_shape():
    X = generate_independent_bernoulli(
        n=100,
        d=8,
        p=0.5,
        seed=42,
    )

    assert X.shape == (100, 8)


def test_binary_values():
    X = generate_independent_bernoulli(
        n=100,
        d=8,
        p=0.5,
        seed=42,
    )

    assert set(np.unique(X)).issubset({0, 1})


def test_reproducibility():
    X1 = generate_independent_bernoulli(
        n=100,
        d=8,
        p=0.5,
        seed=42,
    )

    X2 = generate_independent_bernoulli(
        n=100,
        d=8,
        p=0.5,
        seed=42,
    )

    assert np.array_equal(X1, X2)


def test_different_seeds():
    X1 = generate_independent_bernoulli(
        n=100,
        d=8,
        p=0.5,
        seed=42,
    )

    X2 = generate_independent_bernoulli(
        n=100,
        d=8,
        p=0.5,
        seed=123,
    )

    assert not np.array_equal(X1, X2)
    

def test_empirical_mean_close_to_p():
    X = generate_independent_bernoulli(
        n=100_000,
        d=4,
        p=0.3,
        seed=42,
    )

    assert abs(X.mean() - 0.3) < 0.01