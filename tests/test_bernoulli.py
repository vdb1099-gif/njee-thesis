import numpy as np
import pytest

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
        n=10_000,
        d=4,
        p=0.5,
        seed=42,
    )

    assert abs(X.mean() - 0.5) < 0.02



@pytest.mark.parametrize(
    "n",
    [0, -1, -10],
)
def test_invalid_n(n):
    with pytest.raises(ValueError):
        generate_independent_bernoulli(
            n=n,
            d=4,
            p=0.5,
            seed=42,
        )


@pytest.mark.parametrize(
    "d",
    [0, -1, -10],
)
def test_invalid_d(d):
    with pytest.raises(ValueError):
        generate_independent_bernoulli(
            n=100,
            d=d,
            p=0.5,
            seed=42,
        )


@pytest.mark.parametrize(
    "p",
    [-0.1, 1.1, -1, 2],
)
def test_invalid_p(p):
    with pytest.raises(ValueError):
        generate_independent_bernoulli(
            n=100,
            d=4,
            p=p,
            seed=42,
        )