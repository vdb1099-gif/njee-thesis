from itertools import product
import math

import numpy as np
import pytest

from src.estimators import EntropyEstimator, MillerMadowEstimator, PlugInEstimator
from src.simulation.ground_truth import binary_chain_entropy, exact_entropy


@pytest.fixture(params=[PlugInEstimator, MillerMadowEstimator])
def estimator(request):
    return request.param()


def test_common_contract_and_constant(estimator):
    assert isinstance(estimator, EntropyEstimator)
    assert estimator.estimate(np.full((12, 3), -4)) == 0.0
    assert estimator.estimate(np.array([42])) == 0.0


@pytest.mark.parametrize("k", [1, 2, 5, 16])
def test_balanced_uniform(k):
    samples = np.repeat(np.arange(k), 8)
    assert PlugInEstimator().estimate(samples) == pytest.approx(np.log(k))
    assert MillerMadowEstimator().estimate(samples) == pytest.approx(
        np.log(k) + (k - 1) / (2 * len(samples))
    )


def test_nonuniform_joint_distribution():
    states = np.array(list(product([0, 1], repeat=2)))
    counts = np.array([1, 2, 3, 4])
    samples = np.repeat(states, counts, axis=0)
    expected = -sum(p * np.log(p) for p in [0.1, 0.2, 0.3, 0.4])
    assert exact_entropy((counts / 10).reshape(2, 2)) == pytest.approx(expected)
    assert PlugInEstimator().estimate(samples) == pytest.approx(expected)
    assert MillerMadowEstimator().estimate(samples) == pytest.approx(expected + 3 / 20)


def test_observed_joint_states_not_marginals():
    # Only two of four possible states occur; duplicate columns add no entropy.
    samples = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    assert PlugInEstimator().estimate(samples) == pytest.approx(np.log(2))
    assert MillerMadowEstimator().estimate(samples) == pytest.approx(np.log(2) + 1 / 8)


def test_order_and_label_invariance(estimator):
    samples = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]])
    expected = estimator.estimate(samples)
    assert estimator.estimate(samples[::-1, [2, 0, 1]]) == pytest.approx(expected)
    assert estimator.estimate(samples * 19 - 7) == pytest.approx(expected)
    assert estimator.estimate(samples.astype(bool)) == pytest.approx(expected)
    assert estimator.estimate(samples.astype(float)) == pytest.approx(expected)


def test_one_dimensional_and_wide_inputs(estimator):
    samples = np.array([0, 0, 1, 1])
    expected = estimator.estimate(samples)
    assert estimator.estimate(samples[:, None]) == pytest.approx(expected)
    assert estimator.estimate(np.tile(samples[:, None], (1, 128))) == pytest.approx(expected)


@pytest.mark.parametrize("samples", [
    [], np.empty((0, 2)), np.empty((2, 0)), 1, np.zeros((2, 2, 2)),
    [0, np.nan], [0, np.inf], [0, 0.5], ["a", "b"], [None, 1], [1j, 0],
])
def test_invalid_samples(estimator, samples):
    with pytest.raises(ValueError):
        estimator.estimate(np.asarray(samples))


@pytest.mark.parametrize("epsilon", [0.0, 0.25, 0.5, 1.0])
def test_enumerated_binary_chain(epsilon):
    states = np.array(list(product([0, 1], repeat=3)))
    flips = np.sum(states[:, 1:] != states[:, :-1], axis=1)
    probabilities = 0.5 * epsilon ** flips * (1 - epsilon) ** (2 - flips)
    expected = binary_chain_entropy(epsilon, d=3)
    assert exact_entropy(probabilities) == pytest.approx(expected)
    samples = np.repeat(states, np.rint(probabilities * 32).astype(int), axis=0)
    assert PlugInEstimator().estimate(samples) == pytest.approx(expected)


def test_exact_entropy_zero_and_uniform():
    assert exact_entropy([0, 1, 0]) == 0.0
    assert exact_entropy(np.full((2, 3), 1 / 6)) == pytest.approx(np.log(6))


@pytest.mark.parametrize("pmf", [
    [], 1, [0, 0], [0.2, 0.2], [-0.1, 1.1], [np.nan, 1],
    [np.inf, 0], [1j, 0], ["0.5", "0.5"], [0.5, 0.50000001],
])
def test_invalid_exact_probabilities(pmf):
    with pytest.raises(ValueError):
        exact_entropy(pmf)


def test_masked_inputs_are_rejected():
    # Even an all-False mask must not silently change the accepted input type.
    for mask in ([False, True], [False, False]):
        for function, values in (
            (PlugInEstimator().estimate, [0, 1]),
            (MillerMadowEstimator().estimate, [0, 1]),
            (exact_entropy, [0.5, 0.5]),
        ):
            with pytest.raises(ValueError, match="MaskedArray"):
                function(np.ma.array(values, mask=mask))


def test_array_contract_and_large_distinct_labels(estimator):
    for unsupported in ([2**53, 2**53 + 1, 0.0], [0, 1], (0, 1)):
        with pytest.raises(TypeError, match="NumPy ndarray"):
            estimator.estimate(unsupported)
    # Three distinct labels, K+ = n > 1: no floating-point label conversion.
    for samples in (
        np.array([2**53, 2**53 + 1, 0], dtype=np.int64),
        np.array([2**64 - 1, 2**64 - 2, 0], dtype=np.uint64),
    ):
        expected = math.log(3)
        if isinstance(estimator, MillerMadowEstimator):
            expected += 1 / 3
        assert estimator.estimate(samples) == pytest.approx(expected, abs=1e-14)


def test_inputs_are_not_mutated():
    backing = np.array([[9, 2, -1], [3, 7, 4], [9, 0, -1]])
    original = backing.copy()
    samples = backing[:, ::2]  # Noncontiguous view with duplicate rows.
    samples.flags.writeable = False
    expected = -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3)
    assert PlugInEstimator().estimate(samples) == pytest.approx(expected)
    assert MillerMadowEstimator().estimate(samples) == pytest.approx(expected + 1 / 6)
    np.testing.assert_array_equal(backing, original)
    pmf = np.array([0.0, 0.25, 0.75])
    before = pmf.copy()
    pmf.flags.writeable = False
    assert exact_entropy(pmf) == pytest.approx(-0.25 * math.log(0.25) - 0.75 * math.log(0.75))
    np.testing.assert_array_equal(pmf, before)


def test_pmf_normalization_precision_contract():
    with pytest.raises(ValueError, match="sum to 1"):
        exact_entropy(np.array([0.1, 0.2, 0.7], dtype=np.float32))
    # Exactly representable float32 probabilities remain valid.
    assert exact_entropy(np.array([0.25, 0.75], dtype=np.float32)) == pytest.approx(
        -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
    )
    assert exact_entropy(np.array([0.1, 0.2, 0.7], dtype=np.float64)) == pytest.approx(
        -sum(p * math.log(p) for p in [0.1, 0.2, 0.7])
    )
