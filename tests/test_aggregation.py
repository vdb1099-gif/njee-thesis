"""Manual error/SD calculations, replica bootstrap and paired failure accounting."""

from copy import deepcopy
import math

import numpy as np
import pytest

from src.metrics.aggregation import aggregate_records, bootstrap_seed_for


def rows(estimates=(1., 2., 4.), variant="a"):
    return [{"campaign_id": "campaign", "scenario_id": "scenario", "replica_id": f"r{i}",
             "variant": variant, "status": "complete", "entropy_nats": x, "ground_truth_nats": 2.,
             "fit_runtime_seconds": float(i + 1), "inference_runtime_seconds": .5,
             "runtime_seconds": i + 1.5} for i, x in enumerate(estimates)]


def aggregate(records, comparisons=()):
    return aggregate_records(records, planned_replica_ids=["r0", "r1", "r2"],
                             bootstrap_draws=100, bootstrap_seed=42, paired_comparisons=comparisons)


def test_manual_metrics_ddof_and_runtimes():
    records = rows()
    before = deepcopy(records)
    summary = aggregate(records)["summaries"][0]
    assert summary["n_planned"] == summary["n_attempted"] == summary["n_successful"] == 3
    assert summary["n_failed"] == summary["n_not_attempted"] == 0
    assert summary["bias_nats"] == pytest.approx(1/3)
    assert summary["mae_nats"] == 1.
    assert summary["rmse_nats"] == pytest.approx(math.sqrt(5/3))
    assert summary["sample_std_nats"] == pytest.approx(math.sqrt(7/3))
    assert summary["mean_fit_runtime_seconds"] == 2.
    assert summary["mean_inference_runtime_seconds"] == .5
    assert summary["mean_runtime_seconds"] == 2.5
    assert records == before


def test_bootstrap_manual_percentile_and_order_independence():
    records = rows()
    report = aggregate(records)
    assert report == aggregate(list(reversed(records)))
    interval = report["summaries"][0]["bootstrap"]["bias_nats"]
    rng = np.random.default_rng(interval["seed"])
    errors = np.array([-1., 0., 2.])
    draws = [float(np.mean(errors[rng.integers(0, 3, size=3)])) for _ in range(100)]
    assert interval["lower"] == np.quantile(draws, .025, method="linear")
    assert interval["upper"] == np.quantile(draws, .975, method="linear")
    assert interval["method"] == "percentile" and interval["unit"] == "replica"
    assert interval["confidence_level"] == .95 and interval["draws"] == 100
    assert bootstrap_seed_for(42, ["a"]) == bootstrap_seed_for(42, ["a"])
    assert bootstrap_seed_for(42, ["a"]) != bootstrap_seed_for(42, ["b"])


@pytest.mark.parametrize("successful", [0, 1])
def test_zero_one_successes_no_inferential_interval(successful):
    records = rows()
    for row in records[successful:]:
        row.update(status="failed", entropy_nats=None)
    summary = aggregate(records)["summaries"][0]
    assert summary["n_successful"] == successful and summary["n_failed"] == 3 - successful
    assert summary["sample_std_nats"] is None
    assert summary["bias_nats"] == (-1. if successful else None)
    assert all(ci["lower"] is None and ci["upper"] is None
               and ci["reason"] for ci in summary["bootstrap"].values())


def test_nonfinite_interruption_unattempted_and_missing_timing():
    records = rows()
    records[0]["entropy_nats"] = float("nan")
    records[1]["status"] = "interrupted"
    records[2]["status"] = "not_attempted"
    summary = aggregate(records)["summaries"][0]
    assert summary["n_attempted"] == 2 and summary["n_failed"] == 2
    assert summary["n_successful"] == 0 and summary["n_not_attempted"] == 1
    assert summary["n_interrupted"] == summary["n_invalid_estimates"] == 1
    records = rows()
    records[0]["fit_runtime_seconds"] = None
    summary = aggregate(records)["summaries"][0]
    assert summary["n_successful"] == 3 and summary["n_fit_runtime_seconds"] == 2
    assert summary["mean_fit_runtime_seconds"] == 2.5


def test_paired_replicas_resampled_jointly_with_failures():
    a = rows((1., 2., 4.), "a")
    b = rows((.5, 1.5, 3.5), "b")
    b[1].update(status="failed", entropy_nats=None)
    report = aggregate(a + list(reversed(b)), (("a", "b"),))
    pair = report["paired_comparisons"][0]
    assert pair["n_paired_successful"] == 2 and pair["n_unpaired"] == 1
    assert [d["replica_id"] for d in pair["differences"]] == ["r0", "r2"]
    assert pair["mean_estimate_difference_nats"] == .5
    assert pair["bootstrap"]["estimate_difference_nats"]["lower"] == .5
    assert pair["bootstrap"]["estimate_difference_nats"]["upper"] == .5
    assert pair["bootstrap"]["estimate_difference_nats"]["unit"] == "paired_replica_id"
    assert pair["mean_absolute_error_difference_nats"] == 0.
    # Matching by list position instead of identity would produce different differences.
    assert pair == aggregate(list(reversed(a + b)), (("a", "b"),))["paired_comparisons"][0]


@pytest.mark.parametrize("invalid", ["duplicate", "missing", "truth", "status", "unplanned"])
def test_invalid_raw_identity_or_protocol(invalid):
    records = rows()
    if invalid == "duplicate":
        records.append(deepcopy(records[0]))
    elif invalid == "missing":
        records.pop()
    elif invalid == "truth":
        other = rows(variant="b")
        other[0]["ground_truth_nats"] = 8.
        records += other
    elif invalid == "status":
        records[0]["status"] = "unknown"
    else:
        records[0]["replica_id"] = "unplanned"
    with pytest.raises(ValueError):
        aggregate(records)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_bootstrap_invalid_seed(seed):
    with pytest.raises(ValueError):
        bootstrap_seed_for(seed, "test")


def test_bootstrap_does_not_touch_global_numpy_rng():
    state = np.random.get_state()
    aggregate(rows())
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])
    assert np.random.get_state()[2:] == state[2:]


@pytest.mark.parametrize("paired_successes", [0, 1])
def test_insufficient_pairs_do_not_report_intervals(paired_successes):
    a, b = rows(variant="a"), rows(variant="b")
    for row in b[paired_successes:]:
        row.update(status="failed", entropy_nats=None)
    pair = aggregate(a + b, (("a", "b"),))["paired_comparisons"][0]
    assert pair["n_paired_successful"] == paired_successes
    assert all(ci["lower"] is None and ci["upper"] is None for ci in pair["bootstrap"].values())
