"""Real generation, estimation, serialization and repeatability checks."""

from collections import Counter
import json
import math
import sys

import numpy as np
import pytest

from experiments import smoke_test


@pytest.mark.parametrize("generator,parameter", [
    ("binary_chain", "epsilon"),
    ("independent_bernoulli", "p"),
])
def test_smoke_pipeline(tmp_path, monkeypatch, generator, parameter):
    config = tmp_path / "scenario.yaml"
    config.write_text(
        f"seed: 42\ndata:\n  generator: {generator}\n  n: 101\n"
        f"  d: 3\n  params:\n    {parameter}: 0.3\n", encoding="utf-8"
    )
    calls = []

    def trace(name, original):
        def estimate(self, samples):
            before = samples.copy()
            value = original(self, samples)
            np.testing.assert_array_equal(samples, before)
            calls.append((name, samples, value))
            return value
        return estimate

    for name, cls in (
        ("plug_in", smoke_test.PlugInEstimator),
        ("miller_madow", smoke_test.MillerMadowEstimator),
    ):
        monkeypatch.setattr(cls, "estimate", trace(name, cls.estimate))

    # Only the clock is synthetic; both estimators and all I/O execute for real.
    ticks = iter([10.0, 10.125, 20.0, 20.25, 30.0, 30.5, 40.0, 41.0])
    monkeypatch.setattr(smoke_test, "perf_counter", lambda: next(ticks))
    results = []
    for run, runtimes in enumerate(((0.125, 0.25), (0.5, 1.0))):
        output = tmp_path / f"run_{run}"
        monkeypatch.setattr(sys, "argv", [
            "smoke_test", "--config", str(config), "--output-dir", str(output),
        ])
        smoke_test.main()
        result = json.loads((output / "smoke_test_result.json").read_text(encoding="utf-8"))
        recent = calls[run * 2:run * 2 + 2]
        assert len(recent) == 2
        assert recent[0][1] is recent[1][1]
        samples = recent[0][1]
        assert samples.shape == (101, 3)

        # Independent reference: Python row tuples, Counter and math.log.
        counts = Counter(map(tuple, samples.tolist()))
        expected_plugin = -math.fsum((c / 101) * math.log(c / 101) for c in counts.values())
        binary_h = -0.3 * math.log(0.3) - 0.7 * math.log(0.7)
        truth = math.log(2) + 2 * binary_h if generator == "binary_chain" else 3 * binary_h
        assert result["true_entropy_nats"] == pytest.approx(truth, abs=1e-14)
        assert result["seed"] == 42
        assert result["generator"] == generator
        assert result["params"] == {parameter: 0.3}
        assert result["n"] == 101 and result["d"] == 3
        assert result["dataset_shape"] == [101, 3]
        assert result["component_order"] == [0, 1, 2]
        assert result["entropy_unit"] == "nats" and result["aggregation"] == "joint"
        assert set(result["estimators"]) == {"plug_in", "miller_madow"}
        for index, (name, _, value) in enumerate(recent):
            saved = result["estimators"][name]
            expected = expected_plugin + (0 if name == "plug_in" else (len(counts) - 1) / 202)
            assert type(value) is float
            assert value == pytest.approx(expected, abs=1e-14)
            assert saved["entropy_nats"] == value
            assert saved["error_nats"] == value - result["true_entropy_nats"]
            assert saved["runtime_seconds"] == runtimes[index]
            del saved["runtime_seconds"]
        results.append(result)

    np.testing.assert_array_equal(calls[0][1], calls[2][1])
    assert results[0] == results[1]
