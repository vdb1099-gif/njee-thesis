"""Tiny real CPU integration and provenance/output safety; no accuracy thresholds."""

from dataclasses import fields
import json
import random

import numpy as np
import pytest
import torch
import yaml

from experiments import neural_baseline as baseline
from src.estimators.classical import MillerMadowEstimator, PlugInEstimator
from src.simulation.factory import generate_dataset, get_ground_truth
from src.utils.split import load_split, make_iid_split


@pytest.fixture(autouse=True)
def preserve_state():
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.set_num_threads(threads)


def configuration(tmp_path, *, generator="binary_chain", d=3):
    config = yaml.safe_load((baseline.PROJECT_ROOT / "configs/neural_baseline.yaml").read_text())
    config["data"].update(generator=generator, d=d)
    config["data"]["params"] = {"epsilon" if generator == "binary_chain" else "p": .3}
    config["neural"]["component_order"] = list(reversed(range(d)))
    config["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, path


def read(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("generator", ["binary_chain", "independent_bernoulli"])
def test_real_pipeline_partitions_schema_and_provenance(tmp_path, monkeypatch, generator):
    config, path = configuration(tmp_path, generator=generator)
    calls, reports = [], []
    original_fit = baseline.NeuralJointEntropyEstimator.fit
    original_estimate = baseline.NeuralJointEntropyEstimator.estimate
    def fit(self, train, validation):
        assert train.role == "train" and validation.role == "validation"
        assert train.dataset_id == validation.dataset_id
        assert train.split_id == validation.split_id
        assert not np.intersect1d(train.row_indices, validation.row_indices).size
        calls.append((train, validation))
        result = original_fit(self, train, validation)
        assert self._encoder._train_rows == frozenset(train.row_indices)
        assert all(self._encoder.cardinalities[v] == 2 for v in train.variable_ids)
        return result
    def estimate(self, test):
        assert test.role == "test"
        assert not np.intersect1d(test.row_indices, calls[-1][0].row_indices).size
        assert not np.intersect1d(test.row_indices, calls[-1][1].row_indices).size
        result = original_estimate(self, test)
        reports.append(result)
        return result
    monkeypatch.setattr(baseline.NeuralJointEntropyEstimator, "fit", fit)
    monkeypatch.setattr(baseline.NeuralJointEntropyEstimator, "estimate", estimate)
    directory = baseline.run_baseline(path, invocation=["test", "--config", str(path)])
    manifest = read(directory, "manifest.json")
    records = read(directory, "results.json")
    diagnostics = read(directory, "njee_diagnostics.json")
    split = load_split(directory / "split.npz")
    expected = make_iid_split(config["data"]["n"], seed=config["neural"]["split_seed"],
                              fractions=tuple(config["neural"]["split_fractions"]))
    for role in ("train", "validation", "test"):
        np.testing.assert_array_equal(getattr(split, role), getattr(expected, role))
    with np.load(directory / "dataset.npz", allow_pickle=False) as saved:
        samples = saved["values"]
    np.testing.assert_array_equal(samples, generate_dataset(config))
    np.testing.assert_array_equal(calls[0][0].values, samples[split.train])
    np.testing.assert_array_equal(calls[0][1].values, samples[split.validation])
    assert manifest["status"] == "complete"
    assert set(manifest["output_files"]) == {p.name for p in directory.iterdir()}
    assert manifest["run_id"] == config["run_id"]
    assert manifest["dataset_shape"] == [23, 3]
    assert manifest["dataset_id"] == calls[0][0].dataset_id
    assert manifest["split_id"] == calls[0][0].split_id
    assert manifest["ground_truth_nats"] == float(get_ground_truth(config))
    assert manifest["seeds"]["data_generation"] == config["seed"]
    assert manifest["seeds"]["split"] == config["neural"]["split_seed"]
    assert manifest["seeds"]["training"] == config["neural"]["training_seed"]
    assert manifest["component_order"] == [2, 1, 0]
    assert manifest["resolved_configuration"] == json.loads(json.dumps(config))
    assert len(manifest["git"]["commit"]) == 40 and type(manifest["git"]["dirty"]) is bool
    for key in ("python_version", "numpy_version", "torch_version", "device", "machine", "processor"):
        assert key in manifest["environment"]
    assert manifest["environment"]["device"] == "cpu"
    assert manifest["timestamp_utc"] and manifest["completed_at_utc"]
    assert manifest["invocation"]["argv"] == ["test", "--config", str(path)]
    assert manifest["protocol_note"]
    assert {r["variant"] for r in records} == set(baseline.VARIANTS)
    assert len(records) == 5
    for record in records:
        assert record["signed_error_nats"] == record["entropy_nats"] - record["ground_truth_nats"]
        assert record["runtime_seconds"] >= 0 and record["entropy_unit"] == "nats"
        assert record["sample_counts"] == manifest["sample_counts"]
        if record["variant"] == "njee":
            assert record["entropy_nats"] == reports[0].entropy_nats
            assert record["estimation_samples"]["conditional_test"] == len(split.test)
        else:
            name, partition = record["variant"].rsplit("_", 1)
            classical = PlugInEstimator() if name == "plug_in" else MillerMadowEstimator()
            values = samples if partition == "full" else samples[split.train]
            assert record["entropy_nats"] == classical.estimate(values)
            assert record["estimation_samples"] == {partition: len(values)}
    # Preserve every field, including explicit None fields, warnings and environment.
    assert diagnostics == baseline._json_value(reports[0])
    for term, original in zip(diagnostics["terms"], reports[0].terms):
        assert set(term) == {f.name for f in fields(original)}
    assert diagnostics["entropy_nats"] == sum(t["contribution_nats"] for t in diagnostics["terms"])
    assert len(manifest["seeds"]["effective_terms"]) == 2


def test_repeatability_and_overrides_control_actual_execution(tmp_path):
    config, path = configuration(tmp_path)
    first = baseline.run_baseline(path, output_dir=tmp_path / "chosen", run_id="first")
    second = baseline.run_baseline(path, output_dir=tmp_path / "chosen", run_id="second")
    a, b = read(first, "results.json"), read(second, "results.json")
    for result in a + b:
        del result["runtime_seconds"]
    assert a == b
    da, db = read(first, "njee_diagnostics.json"), read(second, "njee_diagnostics.json")
    for left, right in zip(da["terms"], db["terms"]):
        for key in ("training_runtime_seconds", "evaluation_runtime_seconds"):
            del left[key]
            del right[key]
        assert left == right
    assert read(first, "manifest.json")["resolved_configuration"]["output_dir"] == str(tmp_path / "chosen")
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    with pytest.raises(FileExistsError):
        baseline.run_baseline(path, output_dir=tmp_path / "chosen", run_id="first")
    assert before == {p.name: p.read_bytes() for p in first.iterdir()}
    config["data"] = {"generator": "independent_bernoulli", "n": 10, "d": 1, "params": {"p": 0.}}
    config["neural"]["component_order"] = [0]
    config["neural"]["split_fractions"] = [.4, .3, .3]
    path.write_text(yaml.safe_dump(config))
    changed = baseline.run_baseline(path, run_id="changed")
    results = read(changed, "results.json")
    assert all(r["ground_truth_nats"] == 0. for r in results)
    assert results[0]["entropy_nats"] == 0.
    assert results[0]["sample_counts"] == {"full": 10, "train": 4, "validation": 3, "test": 3}
    assert results[0]["estimation_samples"]["conditional_test"] == 0


@pytest.mark.parametrize("failure", ["generation", "fit", "write", "completion", "environment"])
def test_failed_run_is_never_complete_and_not_reused(tmp_path, monkeypatch, failure):
    config, path = configuration(tmp_path)
    def broken(*args, **kwargs):
        raise RuntimeError("Injected failure")
    if failure == "generation":
        monkeypatch.setattr(baseline, "generate_dataset", broken)
    elif failure == "fit":
        monkeypatch.setattr(baseline.NeuralJointEntropyEstimator, "fit", broken)
    elif failure == "environment":
        monkeypatch.setattr(baseline, "capture_environment", broken)
    else:
        original = baseline._write_json
        def write(path, value):
            if ((failure == "write" and path.name == "njee_diagnostics.json")
                    or (failure == "completion" and path.name == "manifest.json"
                        and value["status"] == "complete")):
                broken()
            original(path, value)
        monkeypatch.setattr(baseline, "_write_json", write)
    with pytest.raises(RuntimeError, match="Injected"):
        baseline.run_baseline(path)
    directory = tmp_path / "output" / config["run_id"]
    manifest = read(directory, "manifest.json")
    assert manifest["status"] == "failed" and manifest["failure"]["type"] == "RuntimeError"
    assert "completed_at_utc" not in manifest
    if failure == "environment":
        assert manifest["environment"] is None and manifest["limitations"]
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        baseline.run_baseline(path)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_interrupted_run_remains_partial(tmp_path, monkeypatch):
    config, path = configuration(tmp_path)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(baseline, "generate_dataset", interrupted)
    with pytest.raises(KeyboardInterrupt):
        baseline.run_baseline(path)
    directory = tmp_path / "output" / config["run_id"]
    assert read(directory, "manifest.json")["status"] == "running"
    assert not (directory / "results.json").exists()
    with pytest.raises(FileExistsError):
        baseline.run_baseline(path)


@pytest.mark.parametrize("change", ["extra", "generator", "order", "policy", "seed", "split", "run_id", "output", "hyperparameter"])
def test_invalid_config_before_output_creation(tmp_path, change):
    config, path = configuration(tmp_path)
    if change == "extra":
        config["ignored"] = True
    elif change == "generator":
        config["data"]["generator"] = "invalid"
    elif change == "order":
        config["neural"]["component_order"] = [0, 1]
    elif change == "policy":
        config["neural"]["alphabet_policy"] = "train_observed"
    elif change == "seed":
        config["seed"] = -1
    elif change == "split":
        config["neural"]["split_fractions"] = [.5, .5, .5]
    elif change == "run_id":
        config["run_id"] = "../escape"
    elif change == "output":
        config["output_dir"] = ""
    else:
        config["neural"]["learning_rate"] = 0.
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        baseline.run_baseline(path)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("content", ["", "[", "- not\n- a mapping"])
def test_malformed_yaml(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        baseline.run_baseline(path)


def test_git_unavailable_is_recorded(tmp_path, monkeypatch):
    _, path = configuration(tmp_path)
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git unavailable")
    monkeypatch.setattr(baseline.subprocess, "run", unavailable)
    directory = baseline.run_baseline(path)
    git = read(directory, "manifest.json")["git"]
    assert git["commit"] is git["dirty"] is None
    assert git["limitation"]


def test_cli_records_invocation_and_overrides(tmp_path):
    _, path = configuration(tmp_path)
    argv = ["--config", str(path), "--output-dir", str(tmp_path / "cli"), "--run-id", "cli-run"]
    baseline.main(argv)
    manifest = read(tmp_path / "cli" / "cli-run", "manifest.json")
    assert manifest["invocation"]["argv"][1:] == argv
    assert manifest["status"] == "complete"
