"""Campaign seed allocation, raw preservation, fail-safe accounting and tiny CPU runs."""

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from experiments import neural_campaign as campaign


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


def configuration(tmp_path):
    base = yaml.safe_load((campaign.baseline.PROJECT_ROOT / "configs/neural_baseline.yaml").read_text())
    base["data"].update(n=9, d=2)
    base["neural"].update(component_order=[1, 0], max_epochs=1)
    (tmp_path / "baseline.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    config = yaml.safe_load((campaign.baseline.PROJECT_ROOT / "configs/neural_campaign.yaml").read_text())
    config.update(baseline_config="baseline.yaml", output_dir=str(tmp_path / "out"))
    config["aggregation"]["bootstrap_draws"] = 20
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, path


def read(directory, name):
    return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))


def test_tiny_cpu_campaign_same_dataset_across_estimators(tmp_path, monkeypatch):
    config, path = configuration(tmp_path)
    generated, classical = [], []
    original = campaign.baseline.generate_dataset
    def generate(config):
        values = original(config)
        generated.append(values)
        return values
    monkeypatch.setattr(campaign.baseline, "generate_dataset", generate)
    for cls in (campaign.baseline.PlugInEstimator, campaign.baseline.MillerMadowEstimator):
        original_estimate = cls.estimate
        def estimate(self, samples, original_estimate=original_estimate):
            if samples.ndim == 2:
                classical.append(samples)
            return original_estimate(self, samples)
        monkeypatch.setattr(cls, "estimate", estimate)
    directory = campaign.run_campaign(path, invocation=["test", str(path)])
    manifest = read(directory, "campaign_manifest")
    raw = read(directory, "raw_results")
    aggregation = read(directory, "aggregation")
    assert len(generated) == 2 and len(classical) == 8
    for i in range(2):
        assert classical[i * 4] is classical[i * 4 + 1] is generated[i]
        assert classical[i * 4 + 2] is classical[i * 4 + 3]
    assert manifest["status"] == "complete" and manifest["purpose"] == "development"
    assert manifest["configuration"]["baseline_config"] == str(tmp_path / "baseline.yaml")
    assert manifest["stream_assumption"] and manifest["timing_policy"] and manifest["resume_policy"]
    assert len(raw) == 10 and len(aggregation["summaries"]) == 5
    assert len({r["scenario_id"] for r in raw}) == 1
    assert len({r["replica_id"] for r in raw}) == 2
    assert len({r["data_seed"] for r in raw}) == 2
    for row in raw:
        assert row["status"] == row["artifact_status"] == "complete"
        assert row["signed_error_nats"] == row["entropy_nats"] - row["ground_truth_nats"]
        assert row["absolute_error_nats"] == abs(row["signed_error_nats"])
        assert row["squared_error_nats"] == row["signed_error_nats"]**2
        assert row["fit_runtime_seconds"] >= 0 and row["inference_runtime_seconds"] >= 0
        replica = directory / row["artifacts"]
        original_results = read(replica, "results")
        original_record = next(r for r in original_results if r["variant"] == row["variant"])
        assert row["entropy_nats"] == original_record["entropy_nats"]
        assert row["sample_counts"] == original_record["sample_counts"]
        provenance = read(replica, "manifest")
        assert provenance["seeds"]["data_generation"] == row["data_seed"]
        assert provenance["seeds"]["split"] == row["split_seed"]
        assert provenance["seeds"]["training"] == row["training_seed"]
        assert provenance["seeds"]["effective_terms"] == row["effective_term_seeds"]
        assert (replica / "njee_diagnostics.json").exists()
    assert all(s["n_planned"] == s["n_attempted"] == s["n_successful"] == 2 for s in aggregation["summaries"])
    assert all(s["n_failed"] == 0 for s in aggregation["summaries"])
    assert all(p["n_paired_successful"] == 2 for p in aggregation["paired_comparisons"])
    # Inspection and rerun refusal preserve every artifact byte.
    before = {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert campaign.inspect_campaign(directory)["aggregation"] == aggregation
    with pytest.raises(FileExistsError):
        campaign.run_campaign(path)
    assert before == {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_repeatable_campaign_numerics(tmp_path):
    config, path = configuration(tmp_path)
    first = campaign.run_campaign(path, output_dir=tmp_path / "a")
    second = campaign.run_campaign(path, output_dir=tmp_path / "b")
    keys = ("replica_id", "variant", "data_seed", "split_seed", "training_seed", "entropy_nats", "signed_error_nats")
    assert [{k: r[k] for k in keys} for r in read(first, "raw_results")] == [
        {k: r[k] for k in keys} for r in read(second, "raw_results")]
    for a, b in zip(read(first, "aggregation")["summaries"], read(second, "aggregation")["summaries"]):
        assert a["bootstrap"] == b["bootstrap"] and a["rmse_nats"] == b["rmse_nats"]


def test_seed_derivation_stable_domains_and_global_rng():
    state = np.random.get_state()
    a = campaign.derive_replica_seeds(123, 0)
    b = campaign.derive_replica_seeds(123, 1)
    assert a == campaign.derive_replica_seeds(123, 0) and a != b
    assert len(set(a.values()) | set(b.values())) == 6
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    # Stable fixed allocation, including the configured uint32 upper boundary.
    expected = int(np.random.SeedSequence(123, spawn_key=(1, 0)).generate_state(1, dtype=np.uint32)[0])
    assert a["data_seed"] == expected
    assert all(0 <= v <= 2**32 - 1 for v in campaign.derive_replica_seeds(2**32 - 1, 2**32 - 1).values())


def test_explicit_seed_identities(tmp_path):
    config, path = configuration(tmp_path)
    del config["master_seed"]
    config["replica_seeds"] = [{"replica_id": f"custom-{i}", "data_seed": i,
                                "split_seed": 2**32 - 1, "training_seed": 0} for i in range(2)]
    path.write_text(yaml.safe_dump(config))
    loaded, _, replicas = campaign.load_campaign_config(path)
    assert replicas == config["replica_seeds"]
    config["replica_seeds"][1]["replica_id"] = "custom-0"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="distinct"):
        campaign.load_campaign_config(path)


@pytest.mark.parametrize("interrupt", [False, True])
def test_failure_and_interruption_accounting(tmp_path, monkeypatch, interrupt):
    config, path = configuration(tmp_path)
    config["n_replicas"] = 3
    path.write_text(yaml.safe_dump(config))
    calls = []
    original = campaign.baseline.run_baseline
    def run(path, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            single, _ = campaign.baseline.load_baseline_config(path)
            root = Path(single["output_dir"]) / single["run_id"]
            root.mkdir(parents=True)
            campaign.baseline._write_json(root / "manifest.json", {"status": "running" if interrupt else "failed"})
            if interrupt:
                raise KeyboardInterrupt()
            raise RuntimeError("Injected failed replica")
        return original(path, **kwargs)
    monkeypatch.setattr(campaign.baseline, "run_baseline", run)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            campaign.run_campaign(path)
    else:
        campaign.run_campaign(path)
    directory = tmp_path / "out" / config["campaign_id"]
    assert len(calls) == (2 if interrupt else 3)
    manifest = read(directory, "campaign_manifest")
    assert manifest["status"] == ("interrupted" if interrupt else "complete_with_failures")
    raw = read(directory, "raw_results")
    assert len(raw) == 15
    assert all(r["status"] == ("interrupted" if interrupt else "failed") for r in raw[5:10])
    assert all(r["artifact_status"] == ("running" if interrupt else "failed") for r in raw[5:10])
    if interrupt:
        assert all(r["status"] == "not_attempted" for r in raw[10:])
    for summary in read(directory, "aggregation")["summaries"]:
        assert summary["n_planned"] == 3 and summary["n_failed"] == 1
        assert summary["n_attempted"] == (2 if interrupt else 3)
        assert summary["n_successful"] == (1 if interrupt else 2)
        assert summary["n_not_attempted"] == (1 if interrupt else 0)
    with pytest.raises(FileExistsError):
        campaign.run_campaign(path)


@pytest.mark.parametrize("kind", ["extra", "both_seeds", "count", "master", "draws", "purpose", "pairs", "path", "id", "duplicate_data"])
def test_malformed_configuration_fails_before_outputs(tmp_path, kind):
    config, path = configuration(tmp_path)
    if kind == "extra":
        config["ignored"] = True
    elif kind == "both_seeds":
        config["replica_seeds"] = []
    elif kind == "count":
        config["n_replicas"] = True
    elif kind == "master":
        config["master_seed"] = -1
    elif kind == "draws":
        config["aggregation"]["bootstrap_draws"] = 0
    elif kind == "purpose":
        config["purpose"] = "unknown"
    elif kind == "pairs":
        config["aggregation"]["paired_comparisons"] = [["njee", "unknown"]]
    elif kind == "path":
        config["baseline_config"] = ""
    elif kind == "id":
        config["campaign_id"] = "../escape"
    else:
        del config["master_seed"]
        config["replica_seeds"] = [{"replica_id": f"r{i}", "data_seed": 0,
                                    "split_seed": 1, "training_seed": 2} for i in range(2)]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        campaign.run_campaign(path)
    assert not (tmp_path / "out").exists()


def test_cli_run_and_read_only_inspection(tmp_path, capsys):
    config, path = configuration(tmp_path)
    campaign.main(["--config", str(path)])
    assert "Campaign artifacts" in capsys.readouterr().out
    directory = tmp_path / "out" / config["campaign_id"]
    campaign.main(["--inspect", str(directory)])
    result = json.loads(capsys.readouterr().out)
    assert result["campaign_manifest"]["status"] == "complete"


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_replica_seed_boundaries(seed):
    with pytest.raises(ValueError):
        campaign.derive_replica_seeds(seed, 0)
    with pytest.raises(ValueError):
        campaign.derive_replica_seeds(0, seed)


def test_single_variable_protocol(tmp_path):
    config, path = configuration(tmp_path)
    basepath = tmp_path / "baseline.yaml"
    base = yaml.safe_load(basepath.read_text())
    base["data"]["d"] = 1
    base["neural"]["component_order"] = [0]
    basepath.write_text(yaml.safe_dump(base))
    directory = campaign.run_campaign(path)
    assert all(r["sample_protocol"] == "train_plugin" for r in read(directory, "raw_results") if r["variant"] == "njee")
    assert all(r["effective_term_seeds"] == [] for r in read(directory, "raw_results"))


def test_seed_collision_rejected_before_creation(tmp_path, monkeypatch):
    _, path = configuration(tmp_path)
    monkeypatch.setattr(campaign, "derive_replica_seeds", lambda *args: {"data_seed": 1, "split_seed": 2, "training_seed": 3})
    with pytest.raises(ValueError, match="collision"):
        campaign.run_campaign(path)
    assert not (tmp_path / "out").exists()
