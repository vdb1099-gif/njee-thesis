"""Sequential one-scenario campaign wrapping the unchanged single-run baseline.

No automatic resume: existing campaign directories are refused. --inspect is
read-only. Failed replicas are recorded and execution continues; interruption
records pending replicas as not_attempted and re-raises KeyboardInterrupt.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from experiments import neural_baseline as baseline
from src.metrics.aggregation import aggregate_records
from src.simulation.factory import get_ground_truth
from src.utils.reproducibility import validate_seed
from src.utils.split import make_iid_split


SEED_POLICY = "njee-campaign-seedsequence-v1"


def derive_replica_seeds(master_seed, replica_index):
    """SeedSequence children keyed by (role, index), independent of call order.

    Child generate_state(1, uint32) supplies the existing configured seed domain.
    Each generator then constructs its own local default_rng/PCG64 stream. We
    assume SeedSequence-mixed streams are independent for these IID simulations;
    distinct 32-bit numbers alone are not proof. Campaigns reject dataset-seed
    collisions rather than silently regenerating data or altering seed allocation.
    """
    validate_seed(master_seed)
    validate_seed(replica_index)
    return {name: int(np.random.SeedSequence(master_seed, spawn_key=(role, replica_index))
                      .generate_state(1, dtype=np.uint32)[0])
            for role, name in enumerate(("data_seed", "split_seed", "training_seed"), start=1)}


def load_campaign_config(path):
    path = Path(path).resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed campaign YAML: {path}") from exc
    common = {"campaign_id", "purpose", "n_replicas", "baseline_config", "output_dir", "aggregation"}
    if not isinstance(config, dict) or set(config) not in (common | {"master_seed"}, common | {"replica_seeds"}):
        raise ValueError("Campaign requires explicit common fields and exactly one seed policy.")
    baseline._run_id(config["campaign_id"])
    if config["purpose"] not in ("development", "final"):
        raise ValueError("purpose must be development or final; tiny campaigns are not final evidence.")
    n = config["n_replicas"]
    if type(n) is not int or not 0 < n <= 2**32:
        raise ValueError("n_replicas must be a positive Python int within the index seed domain.")
    for key in ("baseline_config", "output_dir"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty path string.")
    reference = Path(config["baseline_config"])
    if not reference.is_absolute():
        reference = path.parent / reference
    base, _ = baseline.load_baseline_config(reference)
    config["baseline_config"] = str(reference.resolve())
    aggregation = config["aggregation"]
    if not isinstance(aggregation, dict) or set(aggregation) != {"bootstrap_seed", "bootstrap_draws", "paired_comparisons"}:
        raise ValueError("Unsupported aggregation configuration.")
    validate_seed(aggregation["bootstrap_seed"])
    if type(aggregation["bootstrap_draws"]) is not int or aggregation["bootstrap_draws"] < 1:
        raise ValueError("bootstrap_draws must be a positive Python int.")
    comparisons = aggregation["paired_comparisons"]
    if (not isinstance(comparisons, list) or any(
            not isinstance(p, list) or len(p) != 2 or any(type(v) is not str or v not in baseline.VARIANTS for v in p)
            or p[0] == p[1] for p in comparisons)
            or len({tuple(p) for p in comparisons}) != len(comparisons)):
        raise ValueError("Paired comparisons must be unique pairs of distinct supported variants.")
    if "master_seed" in config:
        validate_seed(config["master_seed"])
        replicas = [{"replica_id": f"replica-{i:06d}", **derive_replica_seeds(config["master_seed"], i)}
                    for i in range(n)]
    else:
        replicas = config["replica_seeds"]
        if not isinstance(replicas, list) or len(replicas) != n:
            raise ValueError("Explicit replica seeds must match n_replicas.")
        for replica in replicas:
            if not isinstance(replica, dict) or set(replica) != {"replica_id", "data_seed", "split_seed", "training_seed"}:
                raise ValueError("Malformed explicit replica seed record.")
            baseline._run_id(replica["replica_id"])
            for name in ("data_seed", "split_seed", "training_seed"):
                validate_seed(replica[name])
    if len({r["replica_id"] for r in replicas}) != n or len({r["data_seed"] for r in replicas}) != n:
        raise ValueError("Replica identities and dataset seeds must be distinct; seed collision detected.")
    return config, base, deepcopy(replicas)


def _raw_row(config, scenario, replica, variant, sizes, truth, conditional_terms):
    return {"campaign_id": config["campaign_id"], "purpose": config["purpose"],
            "scenario_id": scenario, **replica, "variant": variant,
            "estimator": "njee" if variant == "njee" else variant.rsplit("_", 1)[0],
            "sample_protocol": ("train_plugin_plus_test_ce" if conditional_terms else "train_plugin")
            if variant == "njee" else variant.rsplit("_", 1)[1],
            "entropy_nats": None, "ground_truth_nats": truth, "signed_error_nats": None,
            "absolute_error_nats": None, "squared_error_nats": None, "sample_counts": sizes,
            "estimation_samples": None, "runtime_seconds": None, "fit_runtime_seconds": None,
            "inference_runtime_seconds": None, "status": "not_attempted", "failure": None,
            "artifacts": f"replicas/{replica['replica_id']}", "artifact_status": None,
            "entropy_unit": "nats", "effective_term_seeds": []}


def run_campaign(config_path, *, output_dir=None, invocation=None):
    config, base, replicas = load_campaign_config(config_path)
    if output_dir is not None:
        if not str(output_dir).strip():
            raise ValueError("output_dir must be nonempty.")
        config["output_dir"] = str(output_dir)
    root = Path(config["output_dir"]).resolve()
    config["output_dir"] = str(root)
    directory = root / config["campaign_id"]
    scientific = {"data": base["data"], "neural": {k: v for k, v in base["neural"].items()
                                                       if k not in ("split_seed", "training_seed")}}
    scenario = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    truth = float(get_ground_truth(base))
    allocation = make_iid_split(base["data"]["n"], seed=replicas[0]["split_seed"],
                                fractions=tuple(base["neural"]["split_fractions"]))
    sizes = {"full": allocation.n_samples, **{role: len(getattr(allocation, role))
                                             for role in ("train", "validation", "test")}}
    git = baseline._git_provenance()
    root.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    (directory / "configs").mkdir()
    manifest = {"schema_version": "njee-campaign-v1", "campaign_id": config["campaign_id"],
                "purpose": config["purpose"], "status": "running",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "configuration": config,
                "baseline_configuration": base, "scenario_id": scenario, "replicas": replicas,
                "seed_policy": SEED_POLICY if "master_seed" in config else "explicit-uint32-v1",
                "stream_assumption": "Independent local PCG64 streams initialized from SeedSequence-mixed "
                                     "or caller-supplied seeds; differing seed numbers do not prove independence.",
                "timing_policy": "NJEE fit/inference are sums of original term training/evaluation runtimes, "
                                 "including train marginal in fit; they exclude orchestration and some preprocessing. "
                                 "Original total runtime is retained. Classical fit=0, inference=original runtime.",
                "git": git, "invocation": {"config_path": str(Path(config_path).resolve()), "argv": invocation},
                "resume_policy": "No automatic resume; refuse existing directories; inspection is read-only."}
    rows = [_raw_row(config, scenario, replica, variant, sizes, truth, len(base["neural"]["component_order"]) - 1)
            for replica in replicas for variant in baseline.VARIANTS]
    baseline._write_json(directory / "campaign_manifest.json", manifest)
    baseline._write_json(directory / "raw_results.json", rows)
    interrupted = None
    for index, replica in enumerate(replicas):
        config_for_run = deepcopy(base)
        config_for_run.update(seed=replica["data_seed"], run_id=replica["replica_id"],
                              output_dir=str(directory / "replicas"))
        config_for_run["neural"].update(split_seed=replica["split_seed"], training_seed=replica["training_seed"])
        run_config = directory / "configs" / f"{replica['replica_id']}.yaml"
        run_config.write_text(yaml.safe_dump(baseline._json_value(config_for_run)), encoding="utf-8")
        selected = rows[index * len(baseline.VARIANTS):(index + 1) * len(baseline.VARIANTS)]
        for row in selected:
            row["status"] = "running"
        baseline._write_json(directory / "raw_results.json", rows)
        try:
            run = baseline.run_baseline(run_config, invocation=["campaign", str(Path(config_path).resolve()),
                                                               "replica_id", replica["replica_id"]])
            original = json.loads((run / "results.json").read_text(encoding="utf-8"))
            diagnostics = json.loads((run / "njee_diagnostics.json").read_text(encoding="utf-8"))
            provenance = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if provenance["status"] != "complete" or {r["variant"] for r in original} != set(baseline.VARIANTS):
                raise ValueError("Replica output is incomplete or has unexpected variants.")
            if len(original) != len(baseline.VARIANTS):
                raise ValueError("Duplicate replica estimator variant.")
            # Validate all records before publishing any successful row.
            for result in original:
                if (type(result["entropy_nats"]) not in (int, float) or not np.isfinite(result["entropy_nats"])
                        or result["ground_truth_nats"] != truth
                        or result["signed_error_nats"] != result["entropy_nats"] - truth):
                    raise ValueError("Nonfinite estimate or inconsistent replica ground truth/error.")
            by_variant = {r["variant"]: r for r in original}
            for row in selected:
                result = by_variant[row["variant"]]
                row.update(result)
                row.update(status="complete", artifact_status="complete",
                           absolute_error_nats=abs(result["signed_error_nats"]),
                           squared_error_nats=result["signed_error_nats"]**2,
                           effective_term_seeds=provenance["seeds"]["effective_terms"])
                row["fit_runtime_seconds"] = (sum(t["training_runtime_seconds"] for t in diagnostics["terms"])
                                               if row["variant"] == "njee" else 0.)
                row["inference_runtime_seconds"] = (sum(t["evaluation_runtime_seconds"] for t in diagnostics["terms"])
                                                     if row["variant"] == "njee" else result["runtime_seconds"])
        except (Exception, KeyboardInterrupt) as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            for row in selected:
                row.update(status=status, failure={"type": type(exc).__name__, "message": str(exc)})
            run_manifest = directory / "replicas" / replica["replica_id"] / "manifest.json"
            if run_manifest.exists():
                try:
                    artifact_status = json.loads(run_manifest.read_text(encoding="utf-8")).get("status")
                except (OSError, ValueError):
                    artifact_status = "unreadable"
                for row in selected:
                    row["artifact_status"] = artifact_status
            if isinstance(exc, KeyboardInterrupt):
                interrupted = exc
        baseline._write_json(directory / "raw_results.json", rows)
        if interrupted is not None:
            break
    aggregation = aggregate_records(rows, planned_replica_ids=[r["replica_id"] for r in replicas],
                                    bootstrap_draws=config["aggregation"]["bootstrap_draws"],
                                    bootstrap_seed=config["aggregation"]["bootstrap_seed"],
                                    paired_comparisons=config["aggregation"]["paired_comparisons"])
    baseline._write_json(directory / "aggregation.json", aggregation)
    manifest["status"] = ("interrupted" if interrupted is not None else "complete" if
                           all(r["status"] == "complete" for r in rows) else "complete_with_failures")
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    baseline._write_json(directory / "campaign_manifest.json", manifest)
    if interrupted is not None:
        raise interrupted
    return directory


def inspect_campaign(directory):
    """Read artifacts without rerunning, replacing or updating any replica."""
    directory = Path(directory)
    return {name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
            if (directory / f"{name}.json").exists() else None
            for name in ("campaign_manifest", "raw_results", "aggregation")}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run or inspect one-scenario NJEE campaigns.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--inspect", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.inspect is not None:
        if args.output_dir is not None:
            parser.error("--output-dir cannot be used with --inspect")
        print(json.dumps(inspect_campaign(args.inspect), indent=2))
    else:
        invocation = list(sys.argv) if argv is None else ["python -m experiments.neural_campaign", *argv]
        directory = run_campaign(args.config, output_dir=args.output_dir, invocation=invocation)
        print(f"Campaign artifacts saved to: {directory}")


if __name__ == "__main__":
    main()
