"""One-replica binary NJEE baseline; run with python -m experiments.neural_baseline.

CLI paths are relative to the current working directory. Each run owns a new
output_dir/run_id directory; existing directories, including failed runs, are
never reused. Only manifest.status='complete' certifies all outputs were written.
This runner makes no statistical accuracy claim or selection based on test loss.
"""

import argparse
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import Mapping

import numpy as np
from src.estimators.classical import MillerMadowEstimator, PlugInEstimator
from src.estimators.neural import NeuralJointEntropyEstimator
from src.simulation.factory import generate_dataset, get_ground_truth
from src.utils.categorical import CategoricalDataset
from src.utils.config import load_config
from src.utils.neural_config import validate_neural_config
from src.utils.reproducibility import capture_environment
from src.utils.split import make_iid_split, save_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("njee", "plug_in_full", "miller_madow_full", "plug_in_train", "miller_madow_train")


def _run_id(value):
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value)
            or value.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                                *[f"LPT{i}" for i in range(1, 10)]}):
        raise ValueError("run_id must be a portable identifier of 1-64 letters/digits/_/- starting with a letter.")
    return value


def load_baseline_config(path):
    """Reuse generator validation and a nested, fully explicit NeuralConfig.

    Top-level keys are exactly seed, data, neural, output_dir, run_id. Split
    seed/fractions live in neural and control the actual IID allocation. The
    current binary generators require declared {0,1} alphabets, never inferred.
    """
    config = load_config(path)
    if set(config) != {"seed", "data", "neural", "output_dir", "run_id"}:
        raise ValueError("Expected exactly seed, data, neural, output_dir and run_id.")
    if set(config["data"]) != {"generator", "n", "d", "params"}:
        raise ValueError("Unsupported data configuration keys.")
    neural = validate_neural_config(config["neural"], variable_ids=tuple(range(config["data"]["d"])),
                                    n_samples=config["data"]["n"])
    if neural.alphabet_policy != "declared":
        raise ValueError("Binary baseline requires declared binary alphabets.")
    if not isinstance(config["output_dir"], str) or not config["output_dir"].strip():
        raise ValueError("output_dir must be a nonempty path string.")
    _run_id(config["run_id"])
    # YAML lists become validated tuples; persist the settings actually executed.
    config["neural"] = asdict(neural)
    return config, neural


def _json_value(value):
    # dataclasses.asdict cannot copy the report's MappingProxyType environment.
    if is_dataclass(value):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path, value):
    """Replace only this new run's own file; complete status is written last."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_provenance():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
        return {"commit": commit, "dirty": bool(status), "status_porcelain": status.splitlines(),
                "limitation": None}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "status_porcelain": None,
                "limitation": f"Git metadata unavailable: {type(exc).__name__}"}


def run_baseline(config_path, *, output_dir=None, run_id=None, invocation=None):
    """Execute one configured run and return its new directory. Failures propagate.

    Configuration errors occur before output creation. Execution failures leave
    a failed manifest (or running status if interrupted), never complete status.
    No resume, overwriting, replica aggregation or experiment-grid support.
    """
    config_path = Path(config_path).resolve()
    config, neural = load_baseline_config(config_path)
    if output_dir is not None:
        if not str(output_dir).strip():
            raise ValueError("output_dir must be nonempty.")
        config["output_dir"] = str(output_dir)
    if run_id is not None:
        config["run_id"] = _run_id(run_id)
    output = Path(config["output_dir"]).resolve()
    config["output_dir"] = str(output)
    directory = output / config["run_id"]
    git = _git_provenance()  # Before creating run artifacts.
    output.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)  # Also refuses an existing partial/failed run.
    manifest = {
        "schema_version": "njee-single-baseline-v1", "run_id": config["run_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "resolved_configuration": config,
        "invocation": {"config_path": str(config_path), "output_dir": str(output),
                       "run_id": config["run_id"], "argv": invocation},
        "seeds": {"data_generation": config["seed"], "split": neural.split_seed,
                  "training": neural.training_seed, "effective_terms": []},
        "estimator_variants": list(VARIANTS), "entropy_unit": "nats",
        "component_order": neural.component_order, "git": git,
        "environment": None, "limitations": [],
        "protocol_note": "Classical full/train and NJEE use different effective estimation samples.",
    }
    path = directory / "manifest.json"
    _write_json(path, manifest)
    try:
        try:
            manifest["environment"] = capture_environment(mode=neural.reproducibility_mode, device=neural.device)
        except (RuntimeError, OSError) as exc:
            manifest["limitations"].append(f"Environment metadata unavailable: {type(exc).__name__}")
            raise
        manifest["environment_note"] = "Runner environment; active training backend settings are in term diagnostics."
        manifest["limitations"] = ["No cross-hardware/software bitwise guarantee.",
                                    "Single replica; no statistical accuracy or convergence guarantee."]
        values = generate_dataset(config)
        truth = float(get_ground_truth(config))
        split = make_iid_split(len(values), fractions=neural.split_fractions, seed=neural.split_seed)
        digest = hashlib.sha256(values.tobytes(order="C")).hexdigest()
        dataset_id = f"{config['run_id']}:{digest}"
        split_id = f"{config['run_id']}:{split.algorithm}:{split.seed}"
        dataset = CategoricalDataset(values, dataset_id, tuple(range(values.shape[1])))
        train, validation, test = (dataset.batch(split, role, split_id=split_id)
                                   for role in ("train", "validation", "test"))
        save_split(split, directory / "split.npz")
        np.savez(directory / "dataset.npz", values=values)
        sizes = {"full": len(values), "train": len(train.values),
                 "validation": len(validation.values), "test": len(test.values)}
        manifest.update({"ground_truth_nats": truth, "dataset_shape": list(values.shape),
                         "sample_counts": sizes, "dataset_id": dataset_id, "split_id": split_id,
                         "dataset_sha256": digest, "dataset_dtype": str(values.dtype),
                         "split_algorithm": split.algorithm})
        alphabets = {v: np.array([0, 1], dtype=np.int64) for v in dataset.variable_ids}
        estimator = NeuralJointEntropyEstimator(neural, alphabets=alphabets)
        start = perf_counter()
        estimator.fit(train, validation)
        report = estimator.estimate(test)
        runtime = perf_counter() - start
        records = [{"variant": "njee", "entropy_nats": report.entropy_nats,
                    "ground_truth_nats": truth, "signed_error_nats": report.entropy_nats - truth,
                    "sample_counts": sizes, "estimation_samples": {
                        "marginal_train": sizes["train"],
                        "conditional_train": sizes["train"] if len(report.terms) > 1 else 0,
                        "conditional_validation": sizes["validation"] if len(report.terms) > 1 else 0,
                        "conditional_test": sizes["test"] if len(report.terms) > 1 else 0},
                    "runtime_seconds": runtime, "entropy_unit": "nats"}]
        for partition, samples in (("full", values), ("train", train.values)):
            for name, classical in (("plug_in", PlugInEstimator()), ("miller_madow", MillerMadowEstimator())):
                start = perf_counter()
                estimate = classical.estimate(samples)
                records.append({"variant": f"{name}_{partition}", "entropy_nats": estimate,
                                "ground_truth_nats": truth, "signed_error_nats": estimate - truth,
                                "sample_counts": sizes, "estimation_samples": {partition: len(samples)},
                                "runtime_seconds": perf_counter() - start, "entropy_unit": "nats"})
        manifest["seeds"]["effective_terms"] = [
            {"position": t.position, "target": t.target, "conditioning": t.conditioning,
             "initialization_seed": t.initialization_seed, "shuffle_seed": t.shuffle_seed}
            for t in report.terms if t.method == "neural_test_ce"]
        _write_json(directory / "results.json", records)
        _write_json(directory / "njee_diagnostics.json", report)
        manifest["output_files"] = ["dataset.npz", "split.npz", "split.json", "results.json",
                                    "njee_diagnostics.json", "manifest.json"]
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest.pop("completed_at_utc", None)
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(path, manifest)
        raise
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one NJEE binary baseline replica.")
    parser.add_argument("--config", type=Path, default=Path("configs/neural_baseline.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    invocation = list(sys.argv) if argv is None else ["python -m experiments.neural_baseline", *argv]
    directory = run_baseline(args.config, output_dir=args.output_dir, run_id=args.run_id,
                             invocation=invocation)
    print(f"Complete NJEE baseline saved to: {directory}")


if __name__ == "__main__":
    main()
