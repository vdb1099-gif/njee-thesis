"""Replica-level error metrics and deterministic 95% percentile bootstrap.

Bootstrap units are independent synthetic replicas, never rows within a dataset.
Intervals require >=2 successful finite replicas; small development campaigns
still do not justify precision claims. No normal-theory interval is substituted.
Paired resampling uses the intersection of successful replica IDs jointly.
"""

import hashlib
import json
import math

import numpy as np

from src.utils.reproducibility import validate_seed


BOOTSTRAP_VERSION = "njee-replica-bootstrap-sha256-pcg64-v1"


def bootstrap_seed_for(seed, identifier):
    """Dedicated bootstrap domain; stable UTF-8 JSON, first 4 SHA-256 bytes."""
    validate_seed(seed)
    payload = json.dumps([BOOTSTRAP_VERSION, seed, identifier], separators=(",", ":"), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _metrics(estimates, truths):
    errors = estimates - truths
    return {"mean_estimate_nats": float(np.mean(estimates)), "bias_nats": float(np.mean(errors)),
            "mae_nats": float(np.mean(np.abs(errors))), "rmse_nats": float(np.sqrt(np.mean(errors**2))),
            "sample_std_nats": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else None}


def _intervals(estimates, truths, *, draws, seed):
    names = tuple(_metrics(np.array([0., 0.]), np.array([0., 0.])))
    metadata = {"method": "percentile", "confidence_level": .95, "draws": draws,
                "seed": seed, "rng": "PCG64", "unit": "replica", "n_units": len(estimates)}
    if len(estimates) < 2:
        return {name: {**metadata, "statistic": name, "lower": None, "upper": None,
                       "reason": "fewer_than_two_successful_replicas"} for name in names}
    rng = np.random.default_rng(seed)
    values = {name: [] for name in names}
    # One draw at a time bounds memory independently of draws * replica count.
    for _ in range(draws):
        indices = rng.integers(0, len(estimates), size=len(estimates))
        for name, value in _metrics(estimates[indices], truths[indices]).items():
            values[name].append(value)
    return {name: {**metadata, "statistic": name,
                   "lower": float(np.quantile(samples, .025, method="linear")),
                   "upper": float(np.quantile(samples, .975, method="linear")), "reason": None}
            for name, samples in values.items()}


def aggregate_records(records, *, planned_replica_ids, bootstrap_draws, bootstrap_seed,
                      paired_comparisons=()):
    """Aggregate long-form rows without mutating them or excluding failures silently.

    Each campaign/scenario/variant has exactly one row per planned replica, using
    not_attempted placeholders when necessary. Failed includes interrupted or
    nonfinite attempted results; missing timings are reported with separate counts.
    Bias=mean(error), MAE=mean(abs(error)), RMSE=sqrt(mean(error**2)); estimate SD
    uses ddof=1. Zero successes yield None metrics; one yields no SD or interval.
    """
    if (not isinstance(planned_replica_ids, (tuple, list)) or not planned_replica_ids
            or any(not isinstance(v, str) or not v for v in planned_replica_ids)
            or len(set(planned_replica_ids)) != len(planned_replica_ids)):
        raise ValueError("Planned replica IDs must be nonempty and unique.")
    if type(bootstrap_draws) is not int or bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be a positive Python int.")
    validate_seed(bootstrap_seed)
    groups, identities, truths_by_replica = {}, set(), {}
    for row in records:
        key = (row["campaign_id"], row["scenario_id"], row["variant"])
        identity = (*key, row["replica_id"])
        if identity in identities or row["replica_id"] not in planned_replica_ids:
            raise ValueError("Duplicate or unplanned raw-record identity.")
        identities.add(identity)
        if row["status"] not in ("complete", "failed", "interrupted", "running", "not_attempted"):
            raise ValueError("Unknown raw-record status.")
        if _finite(row["ground_truth_nats"]):
            truth_key = (key[0], key[1], row["replica_id"])
            if truth_key in truths_by_replica and truths_by_replica[truth_key] != row["ground_truth_nats"]:
                raise ValueError("Methods within a replica must share ground truth.")
            truths_by_replica[truth_key] = row["ground_truth_nats"]
        groups.setdefault(key, []).append(row)
    summaries, successful = [], {}
    for key, rows in sorted(groups.items()):
        if {r["replica_id"] for r in rows} != set(planned_replica_ids):
            raise ValueError("Every group requires all planned replica rows, including failures/placeholders.")
        rows = sorted(rows, key=lambda r: r["replica_id"])
        valid = [r for r in rows if r["status"] == "complete" and _finite(r["entropy_nats"])
                 and _finite(r["ground_truth_nats"])]
        successful[key] = {r["replica_id"]: r for r in valid}
        estimates = np.array([r["entropy_nats"] for r in valid], dtype=float)
        truths = np.array([r["ground_truth_nats"] for r in valid], dtype=float)
        metrics = _metrics(estimates, truths) if valid else {
            name: None for name in _metrics(np.array([0., 0.]), np.array([0., 0.]))}
        if any(v is not None and not math.isfinite(v) for v in metrics.values()):
            raise ValueError("Aggregate arithmetic produced a nonfinite metric.")
        attempted = sum(r["status"] != "not_attempted" for r in rows)
        summary = {"campaign_id": key[0], "scenario_id": key[1], "variant": key[2],
                   "n_planned": len(rows), "n_attempted": attempted, "n_successful": len(valid),
                   "n_failed": attempted - len(valid), "n_not_attempted": len(rows) - attempted,
                   "n_interrupted": sum(r["status"] == "interrupted" for r in rows), **metrics,
                   "n_invalid_estimates": sum(r["status"] == "complete" and r not in valid for r in rows),
                   "successful_replica_ids": [r["replica_id"] for r in valid],
                   "bootstrap": _intervals(estimates, truths, draws=bootstrap_draws,
                                            seed=bootstrap_seed_for(bootstrap_seed, list(key)))}
        for timing in ("fit_runtime_seconds", "inference_runtime_seconds", "runtime_seconds"):
            times = [r[timing] for r in valid if _finite(r[timing]) and r[timing] >= 0]
            summary[f"mean_{timing}"] = math.fsum(times) / len(times) if times else None
            summary[f"n_{timing}"] = len(times)
        summaries.append(summary)
    pairs = []
    for campaign, scenario in sorted({key[:2] for key in groups}):
        for left, right in paired_comparisons:
            a, b = successful.get((campaign, scenario, left), {}), successful.get((campaign, scenario, right), {})
            ids = sorted(set(a) & set(b))
            differences = [{"replica_id": i, "estimate_difference_nats": a[i]["entropy_nats"] - b[i]["entropy_nats"],
                            "absolute_error_difference_nats": abs(a[i]["entropy_nats"] - a[i]["ground_truth_nats"])
                            - abs(b[i]["entropy_nats"] - b[i]["ground_truth_nats"])} for i in ids]
            pair = {"campaign_id": campaign, "scenario_id": scenario, "left": left, "right": right,
                    "difference_direction": "left_minus_right", "n_planned": len(planned_replica_ids),
                    "n_paired_successful": len(ids), "n_unpaired": len(planned_replica_ids) - len(ids),
                    "differences": differences, "bootstrap": {}}
            for name in ("estimate_difference_nats", "absolute_error_difference_nats"):
                values = np.array([d[name] for d in differences])
                pair[f"mean_{name}"] = float(np.mean(values)) if ids else None
                interval = _intervals(values, np.zeros(len(values)), draws=bootstrap_draws,
                                      seed=bootstrap_seed_for(bootstrap_seed, [campaign, scenario, left, right, name]))
                pair["bootstrap"][name] = {**interval["mean_estimate_nats"], "statistic": f"mean_{name}",
                                            "unit": "paired_replica_id"}
            pairs.append(pair)
    return {"summaries": summaries, "paired_comparisons": pairs,
            "bootstrap_version": BOOTSTRAP_VERSION, "bootstrap_master_seed": bootstrap_seed,
            "bootstrap_note": "95% percentile, linear quantiles; resample replicas with replacement. "
                              "Small development samples and success-only intervals are not final evidence."}
