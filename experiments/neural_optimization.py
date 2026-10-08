"""Task 013A: fixed, paired TRAIN/VALIDATION-only development diagnostic.

No entropy estimator or campaign runner is invoked. Test row indices are stored
as split metadata, but their values are never selected, encoded or evaluated.
"""
import argparse
import _thread
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.neural_campaign import derive_replica_seeds
from src.estimators.neural_training import train_conditional_term
from src.simulation.factory import generate_dataset
from src.utils.categorical import CategoricalBatch, CategoricalEncoder
from src.utils.neural_config import NeuralConfig
from src.utils.split import make_iid_split, save_split

CELLS = (
    ('T1', 'independent_bernoulli', 'p', .5, 4, 800, 130010),
    ('T2', 'binary_chain', 'epsilon', .15, 4, 200, 130011),
    ('T3', 'binary_chain', 'epsilon', .15, 12, 200, 130012),
    ('T4', 'binary_chain', 'epsilon', .15, 12, 800, 130013),
)
CANDIDATES = {'A': (20, 5), 'B': (60, 5), 'C': (60, 10)}
N_REPLICAS = 8
BUDGET_SECONDS = 1800


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _identity(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _configuration(seeds, d, candidate):
    epochs, patience = CANDIDATES[candidate]
    return NeuralConfig(seeds['split_seed'], (.6, .2, .2), seeds['training_seed'],
                        tuple(range(d)), 'declared', 'train_plugin', (16,), 'relu',
                        'adam', .01, 64, epochs, patience, .0001, 'cpu', 'strict', 0)


def prepare_train_validation(values, split, *, dataset_id, split_id):
    """Index only TRAIN/VALIDATION; do not wrap or scan the full value array."""
    ids = tuple(range(values.shape[1]))
    batches = [CategoricalBatch(values[getattr(split, role)], dataset_id, ids,
                                getattr(split, role), role, split_id, split.n_samples)
               for role in ('train', 'validation')]
    encoder = CategoricalEncoder.fit(batches[0], policy='declared',
                                    alphabets={v: np.array([0, 1]) for v in ids})
    return *batches, encoder


def evaluate_candidate(train, validation, encoder, config, *, record, path, deadline):
    """Persist every completed term; preserve partial records on any failure."""
    tick = perf_counter()
    record.update(status='running', terms=[], mean_best_validation_ce_nats=None,
                  started_at_utc=_now(), failure=None)
    _write(path, record)
    try:
        for position, target in enumerate(config.component_order[1:], 2):
            if perf_counter() >= deadline:
                raise TimeoutError('Task 013A execution budget reached.')
            result = train_conditional_term(train, validation, encoder=encoder, config=config,
                                            target=target, conditioning=config.component_order[:position-1])
            term = {k: v for k, v in vars(result).items() if k != 'model'}
            term['position'] = position
            # Guard a malformed trainer return before publishing a success.
            for key in ('best_validation_ce_nats', 'final_training_ce_nats', 'runtime_seconds'):
                if not math.isfinite(term[key]):
                    raise RuntimeError(f'Nonfinite recorded {key}.')
            if any(not math.isfinite(x) for key in ('epoch_training_ce_nats', 'epoch_validation_ce_nats')
                   for x in term[key]):
                raise RuntimeError('Nonfinite recorded epoch history.')
            record['terms'].append(term)
            _write(path, record)
        record.update(status='complete', mean_best_validation_ce_nats=math.fsum(
            t['best_validation_ce_nats'] for t in record['terms']) / len(record['terms']))
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else 'failed',
                      failure={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        record.update(wall_runtime_seconds=perf_counter()-tick, finished_at_utc=_now())
        _write(path, record)
    return record


def select_policy(records):
    """Frozen paired rule; incomplete/failing evidence never yields a selection."""
    expected = {(cell[0], r, c) for cell in CELLS for r in range(N_REPLICAS) for c in CANDIDATES}
    keys = [(r['cell'], r['replica'], r['candidate']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate candidate/replica record.')
    if set(keys) != expected or any(r['status'] != 'complete' for r in records):
        return {'status': 'paused_incomplete_or_failed', 'recommendation': None}
    by_key = dict(zip(keys, records))
    comparisons = {}
    for x, y in (('B', 'A'), ('C', 'A'), ('C', 'B')):
        cells = {}
        for cell, *_ in CELLS:
            differences = []
            for r in range(N_REPLICAS):
                left, right = by_key[cell, r, x], by_key[cell, r, y]
                for field in ('dataset_id', 'split_id', 'seeds', 'categorical_mappings'):
                    if left[field] != right[field]:
                        raise ValueError(f'Candidate pairing differs: {field}.')
                if [(t['initialization_seed'], t['shuffle_seed']) for t in left['terms']] != [
                    (t['initialization_seed'], t['shuffle_seed']) for t in right['terms']]:
                    raise ValueError('Per-term seed pairing differs.')
                a, b = left['mean_best_validation_ce_nats'], right['mean_best_validation_ce_nats']
                if not math.isfinite(a) or not math.isfinite(b):
                    return {'status': 'paused_nonfinite', 'recommendation': None}
                differences.append({'replica': r, 'improvement_nats': b-a})
            mean = math.fsum(row['improvement_nats'] for row in differences)/N_REPLICAS
            positive = sum(row['improvement_nats'] > 0 for row in differences)
            cells[cell] = {'mean_improvement_nats': mean, 'n_strictly_positive': positive,
                           'meets_chain_criterion': mean >= .005 and positive >= 6,
                           'paired_differences': differences}
        qualifies = sum(cells[c]['meets_chain_criterion'] for c in ('T2', 'T3', 'T4')) >= 2 and cells['T1']['mean_improvement_nats'] >= -.005
        comparisons[f'{x}_versus_{y}'] = {'cells': cells, 'qualifies': qualifies}
    b, c = comparisons['B_versus_A']['qualifies'], comparisons['C_versus_A']['qualifies']
    choice = 'A' if not b and not c else 'C' if c and (not b or comparisons['C_versus_B']['qualifies']) else 'B'
    return {'status': 'complete', 'recommendation': choice, 'comparisons': comparisons,
            'qualification_threshold_nats': .005, 'positive_replica_threshold': 6,
            'note': 'Development recommendation only; requires explicit approval.'}


def run_diagnostic(output_dir):
    """Run exactly 32 fresh datasets and 96 candidate evaluations; never resume."""
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    deadline = start + BUDGET_SECONDS
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, capture_output=True, text=True)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT, capture_output=True, text=True)
    manifest = {'schema': 'task013a-train-validation-v1', 'status': 'running', 'started_at_utc': _now(),
                'budget_seconds': BUDGET_SECONDS, 'cells': CELLS, 'candidates': CANDIDATES,
                'n_replicas': N_REPLICAS, 'git': {'commit': git.stdout.strip(), 'status': status.stdout.splitlines()},
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'tests_sha256': hashlib.sha256((PROJECT_ROOT/'tests/test_neural_optimization.py').read_bytes()).hexdigest(),
                'invocation': sys.argv, 'access_policy': 'TRAIN/VALIDATION values only; test indices are metadata.',
                'dataset_identity_policy': 'SHA256 of data configuration and generation seed; train/validation content hashes saved separately.',
                'seed_policy': 'njee-campaign-seedsequence-v1; existing derivation helper only, no campaign execution.'}
    records = [{'cell': cell[0], 'replica': r, 'candidate': c, 'status': 'not_attempted'}
               for cell in CELLS for r in range(N_REPLICAS) for c in CANDIDATES]
    _write(directory/'manifest.json', manifest)
    _write(directory/'evaluations.json', records)
    timer = threading.Timer(BUDGET_SECONDS, _thread.interrupt_main)
    timer.daemon = True
    timer.start()
    try:
        all_seeds = [derive_replica_seeds(cell[-1], r) for cell in CELLS for r in range(N_REPLICAS)]
        if len({s['data_seed'] for s in all_seeds}) != len(all_seeds):
            raise ValueError('Development dataset-seed collision.')
        pilot = PROJECT_ROOT/'results/task012_pilot_20261008'
        pilot_seeds = {r['data_seed'] for p in pilot.glob('*/raw_results.json') for r in json.loads(p.read_text())}
        if pilot_seeds & {s['data_seed'] for s in all_seeds}:
            raise ValueError('Development/pilot dataset-seed collision.')
        for cell, generator, parameter, value, d, n, master in CELLS:
            for replica in range(N_REPLICAS):
                if perf_counter() >= deadline:
                    raise TimeoutError('Task 013A execution budget reached.')
                seeds = derive_replica_seeds(master, replica)
                data_config = {'seed': seeds['data_seed'], 'data': {'generator': generator, 'd': d, 'n': n, 'params': {parameter: value}}}
                values = generate_dataset(data_config)
                split = make_iid_split(n, seed=seeds['split_seed'])
                dataset_id = _identity(data_config)
                split_id = _identity({'dataset': dataset_id, 'seed': split.seed, 'fractions': split.fractions, 'algorithm': split.algorithm})
                train, validation, encoder = prepare_train_validation(values, split, dataset_id=dataset_id, split_id=split_id)
                del values
                replica_dir = directory/cell/f'replica-{replica:06d}'
                replica_dir.mkdir(parents=True, exist_ok=False)
                save_split(split, replica_dir/'split.npz')
                np.savez(replica_dir/'train_validation.npz', train=train.values, validation=validation.values,
                         train_indices=train.row_indices, validation_indices=validation.row_indices)
                provenance = {'dataset_id': dataset_id, 'split_id': split_id, 'seeds': seeds,
                              'data_configuration': data_config, 'categorical_mappings': {str(v): dict(encoder.mappings[v]) for v in encoder.variable_ids},
                              'train_sha256': hashlib.sha256(train.values.tobytes()).hexdigest(),
                              'validation_sha256': hashlib.sha256(validation.values.tobytes()).hexdigest()}
                _write(replica_dir/'provenance.json', provenance)
                paired_seeds = None
                for candidate in CANDIDATES:
                    config = _configuration(seeds, d, candidate)
                    record = next(r for r in records if (r['cell'], r['replica'], r['candidate']) == (cell, replica, candidate))
                    record.update(**provenance, resolved_configuration=asdict(config))
                    print(f'START {cell} replica={replica} candidate={candidate}', flush=True)
                    try:
                        evaluate_candidate(train, validation, encoder, config, record=record,
                                           path=replica_dir/f'{candidate}.json', deadline=deadline)
                        term_seeds = [(t['initialization_seed'], t['shuffle_seed']) for t in record['terms']]
                        if paired_seeds is not None and term_seeds != paired_seeds:
                            raise ValueError('Candidate term seed pairing changed.')
                        paired_seeds = term_seeds
                    finally:
                        _write(directory/'evaluations.json', records)
                    print(f'END {cell} replica={replica} candidate={candidate}: {record["status"]}', flush=True)
        _write(directory/'selection.json', select_policy(records))
        manifest['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else 'failed',
                        failure={'type': type(exc).__name__, 'message': str(exc)})
        _write(directory/'selection.json', {'status': 'paused', 'recommendation': None})
        raise
    finally:
        timer.cancel()
        manifest.update(wall_runtime_seconds=perf_counter()-start, finished_at_utc=_now())
        _write(directory/'manifest.json', manifest)
        _write(directory/'evaluations.json', records)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(run_diagnostic(args.output_dir))
