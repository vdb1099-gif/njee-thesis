"""Portable ablation preparation/control/reporting; default is metadata-only preview."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments import ablation_protocol as protocol
from experiments.information_validation import environment, _git_provenance
from experiments.transfer_validation import write_json, write_csv
from src.estimators.ablation import estimate_entropy
from src.estimators.transfer_entropy import NeuralTransferEntropyEstimator
from src.metrics.ablation import summarize
from src.simulation.binary_chain import generate_binary_chain
from src.simulation.bernoulli import generate_independent_bernoulli
from src.simulation.coupled_binary import CoupledBinaryModel, reference_te, coupling_bound
from src.simulation.ground_truth import binary_chain_entropy
from src.utils.ablation_data import nested_training_subset, discretize_uniform, discretized_entropy, array_digest
from src.utils.categorical import CategoricalDataset
from src.utils.split import make_iid_split, save_split
from src.utils.temporal import TemporalEmbeddingSpec


def process_model():
    t = protocol.definition()['temporal']
    return CoupledBinaryModel(t['coupling'], t['delay'], t['noise_x'], t['noise_y'])


def generate_group(group, seed):
    d = protocol.definition()
    if group == 'temporal':
        t = d['temporal']
        x, y, metadata = process_model().generate(d['n'], seed=seed, burn_in=t['burn_in'], initial=t['initial'])
        return np.column_stack((x, y)), metadata
    if group == 'independent':
        return generate_independent_bernoulli(n=d['n'], d=d['d'], p=.5, seed=seed), {'law': 'independent Bernoulli(.5)'}
    if group == 'binary_chain':
        return generate_binary_chain(d['n'], d['d'], d['epsilon'], seed), {'law': 'binary chain', 'epsilon': d['epsilon']}
    if group == 'discretization':
        return np.random.default_rng(seed).random((d['n'], d['d'])), {'law': 'independent Uniform[0,1)'}
    raise ValueError('Unknown group.')


def fit_temporal(values, cell, seed, identity):
    """Ablation-local alignment, using the unchanged validated TE public API."""
    t = protocol.definition()['temporal']
    spec = TemporalEmbeddingSpec(tuple(cell['source_lags']), tuple(cell['target_lags']), cell['horizon'])
    offset = t['common_span'] - max(spec.source_lags + spec.target_lags)
    if offset < 0 or t['gap'] < t['common_span'] or not t['purge']:
        raise ValueError('Invalid temporal alignment.')
    source, target = (0, 1) if cell['direction'] == 'X_to_Y' else (1, 0)
    sid, tid = ('X', 'Y') if source == 0 else ('Y', 'X')
    config = replace(protocol.config_for(cell, seed), component_order=spec.target_order(tid))
    estimator = NeuralTransferEntropyEstimator(config, embedding=spec, source_id=sid, target_id=tid,
        source_alphabet=np.array([0, 1]), target_alphabet=np.array([0, 1]), gap=t['gap'], purge=t['purge'])
    times = np.arange(offset, len(values), dtype=np.int64)
    estimator.fit(values[offset:, source], values[offset:, target], dataset_id=identity,
                  source_times=times, target_times=times)
    report = estimator.estimate()
    truth = reference_te(process_model(), spec, direction=cell['direction'])
    costs = [term for c in report.cmi_report.components for term in c.entropy.terms]
    checkpoints = {}
    parameters = 0
    for ci, entry in enumerate(estimator._model._components):
        for ti, term in enumerate(entry[4]._results):
            checkpoints[f'component-{ci}-term-{ti}'] = {k: v.detach().cpu().numpy().copy() for k, v in term.model.state_dict().items()}
            parameters += sum(p.numel() for p in term.model.parameters())
    n_train = len(estimator._parts.train.values)
    updates = sum(term.epochs_trained * int(np.ceil(n_train / config.batch_size)) for term in costs)
    effective = []
    for component in report.cmi_report.components:
        effective.append(component.config.training_seed)
        for term in component.entropy.terms:
            effective.extend((term.initialization_seed, term.shuffle_seed))
    return {'estimate_nats': report.transfer_entropy_nats, 'truth_nats': truth['transfer_entropy_nats'],
            'reference': truth, 'diagnostics': report, 'parameter_count': parameters,
            'optimizer_updates': updates, 'term_batch_evaluations': updates, 'n_train': n_train,
            'effective_seeds': effective,
            'alignment_support': [list(p.target_timestamps) for p in report.partitions],
            'initialization_tv_bound': coupling_bound(process_model(), t['burn_in'], t['common_span'])}, checkpoints


def empty_rows(plan):
    return [{'cell_id': c['cell_id'], 'configuration_hash': c['configuration_hash'],
             'replica_id': r['replica_id'], 'group': c['group'], 'status': 'not_attempted',
             'estimate_nats': None, 'truth_nats': None, 'signed_error_nats': None, 'runtime_seconds': None}
            for g in plan['groups'] for r in g['allocation'] for c in plan['cells'] if c['group'] == g['group_id']]


def run_group_replica(group, allocation, cells, directory, *, cancelled=lambda: False):
    """One shared dataset/process and split; all variants are attempted once."""
    d = protocol.definition()
    rows = []
    manifest = {'status': 'running', 'group': group, 'allocation': allocation, 'environment': environment()}
    write_json(directory / 'manifest.json', manifest)
    for cell in cells:
        rows.append({'cell_id': cell['cell_id'], 'configuration_hash': cell['configuration_hash'],
                     'replica_id': allocation['replica_id'], 'group': group, 'status': 'not_attempted',
                     'estimate_nats': None, 'truth_nats': None, 'signed_error_nats': None, 'runtime_seconds': None})
    write_json(directory / 'raw_results.json', rows)
    start = perf_counter()
    try:
        if cancelled():
            raise KeyboardInterrupt('Cancelled before generation.')
        values, generation = generate_group(group, allocation['data']['seed'])
        data_hash = array_digest(values)
        identity = f'{group}:{allocation["replica_id"]}:{data_hash}'
        split_identity = f'{identity}:chronological-span3-gap3' if group == 'temporal' else f'{identity}:iid:{allocation["split"]["seed"]}'
        manifest.update(dataset_id=identity, parent_split_id=split_identity, generation=generation, data_sha256=data_hash)
        np.savez_compressed(directory / 'dataset.npz', values=values)
        if group != 'temporal':
            split = make_iid_split(len(values), seed=allocation['split']['seed'])
            save_split(split, directory / 'split.npz')
        support = None
        for cell, row in zip(cells, rows):
            if cancelled():
                raise KeyboardInterrupt('Cancelled before variant.')
            row.update(status='running', parent_dataset_id=identity, parent_split_id=split_identity,
                       seeds=allocation['fits'][cell['cell_id']])
            write_json(directory / 'raw_results.json', rows)
            cd = directory / cell['cell_id']
            cd.mkdir(exist_ok=False)
            job_start = perf_counter()
            try:
                seed = allocation['fits'][cell['cell_id']]['seed']
                if group == 'temporal':
                    result, checkpoints = fit_temporal(values, cell, seed, identity)
                    if support is not None and support != result['alignment_support']:
                        raise ValueError('Embedding comparisons have unequal chronological supports.')
                    support = result['alignment_support']
                else:
                    categories = discretize_uniform(values, cell['states']) if group == 'discretization' else values
                    dataset = CategoricalDataset(categories, identity, tuple(range(d['d'])))
                    train, validation, test = (dataset.batch(split, role, split_id=split_identity)
                                               for role in ('train', 'validation', 'test'))
                    train = nested_training_subset(train, cell['train_fraction'], seed=allocation['subset']['seed'])
                    config = protocol.config_for(cell, seed, allocation['split']['seed'])
                    report, checkpoints = estimate_entropy(train, validation, test, config=config,
                        alphabets={v: np.arange(cell['states']) for v in dataset.variable_ids},
                        variant=cell['variant'], early_stopping=cell['early_stopping'], cancelled=cancelled)
                    truth = (float(binary_chain_entropy(d['epsilon'], d['d'])) if group == 'binary_chain'
                             else discretized_entropy(d['d'], cell['states']))
                    result = {'estimate_nats': report['entropy_nats'], 'truth_nats': truth, 'diagnostics': report,
                              'parameter_count': report['parameter_count'], 'optimizer_updates': report['optimizer_updates'],
                              'term_batch_evaluations': report['term_batch_evaluations'], 'n_train': len(train.values),
                              'category_sha256': array_digest(categories), 'subset_rows': train.row_indices.tolist()}
                    result['effective_seeds'] = report['effective_seeds']
                if result['effective_seeds'] != allocation['fits'][cell['cell_id']]['effective_seeds']:
                    raise ValueError('Executed initialization/shuffle seeds differ from frozen allocation.')
                if cancelled():
                    raise KeyboardInterrupt('Cancelled after fitting.')
                if not np.isfinite([result['estimate_nats'], result['truth_nats']]).all():
                    raise ValueError('Nonfinite ablation result.')
                write_json(cd / 'diagnostics.json', result)
                for name, arrays in checkpoints.items():
                    np.savez_compressed(cd / (name + '.npz'), **arrays)
                row.update({k: v for k, v in result.items() if k != 'diagnostics'})
                row.update(status='complete', signed_error_nats=result['estimate_nats'] - result['truth_nats'])
            except (Exception, KeyboardInterrupt) as exc:
                row.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                           failure={'type': type(exc).__name__, 'message': str(exc)})
                write_json(cd / 'failure.json', row['failure'])
                if isinstance(exc, KeyboardInterrupt):
                    raise
            finally:
                row['runtime_seconds'] = perf_counter() - job_start
                write_json(directory / 'raw_results.json', rows)
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        failure={'type': type(exc).__name__, 'message': str(exc)})
        if not isinstance(exc, KeyboardInterrupt):
            for row in rows:
                if row['status'] == 'not_attempted':
                    row.update(status='failed', failure=manifest['failure'])
        if isinstance(exc, KeyboardInterrupt):
            raise
    else:
        manifest['status'] = 'complete' if all(r['status'] == 'complete' for r in rows) else 'complete_with_failures'
    finally:
        manifest['runtime_seconds'] = perf_counter() - start
        write_json(directory / 'raw_results.json', rows)
        manifest['artifact_sha256'] = {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in directory.rglob('*') if p.is_file() and p != directory / 'manifest.json'}
        write_json(directory / 'manifest.json', manifest)
    return rows, manifest


def read_records(root, plan):
    expected = {(r['cell_id'], r['replica_id']): r for r in empty_rows(plan)}
    for group in plan['groups']:
        for allocation in group['allocation']:
            directory = root / group['group_id'] / allocation['replica_id']
            path = directory / 'raw_results.json'
            if not path.exists():
                continue
            rows = json.loads(path.read_text(encoding='utf-8'))
            names = {c['cell_id'] for c in plan['cells'] if c['group'] == group['group_id']}
            if len(rows) != len(names) or {r['cell_id'] for r in rows} != names:
                raise ValueError('Incomplete/duplicate variant ledger.')
            for row in rows:
                key = (row['cell_id'], row['replica_id'])
                if key not in expected or row['replica_id'] != allocation['replica_id']:
                    raise ValueError('Unexpected replica identity.')
                if any(row[k] != expected[key][k] for k in ('group', 'configuration_hash')):
                    raise ValueError('Frozen configuration identity changed.')
                if row.get('seeds') is not None and row['seeds'] != allocation['fits'][row['cell_id']]:
                    raise ValueError('Fit seeds changed.')
                expected[key] = row
    return list(expected.values())


def inspect_results(root):
    """Read-only: recompute tables from raw ledgers, validate completed artifacts."""
    root = Path(root)
    saved = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    if saved != protocol.preview(saved['output_root']):
        raise ValueError('Output protocol/hash changed.')
    processes = []
    for group in saved['groups']:
        for allocation in group['allocation']:
            directory = root / group['group_id'] / allocation['replica_id']
            path = directory / 'manifest.json'
            if not path.exists():
                processes.append({'group': group['group_id'], 'replica_id': allocation['replica_id'], 'status': 'not_attempted'})
                continue
            manifest = json.loads(path.read_text(encoding='utf-8'))
            if manifest['allocation'] != allocation or manifest['group'] != group['group_id']:
                raise ValueError('Shared dataset allocation changed.')
            for name, sha in manifest.get('artifact_sha256', {}).items():
                if hashlib.sha256((directory / protocol.relative_path(name)).read_bytes()).hexdigest() != sha:
                    raise ValueError('Artifact hash mismatch.')
            processes.append({'group': group['group_id'], 'replica_id': allocation['replica_id'],
                              'status': manifest['status'], 'runtime_seconds': manifest.get('runtime_seconds')})
    rows = read_records(root, saved)
    return {'protocol_hash': saved['protocol_hash'], 'raw_records': rows, 'analysis': summarize(rows, saved),
            'process_accounting': processes,
            'manifest': json.loads((root / 'manifest.json').read_text(encoding='utf-8'))}


def export_tables(root, plan):
    rows = read_records(root, plan)
    analysis = summarize(rows, plan)
    write_csv(root / 'replica_results.csv', rows)
    write_json(root / 'analysis.json', analysis)
    for name in ('summaries', 'paired_comparisons', 'ordering_variability'):
        write_csv(root / (name + '.csv'), analysis[name])
    terms = []
    for group in plan['groups']:
        for allocation in group['allocation']:
            for cell in (c for c in plan['cells'] if c['group'] == group['group_id']):
                path = root / group['group_id'] / allocation['replica_id'] / cell['cell_id'] / 'diagnostics.json'
                if not path.exists():
                    continue
                diagnostics = json.loads(path.read_text(encoding='utf-8'))['diagnostics']
                common = {'cell_id': cell['cell_id'], 'replica_id': allocation['replica_id']}
                if cell['kind'] == 'entropy':
                    terms.extend({**common, **t} for t in diagnostics['terms'])
                else:
                    for component in diagnostics['cmi_report']['components']:
                        terms.extend({**common, 'component': component['name'], **t}
                                     for t in component['entropy']['terms'])
    write_csv(root / 'term_diagnostics.csv', terms)


def execute(preparation, *, authorized=False, allow_local_windows=False, output_dir=None,
            log_dir=None, cancel_file=None):
    if not authorized:
        raise PermissionError('Separate scientific execution authorization required.')
    if os.name == 'nt' and not allow_local_windows:
        raise PermissionError('Separate local Windows authorization required.')
    plan = protocol.validate_bundle(preparation)
    git = _git_provenance()
    if git['dirty'] is not False:
        raise ValueError('A clean identifiable Git checkpoint is required.')
    root = Path(plan['output_root'] if output_dir is None else output_dir).resolve()
    prep = Path(preparation).resolve()
    logs = root / 'logs' if log_dir is None else Path(log_dir).resolve()
    if root == prep or root in prep.parents or prep in root.parents:
        raise ValueError('Preparation/output overlap.')
    if logs == root or logs in root.parents or logs == prep or logs in prep.parents or prep in logs.parents:
        raise ValueError('Unsafe log directory.')
    if root.exists() or logs.exists():
        raise FileExistsError('No overwrite or resume of existing outputs/logs.')
    env = environment()  # Fail preflight without leaving an orphan output root.
    root.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    write_json(root / 'protocol.json', plan)
    manifest = {'status': 'running', 'protocol_hash': plan['protocol_hash'], 'git': git, 'environment': env,
                'workdir': str(Path.cwd()), 'output_dir': str(root), 'log_dir': str(logs),
                'started_utc': datetime.now(timezone.utc).isoformat()}
    ledger = [{'group': g['group_id'], 'replica_id': r['replica_id'], 'status': 'not_attempted'}
              for g in plan['groups'] for r in g['allocation']]
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'execution_ledger.json', ledger)
    start = perf_counter()
    cancelled = lambda: cancel_file is not None and Path(cancel_file).exists()
    try:
        for group in plan['groups']:
            cells = [c for c in plan['cells'] if c['group'] == group['group_id']]
            for allocation in group['allocation']:
                if cancelled():
                    raise KeyboardInterrupt('Cancelled before dataset replica.')
                entry = next(e for e in ledger if (e['group'], e['replica_id']) == (group['group_id'], allocation['replica_id']))
                directory = root / group['group_id'] / allocation['replica_id']
                directory.mkdir(parents=True, exist_ok=False)
                entry['status'] = 'running'
                write_json(root / 'execution_ledger.json', ledger)
                try:
                    _, result = run_group_replica(group['group_id'], allocation, cells, directory, cancelled=cancelled)
                except BaseException:
                    entry['status'] = 'interrupted_or_failed'
                    raise
                else:
                    entry['status'] = result['status']
                finally:
                    write_json(root / 'execution_ledger.json', ledger)
                    with (logs / 'events.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(entry) + '\n')
        manifest['status'] = 'complete' if all(e['status'] == 'complete' for e in ledger) else 'complete_with_failures'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        failure={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        try:
            export_tables(root, plan)
        except BaseException as exc:
            manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'reporting_failed',
                            reporting_failure={'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            manifest.update(runtime_seconds=perf_counter() - start, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(root / 'manifest.json', manifest)
            write_json(root / 'execution_ledger.json', ledger)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--prepare', type=Path)
    group.add_argument('--execute', type=Path)
    group.add_argument('--report', type=Path)
    parser.add_argument('--authorize-execution', action='store_true')
    parser.add_argument('--allow-local-windows', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--cancel-file', type=Path)
    args = parser.parse_args(argv)
    if args.execute is None and (args.authorize_execution or args.allow_local_windows or args.log_dir or args.cancel_file):
        parser.error('Execution controls require --execute.')
    if args.report and args.output_dir:
        parser.error('--report is read-only.')
    protocol.load_definition()
    if args.execute:
        print(execute(args.execute, authorized=args.authorize_execution, allow_local_windows=args.allow_local_windows,
                      output_dir=args.output_dir, log_dir=args.log_dir, cancel_file=args.cancel_file))
    elif args.prepare:
        print(json.dumps(protocol.materialize(args.prepare, output_root=args.output_dir)['expected']))
    elif args.report:
        print(json.dumps(inspect_results(args.report), allow_nan=False))
    else:
        plan = protocol.preview(args.output_dir)
        print(json.dumps({k: plan[k] for k in ('protocol_hash', 'expected', 'execution_order', 'output_root')}, indent=2))


if __name__ == '__main__':
    main()
