from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

from experiments.final_protocol import digest


HASHES = {
    'entropy': '7560871bfa180eb76d19a74d5012a90413857c4c4982801d8e1c9980e7aabf7b',
    'information': '278f07cda5c7859103c24f9156c035a7960f4e48f33653208ea6943abde4881c',
    'transfer': '56b184b96fcd7bf7a00bb376350e7f3dbf695d0e7ac91e62194b50f3e3e99720',
    'ablation': '878571f73c32fe166b5569ce113e6be6201df6ae21666266428598f5d8854eb1',
}


@lru_cache(maxsize=4)
def frozen_plan(experiment):
    if experiment == 'entropy':
        from experiments.final_protocol import preview
    elif experiment == 'information':
        from experiments.information_protocol import preview
    elif experiment == 'transfer':
        from experiments.transfer_protocol import preview
    elif experiment == 'ablation':
        from experiments.ablation_protocol import preview
    else:
        raise ValueError('Unknown experiment.')
    plan = preview()
    if plan['protocol_hash'] != HASHES[experiment]:
        raise ValueError('Frozen scientific protocol changed; scientific review required.')
    return plan


def read_json(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON literal: {value}')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON object key.')
            result[key] = value
        return result
    def real(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Nonfinite JSON number.')
        return number
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid,
                      parse_float=real, object_pairs_hook=unique)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Input must be an existing nonsymlink directory.')
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise ValueError('Symlink/escaping scientific input is unsupported.')
        if p.is_file():
            result[p.relative_to(root).as_posix()] = file_hash(p)
    return result


def safe_artifact(directory, name):
    if (not isinstance(name, str) or not name or '\\' in name or PureWindowsPath(name).drive
            or Path(name).is_absolute() or '..' in Path(name).parts):
        raise ValueError('Unsafe artifact path.')
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Escaping artifact path.')
    return path


def check_artifacts(directory, manifest):
    for name, expected in manifest.get('artifact_sha256', {}).items():
        p = safe_artifact(directory, name)
        if not p.is_file() or file_hash(p) != expected:
            raise ValueError('Scientific artifact hash mismatch: ' + str(p))


def canonical_rows(saved, expected, keys):
    """Missing rows become explicit NOT-ATTEMPTED records, never successful data."""
    index = {tuple(r[k] for k in keys): r for r in expected}
    seen = set()
    for row in saved:
        key = tuple(row[k] for k in keys)
        if key in seen or key not in index:
            raise ValueError('Duplicate or unplanned scientific record.')
        seen.add(key)
        index[key] = row
    return list(index.values()), len(expected) - len(seen)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def valid_value(row, value, truth):
    if row.get('entropy_unit', 'nats') != 'nats' or row.get('log_base', 'e') not in ('e', 'natural'):
        raise ValueError('Only nats / natural logarithms are supported.')
    if row['status'] not in ('complete', 'failed', 'interrupted', 'running', 'not_attempted'):
        raise ValueError('Unknown scientific status.')
    if row['status'] == 'complete':
        if not all(finite(row.get(k)) for k in (value, truth, 'signed_error_nats')):
            raise ValueError('Nonfinite/missing successful result.')
        if not math.isclose(row['signed_error_nats'], row[value] - row[truth], abs_tol=1e-12, rel_tol=0):
            raise ValueError('Signed error inconsistent with estimate/reference.')
        if not finite(row.get('runtime_seconds')) or row['runtime_seconds'] < 0:
            raise ValueError('Missing/nonfinite successful runtime.')


def reference_metadata(reference, cell, job):
    """Validate stored numerical evidence, NEVER execute the stationary solver."""
    if reference['status'] not in ('analytic_zero', 'numerical_stationary_enumeration'):
        raise ValueError('Unknown TE reference status.')
    if not all(finite(reference.get(k)) for k in ('transfer_entropy_nats', 'first_entropy_nats',
        'second_entropy_nats', 'tv_truncation_bound', 'stationarity_l1_residual', 'entropy_truncation_error_bound_nats')):
        raise ValueError('Missing/nonfinite reference metadata.')
    memory = max(cell.get('delay', 2), *job['source_lags'], *job['target_lags'])
    if (reference.get('memory') != memory or reference.get('n_states') != 4 ** memory
            or type(reference.get('iterations')) is not int or not 1 <= reference['iterations'] <= 10000):
        raise ValueError('Stationary reference state space/iteration metadata differs.')
    if (reference['version'] != 'pair-chain-power-coupling-v1'
            or reference['direction'] != job['direction'] or reference['source_lags'] != job['source_lags']
            or reference['target_lags'] != job['target_lags'] or reference['horizon'] != 1
            or 'excludes roundoff' not in reference['floating_point']
            or not 0 <= reference['tv_truncation_bound'] <= 1e-13
            or not 0 <= reference['stationarity_l1_residual'] <= 1e-10
            or reference['entropy_truncation_error_bound_nats'] < 0
            or not math.isclose(reference['transfer_entropy_nats'], reference['first_entropy_nats'] - reference['second_entropy_nats'], abs_tol=1e-12, rel_tol=0)):
        raise ValueError('Inconsistent numerical reference contract.')
    analytic = cell.get('coupling', .3) == 0 or cell.get('noise_y', .1) == .5 or (job['direction'] == 'Y_to_X' and 1 in job['target_lags'])
    if not analytic and reference['status'] == 'analytic_zero':
        raise ValueError('Unjustified analytic-zero reference label.')
    if analytic and (reference['status'] != 'analytic_zero' or reference['transfer_entropy_nats'] != 0.):
        raise ValueError('Analytic zero reference mislabelled.')
    return {**reference, 'roundoff_certified': False, 'scope': 'stationary finite embedding; not infinite-history TE'}


def _terms(path, cell_id, replica_id, variant, output, issues, required=False):
    if not path.exists():
        if required:
            issues.append(f'Missing diagnostics: {cell_id}/{replica_id}')
        return
    report = read_json(path)
    if 'diagnostics' in report:
        report = report['diagnostics']
    if 'cmi_report' in report:
        report = report['cmi_report']
    components = report.get('components', [{'name': 'joint', 'entropy': report}])
    for component in components:
        entropy = component['entropy']
        for i, term in enumerate(entropy.get('terms', []), 1):
            output.append({'cell_id': cell_id, 'replica_id': replica_id, 'variant': variant,
                           'component': component['name'], 'position': term.get('position', i), **term})


def _identity_manifest(path, expected, issues, required):
    if not path.exists():
        if required:
            issues.append('Missing replica manifest: ' + str(path))
        return None
    manifest = read_json(path)
    check_artifacts(path.parent, manifest)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError('Replica manifest identity mismatch: ' + key)
    if required and manifest.get('status') != 'complete':
        issues.append('Replica not complete: ' + str(path))
    if required and expected and not manifest.get('artifact_sha256'):
        issues.append('Missing replica artifact inventory: ' + str(path))
    return manifest


def temporal_diagnostics(path, row, job, *, n, span, gap, issues):
    if not path.exists():
        return
    from src.utils.split import _allocation
    report = read_json(path)
    report = report.get('diagnostics', report)
    if (report['embedding'] != {'source_lags': job['source_lags'], 'target_lags': job['target_lags'], 'horizon': 1}
            or report['gap'] != gap or report['purge'] is not True or report['split_fractions'] != [.6, .2, .2]):
        raise ValueError('Temporal diagnostic policy differs from frozen job.')
    counts = _allocation(n - span, (.6, .2, .2), 0)
    edges = [0, counts[0], counts[0] + counts[1], sum(counts)]
    parts = report['partitions']
    if len(parts) != 3:
        raise ValueError('Temporal partitions missing.')
    for i, (role, part) in enumerate(zip(('train', 'validation', 'test'), parts)):
        expected = list(range(span + edges[i] + (gap if i else 0), span + edges[i + 1]))
        if part['role'] != role or part['target_timestamps'] != expected:
            raise ValueError('Unaligned temporal prediction timestamps.')
        observations = [[t, *[t - lag for lag in job['source_lags'] + job['target_lags']]] for t in expected]
        if part['observation_timestamps'] != observations:
            raise ValueError('Temporal observation/target leakage or lag mismatch.')
    if row.get('test_timestamps', row.get('alignment_support', [None, None, None])[2]) != parts[2]['target_timestamps']:
        raise ValueError('Raw/diagnostic temporal support mismatch.')


def dataset_integrity(directory, manifest, *, temporal=False, ablation=False, expected_n=None, expected_d=None, split_seed=None, identity_prefix=None):
    """Check persisted arrays and existing split contracts, never regenerate data."""
    import numpy as np
    from src.utils.ablation_data import array_digest
    path = directory / ('process.npz' if temporal and not ablation else 'dataset.npz')
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        values = np.column_stack((data['X'], data['Y'])) if temporal and not ablation else data['values']
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Malformed persisted dataset.')
    if (expected_n is not None and len(values) != expected_n) or (expected_d is not None and values.shape[1] != expected_d):
        raise ValueError('Persisted dataset shape differs from frozen cell.')
    actual = array_digest(values) if ablation else hashlib.sha256(values.tobytes(order='C')).hexdigest()
    field = 'data_sha256' if ablation else 'dataset_sha256'
    if manifest.get(field) != actual:
        raise ValueError('Dataset content/identity hash mismatch.')
    if identity_prefix is not None and manifest.get('dataset_id') != identity_prefix + ':' + actual:
        raise ValueError('Dataset identity is not bound to its persisted content.')
    if not temporal:
        from src.utils.split import load_split
        if not (directory / 'split.npz').exists() or not (directory / 'split.json').exists():
            return False
        split = load_split(directory / 'split.npz')
        if split.n_samples != len(values):
            raise ValueError('Dataset/split sample count mismatch.')
        if split_seed is not None and split.seed != split_seed:
            raise ValueError('Persisted split seed differs from frozen allocation.')
        if not ablation and manifest.get('split_id') != (identity_prefix if identity_prefix and identity_prefix.startswith('replica-') else manifest['dataset_id']) + ':' + split.algorithm + ':' + str(split.seed):
            raise ValueError('Persisted split identity differs from frozen convention.')
        if manifest.get('sample_counts'):
            actual_counts = {'full': len(values), **{k: len(getattr(split, k)) for k in ('train', 'validation', 'test')}}
            if actual_counts != manifest['sample_counts']:
                raise ValueError('Partition counts differ from persisted split.')
    return True


def inference_artifact(path, *, seed, draws, observed, p_values, method, issues):
    """Audit persisted p-value evidence; no model or surrogate path is executed."""
    import numpy as np
    if not path.exists():
        issues.append('Missing inference evidence: ' + str(path))
        return
    data = read_json(path)
    if data['method'] != method or data['seed'] != seed or data['draws'] != draws or data['refits'] != 0:
        raise ValueError('Inference metadata differs from frozen method.')
    actual_observed = data['observed_nats']
    if not isinstance(actual_observed, list):
        actual_observed = [actual_observed]
    saved_p = data.get('p_values', [data.get('p_value')])
    statistics = np.asarray(data['surrogate_statistics_nats'], dtype=float)
    if statistics.ndim == 1:
        statistics = statistics[:, None]
    if (actual_observed != observed or saved_p != p_values or statistics.shape != (draws, len(observed))
            or not np.isfinite(statistics).all()):
        raise ValueError('Inference statistics/raw-record identity mismatch.')
    counts = np.sum(statistics >= np.asarray(observed), axis=0)
    if any(p != (1 + int(k)) / (draws + 1) for p, k in zip(p_values, counts)):
        raise ValueError('Persisted Monte Carlo p-values contradict tail counts.')


def _entropy(root, plan, ledger, issues):
    from experiments.neural_baseline import VARIANTS
    from src.metrics.aggregation import aggregate_records
    from src.metrics.confirmatory import paired_intervals
    cells = []
    all_records, terms = [], []
    for cell in plan['cells']:
        cid = cell['campaign_id']; directory = root / cid
        cm = read_json(directory / 'campaign_manifest.json') if (directory / 'campaign_manifest.json').exists() else None
        if cm is None or cm.get('status') != 'complete':
            issues.append('Missing/incomplete campaign: ' + cid)
        if cm:
            base = cm['baseline_configuration']
            if base['data'] != cell['data'] or base['neural'] != cell['baseline']['neural']:
                raise ValueError('Entropy baseline scientific policy changed.')
            if cm['replicas'] != [{k: r[k] for k in ('replica_id', 'data_seed', 'split_seed', 'training_seed')} for r in cell['replicas']]:
                raise ValueError('Entropy seed allocation changed.')
            if cm.get('purpose') != 'final' or cm['configuration']['aggregation'] != cell['campaign']['aggregation']:
                raise ValueError('Unapproved development/aggregation configuration.')
        scientific = {'data': cell['data'], 'neural': {k: v for k, v in cell['baseline']['neural'].items() if k not in ('split_seed', 'training_seed')}}
        scenario = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
        expected = [{'campaign_id': cid, 'scenario_id': scenario, **{k: r[k] for k in ('replica_id', 'data_seed', 'split_seed', 'training_seed')},
                     'variant': v, 'status': 'not_attempted', 'entropy_nats': None, 'ground_truth_nats': cell['ground_truth_nats'],
                     'signed_error_nats': None, 'runtime_seconds': None, 'fit_runtime_seconds': None, 'inference_runtime_seconds': None}
                    for r in cell['replicas'] for v in VARIANTS]
        path = directory / 'raw_results.json'
        rows, missing = canonical_rows(read_json(path) if path.exists() else [], expected, ('replica_id', 'variant'))
        if missing:
            issues.append(f'{cid}: {missing} missing estimator records')
        baseline = {(r['replica_id'], r['variant']): r for r in expected}
        for row in rows:
            valid_value(row, 'entropy_nats', 'ground_truth_nats')
            for k in ('campaign_id', 'scenario_id', 'data_seed', 'split_seed', 'training_seed', 'ground_truth_nats'):
                if row[k] != baseline[row['replica_id'], row['variant']][k]:
                    raise ValueError('Entropy raw identity/truth mismatch.')
        for replica in cell['replicas']:
            rid = replica['replica_id']; rd = directory / 'replicas' / rid
            good = [r for r in rows if r['replica_id'] == rid and r['status'] == 'complete']
            manifest = _identity_manifest(rd / 'manifest.json', {}, issues, bool(good))
            if manifest and good:
                if not dataset_integrity(rd, manifest, expected_n=cell['data']['n'], expected_d=cell['data']['d'], split_seed=replica['split_seed'], identity_prefix=rid):
                    issues.append('Missing dataset/split artifacts: ' + cid + '/' + rid)
                if manifest['resolved_configuration']['data'] != cell['data'] or manifest['seeds']['data_generation'] != replica['data_seed'] or manifest['seeds']['split'] != replica['split_seed'] or manifest['seeds']['training'] != replica['training_seed']:
                    raise ValueError('Entropy replica configuration/seeds differ.')
                neural = {**cell['baseline']['neural'], 'split_seed': replica['split_seed'], 'training_seed': replica['training_seed']}
                if manifest['resolved_configuration']['neural'] != neural:
                    raise ValueError('Entropy replica neural policy differs.')
                for r in good:
                    if r['sample_counts'] != manifest['sample_counts']:
                        raise ValueError('Entropy sample partitions differ.')
                original = rd / 'results.json'
                if original.exists():
                    source = {r['variant']: r for r in read_json(original)}
                    if len(source) != 5:
                        raise ValueError('Missing/duplicated original baseline variants.')
                    for r in good:
                        if any(r[k] != source[r['variant']][k] for k in ('entropy_nats', 'ground_truth_nats', 'signed_error_nats')):
                            raise ValueError('Campaign and replica estimates differ.')
                else:
                    issues.append('Missing baseline estimates: ' + cid + '/' + rid)
            _terms(rd / 'njee_diagnostics.json', cid, rid, 'njee', terms, issues, any(r['variant'] == 'njee' for r in good))
        aggregation = aggregate_records(rows, planned_replica_ids=[r['replica_id'] for r in cell['replicas']],
            bootstrap_draws=cell['campaign']['aggregation']['bootstrap_draws'], bootstrap_seed=cell['campaign']['aggregation']['bootstrap_seed'],
            paired_comparisons=cell['campaign']['aggregation']['paired_comparisons'])
        cells.append({'cell_id': cid, 'metadata': {'family': cell['data']['generator'], 'n': cell['data']['n'],
            'd': cell['data']['d'], 'parameter': next(iter(cell['data']['params'].values())), 'quantity': 'H'}, 'analysis': aggregation})
        all_records.extend({**r, 'cell_id': cid} for r in rows)
    return cells, all_records, terms, paired_intervals(all_records)


def _information(root, plan, issues):
    from experiments.information_validation import _empty_rows, validate_rows
    from src.metrics.information import summarize_information
    cells, records, terms = [], [], []
    for cell in plan['cells']:
        cid = cell['cell_id']; directory = root / cid
        path = directory / 'raw_results.json'
        rows, missing = canonical_rows(read_json(path) if path.exists() else [], _empty_rows(cell), ('replica_id', 'variant'))
        if missing:
            issues.append(f'{cid}: {missing} missing estimator records')
        validate_rows(cell, rows)
        for row in rows:
            valid_value(row, 'entropy_nats', 'ground_truth_nats')
        for replica in cell['allocation']:
            rid = replica['replica_id']; rd = directory / rid
            good = [r for r in rows if r['replica_id'] == rid and r['status'] == 'complete']
            manifest = _identity_manifest(rd / 'manifest.json', {'seeds': replica, 'truth': cell['truth'],
                'variable_ids': ['X','Y'] if cell['model']['family']=='bsc' else ['X','Y','Z']}, issues, bool(good))
            if manifest:
                if good and not dataset_integrity(rd, manifest, expected_n=cell['n'], expected_d=2 if cell['model']['family']=='bsc' else 3, split_seed=replica['split_seed'], identity_prefix=cid+':'+rid):
                    issues.append('Missing information dataset/split: ' + cid + '/' + rid)
                for r in good:
                    if any(r[k] != manifest[k] for k in ('dataset_id', 'split_id', 'sample_counts')):
                        raise ValueError('Information raw/manifest partition mismatch.')
            _terms(rd / 'neural_diagnostics.json', cid, rid, 'neural', terms, issues, any(r['variant'] == 'neural' for r in good))
            neural = next((r for r in good if r['variant']=='neural'), None)
            if neural and neural['is_null'] and neural.get('test_status') == 'complete':
                inference_artifact(rd / 'independence_test.json', seed=replica['permutation_seed'],
                    draws=plan['definition']['permutation_draws'], observed=[neural['entropy_nats']], p_values=[neural['p_value']],
                    method='frozen-model-heldout-exact-strata-permutation-v1', issues=issues)
        analysis = summarize_information(rows, [r['replica_id'] for r in cell['allocation']],
            bootstrap_draws=plan['definition']['bootstrap_draws'], bootstrap_seed=cell['allocation'][0]['bootstrap_seed'])
        cells.append({'cell_id': cid, 'metadata': {'n': cell['n'], **cell['model'],
            'quantity': 'MI' if cell['model']['family'] == 'bsc' else 'CMI'}, 'analysis': analysis})
        records.extend({**r, 'cell_id': cid} for r in rows)
    return cells, records, terms, []


def _transfer(root, plan, issues):
    from experiments.transfer_validation import empty_rows
    from src.metrics.transfer import summarize_transfer
    cells, records, terms = [], [], []
    for cell in plan['cells']:
        cid = cell['cell_id']; directory = root / cid
        rows = []
        refs = read_json(directory / 'references.json') if (directory / 'references.json').exists() else {}
        for job in cell['jobs']:
            if job['job_id'] in refs:
                reference_metadata(refs[job['job_id']], cell, job)
        for replica in cell['allocation']:
            rid = replica['replica_id']; rd = directory / rid
            path = rd / 'raw_results.json'
            own, missing = canonical_rows(read_json(path) if path.exists() else [], empty_rows(cell, replica), ('replica_id', 'job_id'))
            if missing:
                issues.append(f'{cid}/{rid}: {missing} missing TE jobs')
            manifest = _identity_manifest(rd / 'manifest.json', {'cell_id': cid, 'replica': replica}, issues, any(r['status'] == 'complete' for r in own))
            if manifest and any(r['status'] == 'complete' for r in own) and not dataset_integrity(rd, manifest, temporal=True, expected_n=cell['n'], expected_d=2, identity_prefix=cid+':'+rid):
                issues.append('Missing temporal process artifact: ' + cid + '/' + rid)
            for r in own:
                valid_value(r, 'te_nats', 'reference_nats')
                if r['configuration_hash'] != cell['configuration_hash'] or r['data_seed'] != replica['data']['seed'] or r['training_seed'] != replica['jobs'][r['job_id']]['seed']:
                    raise ValueError('TE allocation/configuration changed.')
                if r['status'] == 'complete':
                    ref = refs.get(r['job_id'])
                    if ref is None:
                        issues.append('Missing TE cell reference: ' + cid)
                    elif (r['reference_nats'], r['reference_status'], r['reference_error_bound_nats']) != (ref['transfer_entropy_nats'], ref['status'], ref['entropy_truncation_error_bound_nats']):
                        raise ValueError('TE row/reference mismatch.')
                    if manifest and r['dataset_id'] != manifest['dataset_id'] + ':temporal:' + r['split_id'].split(':')[-1]:
                        raise ValueError('TE dataset/temporal split identity mismatch.')
                _terms(rd / r['job_id'] / 'diagnostics.json', cid, rid, r['job_id'], terms, issues, r['status'] == 'complete')
                if r['status'] == 'complete':
                    job = next(j for j in cell['jobs'] if j['job_id'] == r['job_id'])
                    temporal_diagnostics(rd / r['job_id'] / 'diagnostics.json', r, job,
                        n=cell['n'], span=plan['definition']['common_span'], gap=plan['definition']['gap'], issues=issues)
            for direction in ('X_to_Y', 'Y_to_X'):
                primary = [r for r in own if r['direction']==direction and r['embedding']=='canonical']
                if all(r['status']=='complete' and r.get('test_status')=='complete' for r in primary):
                    inference_artifact(rd / (direction + '-surrogates.json'), seed=replica['surrogates'][direction]['seed'],
                        draws=plan['definition']['surrogate_draws'], observed=[r['te_nats'] for r in primary],
                        p_values=[r['p_value'] for r in primary], method='oracle-conditional-path-frozen-statistic-v1', issues=issues)
            rows.extend(own)
        analysis = summarize_transfer(rows, [r['replica_id'] for r in cell['allocation']], cell['jobs'],
            true_delay=cell['delay'], coupling=cell['coupling'], candidates=plan['definition']['candidates'],
            draws=plan['definition']['bootstrap_draws'], seed=cell['bootstrap']['seed'])
        cells.append({'cell_id': cid, 'metadata': {k: cell[k] for k in ('n', 'coupling', 'delay')},
                      'analysis': analysis, 'references': refs})
        records.extend(rows)
    return cells, records, terms, []


def _ablation(root, plan, issues):
    from experiments.ablation_validation import empty_rows
    from src.metrics.ablation import summarize
    expected = empty_rows(plan)
    rows, terms = [], []
    for group in plan['groups']:
        own_cells = [c for c in plan['cells'] if c['group'] == group['group_id']]
        for allocation in group['allocation']:
            rid = allocation['replica_id']; rd = root / group['group_id'] / rid
            path = rd / 'raw_results.json'
            own, missing = canonical_rows(read_json(path) if path.exists() else [],
                [r for r in expected if r['group'] == group['group_id'] and r['replica_id'] == rid], ('cell_id', 'replica_id'))
            if missing:
                issues.append(f'{group["group_id"]}/{rid}: {missing} missing ablation variants')
            manifest = _identity_manifest(rd / 'manifest.json', {'allocation': allocation, 'group': group['group_id']}, issues, any(r['status'] == 'complete' for r in own))
            if manifest and any(r['status'] == 'complete' for r in own) and not dataset_integrity(rd, manifest, temporal=group['group_id'] == 'temporal', ablation=True,
                expected_n=plan['definition']['n'], expected_d=2 if group['group_id']=='temporal' else plan['definition']['d'], split_seed=allocation['split']['seed'], identity_prefix=group['group_id']+':'+rid):
                issues.append('Missing ablation dataset/split: ' + group['group_id'] + '/' + rid)
            for row in own:
                cell = next(c for c in own_cells if c['cell_id'] == row['cell_id'])
                valid_value(row, 'estimate_nats', 'truth_nats')
                if row['configuration_hash'] != cell['configuration_hash']:
                    raise ValueError('Ablation configuration hash changed.')
                if row['status'] == 'complete':
                    if row['seeds'] != allocation['fits'][cell['cell_id']] or (manifest and (row['parent_dataset_id'], row['parent_split_id']) != (manifest['dataset_id'], manifest['parent_split_id'])):
                        raise ValueError('Ablation parent pairing/seed mismatch.')
                    if cell['kind'] == 'entropy':
                        from src.simulation.ground_truth import binary_chain_entropy
                        from src.utils.ablation_data import discretized_entropy
                        truth = float(binary_chain_entropy(plan['definition']['epsilon'], plan['definition']['d'])) if cell['group'] == 'binary_chain' else discretized_entropy(plan['definition']['d'], cell['states'])
                        if row['truth_nats'] != truth:
                            raise ValueError('Ablation analytic ground truth changed.')
                    else:
                        ref = reference_metadata(row['reference'], plan['definition']['temporal'], cell)
                        if row['truth_nats'] != ref['transfer_entropy_nats']:
                            raise ValueError('Ablation TE reference differs.')
                        temporal_diagnostics(rd / cell['cell_id'] / 'diagnostics.json', row, cell,
                            n=plan['definition']['n'], span=plan['definition']['temporal']['common_span'],
                            gap=plan['definition']['temporal']['gap'], issues=issues)
                _terms(rd / cell['cell_id'] / 'diagnostics.json', cell['cell_id'], rid, cell['name'], terms, issues, row['status'] == 'complete')
            rows.extend(own)
    analysis = summarize(rows, plan)
    cells = [{'cell_id': c['cell_id'], 'metadata': {k: c[k] for k in ('group', 'factor', 'name', 'states', 'train_fraction', 'variant', 'kind')},
              'analysis': {'summaries': [s for s in analysis['summaries'] if s['cell_id'] == c['cell_id']]}} for c in plan['cells']]
    return cells, rows, terms, analysis


def validate_source(experiment, root, *, ledger_path=None):
    """All corruption is fatal in BOTH modes; incompleteness is an eligibility issue."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Symlink scientific input root is unsupported.')
    root = root.resolve()
    before = inventory(root)
    plan = frozen_plan(experiment)
    issues = []
    if any(name.endswith('.tmp') for name in before):
        issues.append('Unfinished temporary source artifacts are present.')
    if experiment == 'entropy':
        ledger_file = root / 'control' / 'ledger.json' if ledger_path is None else Path(ledger_path).resolve()
        if not ledger_file.is_relative_to(root):
            raise ValueError('Reporting ledger must be archived inside the input root.')
        manifest = read_json(ledger_file) if ledger_file.exists() else {}
        ledger = manifest.get('campaigns', [])
        expected_ids = {c['campaign_id'] for c in plan['cells']}
        keys = [e['campaign_id'] for e in ledger]
        if len(keys) != len(set(keys)) or set(keys) - expected_ids:
            raise ValueError('Duplicate/unplanned campaign ledger entry.')
        if set(keys) != expected_ids or any(e['status'] != 'complete' for e in ledger):
            issues.append('Incomplete controller ledger.')
    else:
        saved = read_json(root / 'protocol.json')
        if {k: v for k, v in saved.items() if k != 'output_root'} != {k: v for k, v in plan.items() if k != 'output_root'}:
            raise ValueError('Scientific output protocol/configuration hash mismatch.')
        manifest = read_json(root / 'manifest.json') if (root / 'manifest.json').exists() else {}
        ledger = read_json(root / 'execution_ledger.json') if (root / 'execution_ledger.json').exists() else []
        expected_ids = {(g['group_id'], r['replica_id']) for g in plan['groups'] for r in g['allocation']} if experiment == 'ablation' else {c['cell_id'] for c in plan['cells']}
        keys = [(e['group'], e['replica_id']) for e in ledger] if experiment == 'ablation' else [e['cell_id'] for e in ledger]
        if len(keys) != len(set(keys)) or set(keys) - expected_ids:
            raise ValueError('Duplicate/unplanned execution ledger entry.')
        if set(keys) != expected_ids or any(e['status'] != 'complete' for e in ledger):
            issues.append('Incomplete execution ledger.')
    if manifest and manifest.get('protocol_hash') != plan['protocol_hash']:
        raise ValueError('Execution manifest protocol hash mismatch.')
    if manifest.get('status') != 'complete':
        issues.append('Execution is not complete.')
    if not manifest.get('environment'):
        issues.append('Missing execution environment provenance.')
    git = manifest.get('git', {'commit': manifest.get('git_commit'), 'dirty': None})
    if not git.get('commit') or (experiment != 'entropy' and git.get('dirty') is not False):
        issues.append('Missing clean execution-code provenance.')
    adapter = {'entropy': _entropy, 'information': _information, 'transfer': _transfer, 'ablation': _ablation}[experiment]
    result = adapter(root, plan, ledger, issues) if experiment == 'entropy' else adapter(root, plan, issues)
    cells, rows, terms, extra = result
    if experiment == 'entropy':
        allowed = {c['campaign_id'] + '/raw_results.json' for c in plan['cells']}
    elif experiment == 'information':
        allowed = {c['cell_id'] + '/raw_results.json' for c in plan['cells']}
    elif experiment == 'transfer':
        allowed = {c['cell_id'] + '/raw_results.json' for c in plan['cells']}
        allowed.update(c['cell_id'] + '/' + r['replica_id'] + '/raw_results.json' for c in plan['cells'] for r in c['allocation'])
    else:
        allowed = {g['group_id'] + '/' + r['replica_id'] + '/raw_results.json' for g in plan['groups'] for r in g['allocation']}
    if {name for name in before if name.endswith('raw_results.json')} - allowed:
        raise ValueError('Unplanned scientific raw-result files in input root.')
    incomplete = sum(r['status'] != 'complete' for r in rows)
    if incomplete:
        issues.append(f'{incomplete} planned records are unsuccessful/unattempted.')
    if experiment == 'information' and any(r.get('test_status') != 'complete' for r in rows if r.get('is_null') and r['variant'] == 'neural'):
        issues.append('Incomplete null-information tests.')
    if experiment == 'transfer' and any(r.get('test_status') != 'complete' for r in rows if r['embedding'] == 'canonical'):
        issues.append('Incomplete fixed-family surrogate tests.')
    if experiment == 'transfer' and any(not r['family']['complete'] for c in cells for r in c['analysis']['replica_metrics']):
        issues.append('Incomplete eight-test family decisions.')
    after = inventory(root)
    if before != after:
        raise ValueError('Scientific inputs changed during validation.')
    return {'experiment': experiment, 'protocol_hash': plan['protocol_hash'], 'source_root': str(root),
            'source_inventory': before, 'source_inventory_hash': digest(before), 'source_manifest': manifest,
            'source_ledger': ledger, 'cells': cells, 'raw_records': rows, 'terms': terms, 'extra_analysis': extra,
            'issues': sorted(set(issues)), 'complete': not issues, 'unit': 'nats', 'log_base': 'e',
            'planned_cells': len(plan['cells']), 'observed_cells': len({r['cell_id'] for r in rows if r['status'] != 'not_attempted'}),
            'planned_records': len(rows), 'complete_records': len(rows) - incomplete,
            'artificial_fixture': bool(manifest.get('artificial_fixture', False))}
