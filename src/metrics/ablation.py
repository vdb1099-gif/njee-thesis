"""Exploratory ablation summaries; independent datasets/processes are the units."""
import math

import numpy as np

from src.metrics.aggregation import _metrics, _intervals, bootstrap_seed_for


def summarize(rows, plan):
    replicas = [r['replica_id'] for r in plan['groups'][0]['allocation']]
    expected = {(c['cell_id'], r) for c in plan['cells'] for r in replicas}
    if len(rows) != len(expected) or {(r['cell_id'], r['replica_id']) for r in rows} != expected:
        raise ValueError('Missing/duplicate/unplanned ablation records.')
    valid = {}
    for row in rows:
        if row['status'] not in ('complete', 'failed', 'interrupted', 'running', 'not_attempted'):
            raise ValueError('Unknown status.')
        if row['status'] == 'complete':
            if (not all(type(row[k]) in (int, float) and math.isfinite(row[k])
                        for k in ('estimate_nats', 'truth_nats', 'signed_error_nats'))
                    or row['signed_error_nats'] != row['estimate_nats'] - row['truth_nats']):
                raise ValueError('Invalid successful estimate/error.')
            valid[row['cell_id'], row['replica_id']] = row
    draws, seed = plan['definition']['bootstrap_draws'], plan['definition']['bootstrap_seed']
    summaries = []
    for cell in plan['cells']:
        all_rows = [r for r in rows if r['cell_id'] == cell['cell_id']]
        good = [r for r in all_rows if r['status'] == 'complete']
        estimates = np.array([r['estimate_nats'] for r in good])
        truths = np.array([r['truth_nats'] for r in good])
        summaries.append({'cell_id': cell['cell_id'], 'factor': cell['factor'],
                          'n_planned': len(replicas), 'n_complete': len(good),
                          'n_not_attempted': sum(r['status'] == 'not_attempted' for r in all_rows),
                          'n_failed': sum(r['status'] in ('failed', 'interrupted', 'running') for r in all_rows),
                          'metrics': _metrics(estimates, truths) if good else None,
                          'bootstrap': _intervals(estimates, truths, draws=draws, seed=bootstrap_seed_for(seed, cell['cell_id'])),
                          'mean_runtime_seconds': float(np.mean([r['runtime_seconds'] for r in good])) if good else None,
                          'computational_cost': [{k: r.get(k) for k in
                              ('replica_id', 'optimizer_updates', 'term_batch_evaluations', 'parameter_count', 'n_train')}
                                                 for r in good]})
    pairs = []
    for comparison in plan['comparisons']:
        ids = [r for r in replicas if (comparison['left'], r) in valid and (comparison['right'], r) in valid]
        differences = []
        for rid in ids:
            left, right = (valid[c, rid] for c in (comparison['left'], comparison['right']))
            if (left['parent_dataset_id'], left['parent_split_id']) != (right['parent_dataset_id'], right['parent_split_id']):
                raise ValueError('Pairing provenance differs.')
            if comparison['same_estimand'] and left['truth_nats'] != right['truth_nats']:
                raise ValueError('Paired common-quantity ground truths differ.')
            differences.append({'replica_id': rid,
                'signed_error_difference_nats': left['signed_error_nats'] - right['signed_error_nats'],
                'absolute_error_difference_nats': abs(left['signed_error_nats']) - abs(right['signed_error_nats']),
                'runtime_difference_seconds': left['runtime_seconds'] - right['runtime_seconds']})
        intervals = {}
        for name in ('signed_error_difference_nats', 'absolute_error_difference_nats', 'runtime_difference_seconds'):
            values = np.array([v[name] for v in differences])
            interval = _intervals(values, np.zeros(len(values)), draws=draws,
                    seed=bootstrap_seed_for(seed, [comparison['left'], comparison['right'], name]))['mean_estimate_nats']
            interval = {**interval, 'statistic': 'mean_' + name,
                        'measurement_unit': 'seconds' if name.endswith('seconds') else 'nats',
                        'sampling_unit': 'paired independent dataset/process replica'}
            intervals[name] = {'mean': float(np.mean(values)) if ids else None,
                               'interval': interval}
        pairs.append({**comparison, 'n_planned': len(replicas), 'n_pairs': len(ids),
                      'n_incomplete_pairs': len(replicas) - len(ids), 'differences': differences, 'statistics': intervals})
    ordering = []
    for group in ('independent', 'binary_chain'):
        cells = [c for c in plan['cells'] if c['group'] == group and c['factor'] in ('reference', 'ordering')]
        if len(cells) < 2:
            continue
        for rid in replicas:
            selected = [valid.get((c['cell_id'], rid)) for c in cells]
            complete = all(r is not None for r in selected)
            values = [r['estimate_nats'] for r in selected] if complete else []
            ordering.append({'group': group, 'replica_id': rid, 'n_orders_planned': len(cells),
                'complete': complete, 'within_dataset_order_sd_nats': float(np.std(values, ddof=1)) if complete else None,
                'within_dataset_order_range_nats': float(np.ptp(values)) if complete else None})
    return {'summaries': summaries, 'paired_comparisons': pairs, 'ordering_variability': ordering,
            'inference': 'exploratory pointwise 95% replica bootstrap; no multiplicity-adjusted confirmatory claims',
            'limitations': 'Success-only intervals can be selection-biased. Orders/terms/windows are not independent replicas. '
                           'Order spread also includes independent training initialization variation.'}
