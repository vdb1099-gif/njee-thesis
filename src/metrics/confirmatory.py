import math
import statistics

from scipy.stats import t

from experiments.final_protocol import PRIMARY, REPLICAS, cells
from experiments.neural_baseline import VARIANTS

FAMILY_ALPHA = .05
COMPARISONS = 16
INDIVIDUAL_ALPHA = FAMILY_ALPHA / COMPARISONS


def paired_intervals(records):
    """Full-precision D=abs(NJEE error)-abs(classical error); require all 30.

    Independent replica differences, finite variance and normality (or an adequate
    t approximation) are assumptions. Bonferroni requires no independence between
    comparisons. Incomplete comparisons remain in the 16-comparison family.
    """
    primary = {e['campaign_id']: e for e in cells() if e['cell'] in PRIMARY}
    index = {}
    scenarios = {}
    for row in records:
        cid = row['campaign_id']
        if cid not in primary:
            continue
        key = cid, row['replica_id'], row['variant']
        if row['replica_id'] not in REPLICAS or row['variant'] not in VARIANTS:
            raise ValueError('Unplanned confirmatory identity.')
        if key in index:
            raise ValueError('Duplicate confirmatory identity.')
        if 'scenario_id' in row:
            scenarios.setdefault(cid, set()).add(row['scenario_id'])
            if len(scenarios[cid]) > 1:
                raise ValueError('Mixed scientific scenarios in one primary cell.')
        index[key] = row
    results = []
    for cid, cell in primary.items():
        for variant in VARIANTS[1:]:
            differences, missing = [], []
            for rid in REPLICAS:
                pair = [index.get((cid, rid, method)) for method in ('njee', variant)]
                if any(r is None or r['status'] != 'complete' for r in pair):
                    missing.append(rid)
                    continue
                errors = []
                for row in pair:
                    estimate, truth, error = (row[k] for k in ('entropy_nats', 'ground_truth_nats', 'signed_error_nats'))
                    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (estimate, truth, error)):
                        raise ValueError('Nonfinite successful confirmatory record.')
                    if not math.isclose(error, estimate-truth, rel_tol=0, abs_tol=1e-12):
                        raise ValueError('Inconsistent signed error.')
                    errors.append(error)
                if pair[0]['ground_truth_nats'] != pair[1]['ground_truth_nats']:
                    raise ValueError('Paired methods have different ground truths.')
                for field in ('data_seed', 'split_seed', 'training_seed', 'sample_counts'):
                    if field in pair[0] or field in pair[1]:
                        if pair[0].get(field) != pair[1].get(field):
                            raise ValueError(f'Pairing differs: {field}')
                differences.append({'replica_id': rid, 'difference_nats': abs(errors[0])-abs(errors[1])})
            complete = len(differences) == 30
            result = {'cell': cell['cell'], 'campaign_id': cid, 'left': 'njee', 'right': variant,
                      'method': 'bonferroni_two_sided_paired_t',
                      'quantity': 'absolute_error_left_minus_right', 'family_alpha': FAMILY_ALPHA,
                      'family_size': COMPARISONS, 'individual_alpha': INDIVIDUAL_ALPHA,
                      'confidence_level': 1-INDIVIDUAL_ALPHA, 'n_planned': 30, 'n_pairs': len(differences),
                      'status': 'complete' if complete else 'incomplete', 'missing_replica_ids': missing,
                      'df': 29 if complete else None, 'mean_difference_nats': None,
                      'sample_std_nats': None, 'standard_error_nats': None, 'critical_value': None,
                      'lower_nats': None, 'upper_nats': None, 'zero_observed_variance': None,
                      'differences': differences}
            if complete:
                values = [r['difference_nats'] for r in differences]
                mean, sd = statistics.mean(values), statistics.stdev(values)
                se, critical = sd/math.sqrt(30), float(t.ppf(1-INDIVIDUAL_ALPHA/2, 29))
                result.update(mean_difference_nats=mean, sample_std_nats=sd, standard_error_nats=se,
                              critical_value=critical, lower_nats=mean-critical*se,
                              upper_nats=mean+critical*se, zero_observed_variance=sd == 0)
                if any(not math.isfinite(result[k]) for k in ('mean_difference_nats','lower_nats','upper_nats')):
                    raise ValueError('Nonfinite confirmatory arithmetic.')
            results.append(result)
    return results
