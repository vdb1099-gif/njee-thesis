import math

import numpy as np
from scipy.stats import beta

from src.estimators.classical import PlugInEstimator, MillerMadowEstimator
from src.metrics.aggregation import aggregate_records, bootstrap_seed_for, _metrics, _intervals
from src.utils.reproducibility import validate_seed


def classical_information(values, variable_ids, *, conditional, method):
    """Use the SAME observations for all entropy terms, including observed-K MM.

    MI: H(X)+H(Y)-H(X,Y); CMI: H(X,Z)+H(Y,Z)-H(Z)-H(X,Y,Z).
    Components are returned as H(X|Z), H(X|Y,Z) (Z empty for MI).
    """
    if method not in ('plug_in', 'miller_madow'):
        raise ValueError('Unsupported classical method.')
    estimator = PlugInEstimator() if method == 'plug_in' else MillerMadowEstimator()
    def h(order):
        return estimator.estimate(values[:, [variable_ids.index(v) for v in order]]) if order else 0.
    z = ('Z',) if conditional else ()
    a = h(('X',)+z)-h(z)
    b = h(('X', 'Y')+z)-h(('Y',)+z)
    return {'information_nats': a-b, 'first_entropy_nats': a, 'second_entropy_nats': b}


def heldout_permutation_test(estimator, test, *, conditional, seed, draws=999, alpha=.05,
                             observed=None, cancelled=lambda: False):
    """Frozen TRAIN/VALIDATION-selected statistic, evaluated only on TEST.

    Under H0, Y is exchangeable globally (MI) or within exact Z strata (CMI).
    Permute Y in test only; do NOT retrain, select or change any setting. T is
    raw signed information, upper-tail test, ties count as exceedances. Monte
    Carlo p=(1+#(T_perm>=T_obs))/(B+1), reject at p<=alpha. This is conditional
    on training/validation and the test X,Z and Y multisets. Degenerate strata
    yield p=1. No inference from merely T>0. Not a general continuous-Z test.
    """
    from dataclasses import replace
    if test.role != 'test':
        raise ValueError('Randomization requires held-out test role.')
    if type(draws) is not int or draws < 1 or type(alpha) not in (float, int) or not 0 < alpha < 1:
        raise ValueError('Invalid randomization settings.')
    validate_seed(seed)
    if observed is None:
        observed = estimator.estimate(test).information_nats
    if not math.isfinite(observed):
        raise ValueError('Nonfinite observed statistic.')
    y = test.variable_ids.index('Y')
    z = test.values[:, test.variable_ids.index('Z')] if conditional else np.zeros(len(test.values))
    groups = [np.flatnonzero(z == label) for label in np.unique(z)]
    rng = np.random.default_rng(seed)
    values, exceed = [], 0
    for _ in range(draws):
        if cancelled():
            raise KeyboardInterrupt('Cancellation requested during randomization.')
        shuffled = test.values.copy()
        for indices in groups:
            shuffled[indices, y] = test.values[rng.permutation(indices), y]
        value = estimator.estimate(replace(test, values=shuffled)).information_nats
        if not math.isfinite(value):
            raise RuntimeError('Nonfinite surrogate statistic.')
        values.append(value)
        exceed += value >= observed
    p = (1+exceed)/(draws+1)
    return {'method': 'frozen-model-heldout-exact-strata-permutation-v1', 'seed': seed,
            'draws': draws, 'alpha': alpha, 'p_value': p, 'reject': bool(p <= alpha),
            'observed_nats': observed, 'surrogate_statistics_nats': values,
            'ties': 'greater_or_equal', 'refits': 0}


def binomial_interval(successes, trials):
    """Pointwise exact 95% Clopper-Pearson interval, independent replicas as units."""
    if type(trials) is not int or type(successes) is not int or not 0 <= successes <= trials:
        raise ValueError('Invalid binomial counts.')
    if not trials:
        return {'lower': None, 'upper': None, 'confidence_level': .95, 'method': 'Clopper-Pearson'}
    return {'lower': float(beta.ppf(.025, successes, trials-successes+1)) if successes else 0.,
            'upper': float(beta.ppf(.975, successes+1, trials-successes)) if successes < trials else 1.,
            'confidence_level': .95, 'method': 'Clopper-Pearson'}


def error_correlation(pairs, *, draws, seed):
    """Pearson correlation of component ERRORS; joint replica bootstrap, no row bootstrap."""
    validate_seed(seed)
    if type(draws) is not int or draws < 1:
        raise ValueError('draws must be a positive integer.')
    values = np.asarray(pairs, dtype=float)
    if len(values) and (values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all()):
        raise ValueError('Expected finite paired component errors.')
    def corr(v):
        if len(v) < 3 or np.ptp(v[:, 0]) == 0 or np.ptp(v[:, 1]) == 0:
            return None
        return float(np.corrcoef(v.T)[0, 1])
    point = corr(values)
    samples = []
    if point is not None:
        rng = np.random.default_rng(seed)
        for _ in range(draws):
            r = corr(values[rng.integers(0, len(values), len(values))])
            if r is not None:
                samples.append(r)
    return {'pearson_r': point, 'n_pairs': len(values), 'bootstrap_draws': draws,
            'valid_bootstrap_draws': len(samples), 'seed': seed,
            'lower': float(np.quantile(samples, .025)) if len(samples) == draws else None,
            'upper': float(np.quantile(samples, .975)) if len(samples) == draws else None,
            'reason': 'undefined_or_degenerate_resamples' if point is None or len(samples) != draws else None,
            'note': 'Correlation is descriptive; zero correlation does not establish independence.'}


def summarize_information(rows, replica_ids, *, bootstrap_draws=10000, bootstrap_seed=271816):
    """One-cell complete ledger, including all failures and unattempted placeholders."""
    variants = ('neural', 'plug_in_full', 'miller_madow_full', 'plug_in_train', 'miller_madow_train')
    expected = {(i, v) for i in replica_ids for v in variants}
    if {(r['replica_id'], r['variant']) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError('Missing, duplicate or unexpected planned records.')
    for r in rows:
        if r['status'] == 'complete':
            values = [r[k] for k in ('entropy_nats', 'ground_truth_nats', 'signed_error_nats',
                                     'first_entropy_nats', 'second_entropy_nats',
                                     'first_truth_nats', 'second_truth_nats')]
            if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
                raise ValueError('Nonfinite complete information record.')
            if (r['entropy_nats'] != r['first_entropy_nats']-r['second_entropy_nats']
                    or ('information_nats' in r and r['information_nats'] != r['entropy_nats'])
                    or r['signed_error_nats'] != r['entropy_nats']-r['ground_truth_nats']):
                raise ValueError('Inconsistent signed component arithmetic.')
    report = aggregate_records(rows, planned_replica_ids=replica_ids, bootstrap_draws=bootstrap_draws,
                               bootstrap_seed=bootstrap_seed,
                               paired_comparisons=[('neural', v) for v in variants[1:]])
    for summary in report['summaries']:
        valid = [r for r in rows if r['variant'] == summary['variant'] and r['status'] == 'complete']
        negative = sum(r['entropy_nats'] < 0 for r in valid)
        summary['negative_frequency'] = negative/len(valid) if valid else None
        summary['negative_count'] = negative
        summary['negative_interval'] = binomial_interval(negative, len(valid))
        for timing in ('independence_runtime_seconds', 'replica_wall_runtime_seconds'):
            times = [r.get(timing) for r in valid if type(r.get(timing)) in (int,float)
                     and math.isfinite(r[timing]) and r[timing] >= 0]
            summary['mean_'+timing] = math.fsum(times)/len(times) if times else None
            summary['n_'+timing] = len(times)
        attempted_times = [r.get('replica_wall_runtime_seconds') for r in rows
                           if r['variant']==summary['variant'] and r['status']!='not_attempted'
                           and type(r.get('replica_wall_runtime_seconds')) in (int,float)
                           and math.isfinite(r['replica_wall_runtime_seconds'])]
        summary['attempted_wall_runtime_seconds'] = math.fsum(attempted_times) if attempted_times else None
        summary['n_attempted_wall_timings'] = len(attempted_times)
        clipped = np.maximum([r['entropy_nats'] for r in valid], 0.)
        truths = np.array([r['ground_truth_nats'] for r in valid])
        clip_seed = bootstrap_seed_for(bootstrap_seed, [summary['variant'], 'secondary-clipped'])
        summary['secondary_clipped_only'] = {
            'metrics': _metrics(clipped, truths) if valid else None,
            'bootstrap': _intervals(clipped, truths, draws=bootstrap_draws, seed=clip_seed),
            'note': 'Secondary display only; raw estimates, primary errors and decisions stay signed.'}
        pairs = [(r['first_entropy_nats']-r['first_truth_nats'],
                  r['second_entropy_nats']-r['second_truth_nats']) for r in valid]
        summary['component_error_correlation'] = error_correlation(
            pairs, draws=bootstrap_draws,
            seed=bootstrap_seed_for(bootstrap_seed, [summary['variant'], 'component-errors']))
    nulls = [r for r in rows if r['variant'] == 'neural' and r['is_null']]
    tested = [r for r in nulls if r['status']=='complete' and r.get('test_status') == 'complete']
    rejects = sum(r['reject'] for r in tested)
    for r in tested:
        if (type(r['p_value']) not in (float,int) or not math.isfinite(r['p_value'])
                or not 0 <= r['p_value'] <= 1 or type(r['reject']) is not bool
                or r['reject'] != (r['p_value'] <= .05)):
            raise ValueError('Inconsistent independence decision.')
    report['false_positive_analysis'] = {
        'n_planned_null': len(nulls), 'n_tested': len(tested), 'n_untested': len(nulls)-len(tested),
        'n_rejected': rejects, 'rate': rejects/len(tested) if tested else None,
        'interval': binomial_interval(rejects, len(tested)),
        'planned_rate_bounds': [rejects/len(nulls), (rejects+len(nulls)-len(tested))/len(nulls)] if nulls else None,
        'decision': 'neural held-out permutation p<=0.05; not raw information>0',
        'classical_testing': 'not performed; classical comparisons are estimation-only',
        'inference': 'pointwise; no familywise significance claim across cells'}
    report['inference'] = 'Exploratory pointwise 95% intervals; no confirmatory superiority claims.'
    return report
