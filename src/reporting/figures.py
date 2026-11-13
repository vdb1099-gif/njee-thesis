"""Deterministic static Matplotlib recipes; only validated table values are plotted."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class Recipe:
    name: str
    table: str
    x: str
    y: str
    ylabel: str
    facets: tuple = ()
    series: str = 'variant'
    interval: str | None = None
    filters: tuple = ()


def recipes(experiment):
    """This inventory is available without any experimental observations."""
    result = []
    def add(name, table, x, y, ylabel, facets=(), series='variant', interval=None, filters=()):
        result.append(Recipe(name, table, x, y, ylabel, facets, series, interval, filters))
    metrics = [('bias_nats', 'Bias (nats)'), ('rmse_nats', 'RMSE (nats)')]
    if experiment == 'entropy':
        for metric, label in metrics:
            add(metric + '-sample-size', 'summaries', 'n', metric, label,
                ('family', 'parameter', 'd'), interval=metric)
        add('rmse-dimensionality', 'summaries', 'd', 'rmse_nats', 'RMSE (nats)',
            ('family', 'parameter', 'n'), interval='rmse_nats')
        add('runtime-sample-size', 'summaries', 'n', 'mean_runtime_seconds', 'Mean runtime (seconds)', ('family', 'parameter', 'd'))
        add('generator-family-comparison', 'summaries', 'scenario', 'rmse_nats', 'RMSE (nats)', ('n', 'd'), interval='rmse_nats')
        add('confirmatory-adjusted', 'confirmatory', 'right', 'mean_difference_nats',
            'Paired absolute-error difference: NJEE - classical (nats)', ('cell',), 'left', 'confirmatory')
    elif experiment == 'information':
        facets = ('quantity', 'family', 'epsilon_x', 'epsilon_y')
        for metric, label in [('mean_estimate_nats', 'Signed information estimate (nats)'), *metrics,
                              ('mae_nats', 'MAE (nats)'), ('sample_std_nats', 'Sample SD (nats)')]:
            add(metric + '-sample-size', 'summaries', 'n', metric, label, facets, interval=metric)
        add('negative-frequency', 'accuracy', 'n', 'rate', 'Negative-estimate frequency', facets,
            interval='direct', filters=(('metric', 'negative_frequency'),))
        add('component-errors', 'components', 'first_error_nats', 'second_error_nats',
            'Second component entropy error (nats)', ('cell_id',), 'variant')
        add('component-error-correlation', 'correlations', 'n', 'pearson_r', 'Component-error Pearson r', facets, interval='direct')
        add('null-false-positive-rate', 'accuracy', 'n', 'rate', 'False-positive rate', facets,
            interval='direct', filters=(('metric', 'false_positive_rate'),))
        add('paired-absolute-error', 'paired_comparisons', 'n', 'mean', 'Paired absolute-error difference (nats)',
            facets, 'right', 'direct', (('metric', 'absolute_error_difference_nats'),))
        add('runtime-sample-size', 'summaries', 'n', 'mean_runtime_seconds', 'Mean runtime (seconds)', facets)
    elif experiment == 'transfer':
        for x, facets in [('n', ('coupling', 'delay', 'embedding', 'lag')),
                          ('coupling', ('n', 'delay', 'embedding', 'lag')),
                          ('delay', ('n', 'coupling', 'embedding', 'lag'))]:
            add('directional-te-' + x, 'summaries', x, 'mean_estimate_nats', 'Signed TE (nats)', facets,
                'direction', 'mean_estimate_nats')
            add('direction-difference-' + x, 'paired_comparisons', x, 'mean', 'Oracle-delay TE(X to Y) - TE(Y to X) (nats)',
                tuple(f for f in facets if f not in ('embedding', 'lag')), 'left', 'direct')
            for metric, label in [('direction_correct', 'Direction-identification accuracy'),
                                  ('lag_correct', 'Validation-selected lag accuracy'), ('false_link', 'Structural-null false-link frequency')]:
                add(metric + '-' + x, 'accuracy', x, 'rate', label,
                    tuple(f for f in facets if f not in ('embedding', 'lag')), 'metric', 'direct', (('metric', metric),))
        add('embedding-sensitivity', 'summaries', 'embedding', 'mean_estimate_nats', 'Signed finite-history TE (nats)',
            ('n', 'coupling', 'delay', 'lag'), 'direction', 'mean_estimate_nats')
        add('te-rmse', 'summaries', 'n', 'rmse_nats', 'Finite-history TE RMSE (nats)',
            ('coupling', 'delay', 'embedding', 'lag'), 'direction', 'rmse_nats')
        add('runtime', 'summaries', 'n', 'mean_runtime_seconds', 'Mean job runtime (seconds)',
            ('coupling', 'delay', 'embedding', 'lag'), 'direction')
        add('failure-counts', 'summaries', 'n', 'n_failed', 'Failed/interrupted/running jobs',
            ('coupling', 'delay', 'embedding', 'lag'), 'direction')
    elif experiment == 'ablation':
        for factor in ('architecture', 'sharing', 'training_size', 'early_stopping', 'discretization', 'embedding'):
            add(factor + '-rmse', 'summaries', 'name', 'rmse_nats', 'RMSE relative to cell-specific truth (nats)',
                ('group',), 'factor', 'rmse_nats', (('factor', (factor, 'reference')),))
        add('ordering-variability', 'ordering', 'replica_id', 'within_dataset_order_sd_nats',
            'Within-dataset order SD, including initialization (nats)', ('group',), 'group')
        add('paired-error-differences', 'paired_comparisons', 'left', 'mean', 'Paired absolute-error difference (nats)',
            ('group', 'same_estimand'), 'right', 'direct', (('metric', 'absolute_error_difference_nats'),))
        for value, label in [('parameter_count', 'Stored parameters'), ('optimizer_updates', 'Optimizer updates'),
                             ('term_batch_evaluations', 'Training term-batch evaluations'), ('runtime_seconds', 'Observed runtime (seconds)')]:
            add(value, 'costs', 'name', value, label, ('group',), 'factor')
    else:
        raise ValueError('Unknown experiment.')
    if experiment in ('entropy', 'information'):
        add('conditional-term-validation', 'terms', 'position', 'best_validation_ce_nats',
            'Best validation CE (nats); individual fitted terms', ('cell_id', 'component'), 'variant')
    return result


LABELS = {'n': 'Sample / series length N', 'd': 'Dimension d', 'coupling': 'Configured coupling strength',
          'scenario': 'Generator family and frozen parameter',
          'delay': 'Physical delay (time steps)', 'first_error_nats': 'First component entropy error (nats)',
          'position': 'Chain-rule term position', 'replica_id': 'Independent dataset replica',
          'name': 'Frozen ablation configuration', 'embedding': 'Temporal embedding',
          'left': 'Left configuration (paired contrast)', 'right': 'Classical estimator / sample partition'}
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#333333')
STYLES = ('-', '--', '-.', ':')
MARKERS = ('o', 's', '^', 'D', 'v', 'P')
METHODS = {'njee': 'NJEE', 'neural': 'Neural', 'plug_in_full': 'Plug-in (full data)',
           'plug_in_train': 'Plug-in (TRAIN only)', 'miller_madow_full': 'Miller–Madow (full data)',
           'miller_madow_train': 'Miller–Madow (TRAIN only)', 'X_to_Y': 'X → Y', 'Y_to_X': 'Y → X'}


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def generate_figures(directory, tables, experiment, *, mode, artificial=False, dpi=300):
    """Each plotted point maps to its exact CSV row; missing values are recorded."""
    directory = Path(directory)
    inventory = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
        'axes.titlesize': 9, 'legend.fontsize': 7, 'svg.fonttype': 'none',
        'svg.hashsalt': 'njee-thesis-report-v1', 'savefig.dpi': dpi, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .2}):
        for recipe in recipes(experiment):
            source = tables[recipe.table]
            eligible, omitted = [], []
            for i, row in enumerate(source):
                if any(row.get(k) not in v if isinstance(v, tuple) else row.get(k) != v for k, v in recipe.filters):
                    continue
                if row.get(recipe.x) is None or not _finite(row.get(recipe.y)):
                    omitted.append(i)
                    continue
                eligible.append((i, row))
            facets = {}
            for i, row in eligible:
                key = json.dumps({k: row.get(k) for k in recipe.facets}, sort_keys=True)
                facets.setdefault(key, []).append((i, row))
            if not facets:
                inventory.append({'recipe': recipe.name, 'status': 'unavailable', 'reason': 'no valid eligible observations',
                                  'source_table': recipe.table, 'omitted_rows': omitted})
            for facet, data in sorted(facets.items()):
                stem = recipe.name + '-' + hashlib.sha256(facet.encode()).hexdigest()[:10]
                fig, ax = plt.subplots(figsize=(6.4, 4.0), layout='constrained')
                categorical = not all(_finite(row[recipe.x]) for _, row in data)
                labels = sorted({str(r[recipe.x]) for _, r in data}) if categorical else []
                groups = sorted({str(r.get(recipe.series, 'observations')) for _, r in data})
                point_map, interval_count, reference_curves = [], 0, set()
                for gi, group in enumerate(groups):
                    points = [(i, r) for i, r in data if str(r.get(recipe.series, 'observations')) == group]
                    points.sort(key=lambda p: str(p[1][recipe.x]) if categorical else p[1][recipe.x])
                    xs = [labels.index(str(r[recipe.x])) if categorical else r[recipe.x] for _, r in points]
                    ys = [r[recipe.y] for _, r in points]
                    # Never connect multiple observations at the same x as if a trajectory.
                    repeated = len(set(xs)) != len(xs)
                    ax.plot(xs, ys, marker=MARKERS[gi % len(MARKERS)], markersize=4,
                            linestyle='none' if categorical or repeated or recipe.table in ('components', 'terms', 'costs', 'ordering') else STYLES[gi % len(STYLES)],
                            color=COLORS[gi % len(COLORS)], alpha=.65 if repeated else 1,
                            label=METHODS.get(group, group), linewidth=1.1)
                    for x, y, (index, row) in zip(xs, ys, points):
                        interval = None
                        if recipe.interval == 'confirmatory':
                            interval = (row.get('lower_nats'), row.get('upper_nats'))
                        elif recipe.interval == 'direct':
                            interval = (row.get('lower'), row.get('upper'))
                        elif recipe.interval:
                            match = [r for r in tables['uncertainty'] if r['cell_id'] == row['cell_id']
                                     and r.get('variant') == row.get('variant') and r['metric'] == recipe.interval]
                            if len(match) == 1:
                                interval = (match[0]['lower'], match[0]['upper'])
                        if interval and all(_finite(v) for v in interval):
                            # Percentile bootstrap intervals need not contain their point estimate.
                            ax.vlines(x, interval[0], interval[1], color=COLORS[gi % len(COLORS)], linewidth=1)
                            interval_count += 1
                        point_map.append({'csv_row_index_zero_based': index, 'x': row[recipe.x], 'y': y,
                                          'series': group, 'interval': interval, 'reference_nats': row.get('reference_nats'),
                                          'n_complete_or_evaluated': row.get('n_complete', row.get('n_evaluated', row.get('n_pairs'))),
                                          'n_planned': row.get('n_planned')})
                    if recipe.y == 'mean_estimate_nats':
                        references = [(x, r.get('reference_nats')) for x, (_, r) in zip(xs, points) if _finite(r.get('reference_nats'))]
                        if references and tuple(references) not in reference_curves:
                            reference_curves.add(tuple(references))
                            kinds = {kind for _, r in points for kind in r.get('reference_kinds', [])}
                            qualifier = 'analytic zero' if kinds == {'analytic_zero'} else 'numerical stationary' if kinds == {'numerical_stationary_enumeration'} else 'analytic'
                            ax.plot([x for x, _ in references], [y for _, y in references], color=COLORS[gi % len(COLORS)],
                                    linestyle=':', marker='x', markersize=4, linewidth=.8,
                                    label=(METHODS.get(group, group) + ' reference (' + qualifier + ')') if experiment=='transfer' else 'Analytic ground truth')
                if categorical:
                    step = max(1, math.ceil(len(labels) / 12))
                    ticks = sorted(set(range(0, len(labels), step)) | {len(labels)-1})
                    shown = [str(int(labels[i].split('-')[-1])) if recipe.x=='replica_id' else METHODS.get(labels[i], labels[i]) for i in ticks]
                    ax.set_xticks(ticks, shown, rotation=35, ha='right', fontsize=6)
                elif recipe.x=='first_error_nats':
                    left, right = ax.get_xlim()
                    ax.set_xlim(min(0, left), max(0, right))
                if recipe.y == 'rate':
                    ax.set_ylim(0, 1)
                elif recipe.y == 'pearson_r':
                    ax.set_ylim(-1, 1)
                else:
                    bottom, top = ax.get_ylim()
                    ax.set_ylim(min(0, bottom), max(0, top))
                    if bottom < 0 < top or recipe.y in ('bias_nats', 'mean_difference_nats', 'mean'):
                        ax.axhline(0, color='#777777', linewidth=.6)
                ax.set_xlabel(LABELS.get(recipe.x, recipe.x))
                ax.set_ylabel(textwrap.fill(recipe.ylabel, width=38, break_long_words=False))
                label = 'ARTIFICIAL FIXTURE — NOT SCIENTIFIC RESULTS' if artificial else ('DIAGNOSTIC — NOT FINAL' if mode == 'diagnostic' else 'FINAL — APPROVED COMPLETE INPUTS')
                ax.set_title(label + '\n' + ', '.join(f'{k}={v}' for k, v in json.loads(facet).items()), fontsize=8)
                if len(groups) <= 12:
                    ax.legend(loc='best', frameon=False)
                else:
                    ax.text(.01, .99, 'Series identities retained in companion data map.', transform=ax.transAxes, va='top', fontsize=7)
                files = [stem + '.svg', stem + '.png']
                if any((directory / name).exists() for name in files):
                    plt.close(fig)
                    raise FileExistsError('Figure name collision; refuse overwrite.')
                try:
                    fig.savefig(directory / files[0], metadata={'Date': None, 'Creator': 'NJEE scientific reporting v1'})
                    fig.savefig(directory / files[1], dpi=dpi, metadata={'Software': 'NJEE scientific reporting v1'})
                finally:
                    plt.close(fig)
                caption = (f'{recipe.ylabel} against {LABELS.get(recipe.x, recipe.x)}. Facet: {facet}. '
                    f'{len(data)} observed points; {interval_count} uncertainty intervals. '
                    + ('Bonferroni paired t intervals, confidence 99.6875%, 16-comparison family.' if recipe.interval == 'confirmatory'
                       else 'Intervals, where available, are pointwise exploratory 95%; missing intervals are not imputed.')
                    + ' Independent datasets/processes are replication units. Signed primary estimates retained. '
                    + 'Success-only summaries may be biased when runs fail; exact denominators and omissions are in companion tables. '
                    + ('TE is observational; finite-history numerical references exclude roundoff from truncation bounds. '
                       'Surrogate calibration is conditional on configured structural-null path laws.' if experiment == 'transfer' else '')
                    + ('Cell-specific estimands may differ; consult same_estimand and reference tables.' if experiment == 'ablation' else ''))
                inventory.append({'recipe': recipe.name, 'status': 'generated', 'files': files, 'facet': json.loads(facet),
                    'source_table': recipe.table, 'source_csv': 'tables/' + recipe.table + '.csv', 'points': point_map,
                    'omitted_rows': omitted, 'caption': caption, 'mode': mode, 'dpi': dpi})
    return inventory
