"""Lossless normalized source tables plus CSV/LaTeX exports."""
import csv
import json
import math
from pathlib import Path


TABLE_NAMES = ('raw_records', 'summaries', 'uncertainty', 'paired_comparisons', 'confirmatory',
               'accuracy', 'components', 'correlations', 'terms', 'references', 'ordering', 'costs', 'eligibility')


def normalized_tables(report):
    """Formatting adapter only: all inferential quantities come from validated metrics."""
    tables = {name: [] for name in TABLE_NAMES}
    tables['raw_records'] = report['raw_records']
    tables['eligibility'] = [{'issue': s, 'final_eligible': False} for s in report['issues']] or [
        {'issue': None, 'complete_inputs': report['complete'], 'approval_required': True,
         'artificial_fixture': report['artificial_fixture'], 'final_eligible': False}]
    metadata = {c['cell_id']: c['metadata'] for c in report['cells']}
    tables['terms'] = [{**metadata[t['cell_id']], **t} for t in report['terms']]
    for cell in report['cells']:
        cid, meta, analysis = cell['cell_id'], cell['metadata'], cell['analysis']
        for summary in analysis['summaries']:
            common = {'cell_id': cid, **meta, 'variant': summary.get('variant', summary.get('job_id', meta.get('name'))),
                      **{k: summary[k] for k in ('embedding', 'lag', 'direction') if k in summary}}
            metrics = summary.get('metrics') or {k: summary.get(k) for k in
                ('mean_estimate_nats', 'bias_nats', 'mae_nats', 'rmse_nats', 'sample_std_nats')}
            row = {**common, **metrics, 'n_planned': summary['n_planned'],
                   'n_complete': summary.get('n_complete', summary.get('n_successful', 0)),
                   'n_failed': summary['n_failed'], 'n_not_attempted': summary['n_not_attempted'],
                   'mean_runtime_seconds': summary.get('mean_runtime_seconds'),
                   'negative_frequency': summary.get('negative_frequency'),
                   'inference': 'exploratory_pointwise', 'unit': 'nats',
                   'reference_statuses': summary.get('reference_statuses'),
                   'max_reference_truncation_bound_nats': summary.get('max_reference_truncation_bound_nats')}
            matching = [r for r in report['raw_records'] if r['cell_id'] == cid and r['status'] == 'complete'
                        and r.get('variant', r.get('job_id', meta.get('name'))) == common['variant']]
            truths = {r.get('ground_truth_nats', r.get('reference_nats', r.get('truth_nats'))) for r in matching}
            row['reference_nats'] = next(iter(truths)) if len(truths) == 1 else None
            row['reference_kinds'] = sorted({r.get('reference_status', r.get('reference', {}).get('status', 'analytic')) for r in matching})
            if 'family' in meta and 'parameter' in meta:
                row['scenario'] = f"{meta['family']} / parameter={meta['parameter']}"
            tables['summaries'].append(row)
            for metric, interval in summary['bootstrap'].items():
                tables['uncertainty'].append({**common, 'metric': metric, **interval,
                                               'inference': 'exploratory_pointwise'})
            if summary.get('negative_interval'):
                tables['accuracy'].append({**common, 'metric': 'negative_frequency',
                    'rate': summary['negative_frequency'], 'count': summary['negative_count'],
                    'n_evaluated': row['n_complete'], 'n_planned': row['n_planned'],
                    'n_missing': row['n_planned'] - row['n_complete'], **summary['negative_interval']})
            if summary.get('component_error_correlation'):
                tables['correlations'].append({**common, **summary['component_error_correlation'],
                                               'inference': 'exploratory_pointwise'})
        for pair in analysis.get('paired_comparisons', []):
            for metric, interval in pair['bootstrap'].items():
                tables['paired_comparisons'].append({'cell_id': cid, **meta, 'left': pair['left'], 'right': pair['right'],
                    'metric': metric, 'mean': pair['mean_' + metric], 'n_planned': pair['n_planned'],
                    'n_pairs': pair['n_paired_successful'], 'n_incomplete_pairs': pair['n_unpaired'],
                    'same_estimand': True, **interval, 'inference': 'exploratory_pointwise'})
        if 'false_positive_analysis' in analysis:
            f = analysis['false_positive_analysis']
            tables['accuracy'].append({'cell_id': cid, **meta, 'metric': 'false_positive_rate',
                'rate': f['rate'], 'count': f['n_rejected'], 'n_evaluated': f['n_tested'],
                'n_planned': f['n_planned_null'], 'n_missing': f['n_untested'], **f['interval'],
                'planned_rate_bounds': f['planned_rate_bounds'], 'decision': f['decision']})
        for name, item in analysis.get('accuracy', {}).items():
            tables['accuracy'].append({'cell_id': cid, **meta, 'metric': name, 'rate': item['rate'],
                'count': item['count'], 'n_evaluated': item['n_evaluated'], 'n_planned': item['n_eligible'],
                'n_missing': item['n_missing_or_inapplicable'], **item['interval'],
                'planned_rate_bounds': item['planned_bounds']})
        if 'direction_difference' in analysis:
            item = analysis['direction_difference']
            tables['paired_comparisons'].append({'cell_id': cid, **meta, 'left': 'X_to_Y', 'right': 'Y_to_X',
                'metric': 'oracle_direction_difference_nats', 'mean': item['mean_nats'], 'n_pairs': item['n_pairs'],
                'same_estimand': False, 'interpretation': 'oracle true-delay directional information contrast',
                **item['bootstrap'], 'inference': 'exploratory_pointwise'})
        for job, reference in cell.get('references', {}).items():
            tables['references'].append({'cell_id': cid, **meta, 'variant': job, **reference,
                                         'roundoff_certified': False})
    extra = report['extra_analysis']
    if report['experiment'] == 'entropy':
        tables['confirmatory'] = extra
    elif report['experiment'] == 'ablation':
        tables['ordering'] = extra['ordering_variability']
        for pair in extra['paired_comparisons']:
            for metric, statistic in pair['statistics'].items():
                tables['paired_comparisons'].append({'cell_id': pair['left'], **metadata[pair['left']],
                    **{k: pair[k] for k in ('left', 'right', 'same_estimand', 'interpretation', 'n_planned', 'n_pairs', 'n_incomplete_pairs')},
                    'metric': metric, 'mean': statistic['mean'], **statistic['interval'],
                    'inference': 'exploratory_pointwise'})
    for r in report['raw_records']:
        meta = metadata[r['cell_id']]
        common = {'cell_id': r['cell_id'], 'replica_id': r['replica_id'], **meta,
                  'variant': r.get('variant', r.get('job_id', meta.get('name'))), 'status': r['status']}
        if 'first_entropy_nats' in r and r['status'] == 'complete':
            tables['components'].append({**common,
                'first_error_nats': r['first_entropy_nats'] - r['first_truth_nats'],
                'second_error_nats': r['second_entropy_nats'] - r['second_truth_nats']})
        tables['costs'].append({**common, **{k: r.get(k) for k in
            ('runtime_seconds', 'parameter_count', 'optimizer_updates', 'term_batch_evaluations', 'n_train')},
            'surrogate_family_runtime_seconds': r.get('surrogate_family_runtime_seconds'),
            'note': 'Shared surrogate-family runtime repeats across candidates; do not sum.'})
        if r.get('reference'):
            tables['references'].append({**common, **r['reference'], 'roundoff_certified': False})
    return tables


def json_cell(value):
    if value is None:
        return ''
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite table value; must be explicitly unavailable.')
    return str(value)


def latex_escape(value):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(c, c) for c in value)


def table_columns(rows):
    """Empty/inapplicable tables retain a parseable header, with zero data rows."""
    return sorted({k for r in rows for k in r}) or ['status', 'reason']


def export_table(directory, name, rows):
    """Full-precision machine CSV + compact longtable TeX, without significance stars."""
    columns = table_columns(rows)
    with (directory / (name + '.csv')).open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_cell(v) for k, v in row.items()})
    # Wide audit tables remain in CSV; TeX is a vertical record table, avoiding
    # unreadable dozens-of-columns layouts. Scientific summaries use selected columns.
    layouts = {
        'summaries': ('cell_id', 'variant', 'n_complete', 'bias_nats', 'mae_nats', 'rmse_nats', 'sample_std_nats'),
        'uncertainty': ('cell_id', 'variant', 'metric', 'n_units', 'confidence_level', 'lower', 'upper'),
        'paired_comparisons': ('left', 'right', 'metric', 'n_pairs', 'mean', 'lower', 'upper', 'same_estimand'),
        'confirmatory': ('cell', 'right', 'n_pairs', 'confidence_level', 'mean_difference_nats', 'lower_nats', 'upper_nats'),
        'accuracy': ('cell_id', 'metric', 'count', 'n_evaluated', 'n_missing', 'rate', 'lower', 'upper'),
        'references': ('cell_id', 'variant', 'status', 'transfer_entropy_nats', 'entropy_truncation_error_bound_nats', 'roundoff_certified'),
        'costs': ('cell_id', 'variant', 'replica_id', 'runtime_seconds', 'parameter_count', 'optimizer_updates', 'term_batch_evaluations'),
        'eligibility': ('issue', 'final_eligible'),
        'correlations': ('cell_id', 'variant', 'n_pairs', 'pearson_r', 'lower', 'upper', 'valid_bootstrap_draws'),
    }
    selected = [c for c in layouts.get(name, ('cell_id', 'variant', 'replica_id', 'status')) if c in columns]
    if not selected:
        selected = columns[:4]
    headers = {'cell_id': 'Cell', 'variant': 'Method', 'metric': 'Quantity', 'n_complete': 'n complete',
               'n_planned': 'n planned', 'n_pairs': 'n pairs', 'n_units': 'n units', 'confidence_level': 'Confidence',
               'mean_estimate_nats': 'Estimate (nats)', 'bias_nats': 'Bias (nats)', 'mae_nats': 'MAE (nats)',
               'rmse_nats': 'RMSE (nats)', 'sample_std_nats': 'SD (nats)', 'mean_difference_nats': 'Difference (nats)',
               'lower_nats': 'Lower (nats)', 'upper_nats': 'Upper (nats)', 'parameter_count': 'Parameters',
               'optimizer_updates': 'Updates', 'term_batch_evaluations': 'Term batches', 'runtime_seconds': 'Seconds'}
    text_columns = {'cell_id', 'variant', 'metric', 'left', 'right', 'issue', 'status', 'replica_id'}
    count_text = max(1, sum(c in text_columns for c in selected))
    specification = ''.join(f'p{{{.40 / count_text:.3f}\\linewidth}}' if c in text_columns else 'r' for c in selected) or 'l'
    lines = [r'% Generated from validated source table ' + name + r'; full precision in companion CSV.',
             '% Columns: ' + ', '.join(latex_escape(c) for c in selected), r'\begingroup\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{longtable}{' + specification + '}', r'\hline']
    if selected:
        lines.extend([' & '.join(latex_escape(headers.get(c, c.replace('_', ' '))) for c in selected) + r' \\', r'\hline'])
    for row in rows:
        values = []
        for c in selected:
            value = row.get(c)
            # Presentation rounding only; raw values remain in CSV/JSON.
            label = (str(value) if c=='confidence_level' else format(value, '.4g')) if isinstance(value, float) else json_cell(value)
            escaped = latex_escape(label)
            if c in text_columns and len(label)>15:
                escaped = escaped.replace(r'\_', r'\_\allowbreak{}')
            values.append(escaped if label else r'\textemdash{}')
        lines.append(' & '.join(values) + r' \\')
    lines.extend([r'\hline', r'\end{longtable}', r'\endgroup'])
    with (directory / (name + '.tex')).open('x', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
