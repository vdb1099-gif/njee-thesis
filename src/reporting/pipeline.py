"""Unified reporting lifecycle."""
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
from pathlib import Path
import subprocess
import sys

import matplotlib
import numpy as np

from experiments.final_protocol import digest
from src.reporting.adapters import (HASHES, frozen_plan, validate_source, inventory,
                                    read_json, file_hash, safe_artifact)
from src.reporting.figures import recipes, generate_figures
from src.reporting.tables import TABLE_NAMES, normalized_tables, export_table, table_columns


SCHEMA = 'njee-unified-report-v1'


def reporting_plan(experiment=None):
    """No observations, datasets, estimator fits, source roots or output writes."""
    experiments = [experiment] if experiment else list(HASHES)
    return {'schema': SCHEMA, 'mode': 'plan_only_no_observations', 'experiments': [
        {'experiment': e, 'protocol_hash': frozen_plan(e)['protocol_hash'],
         'scientific_counts': frozen_plan(e)['expected'], 'tables': list(TABLE_NAMES),
         'figure_recipes': [asdict(r) for r in recipes(e)]} for e in experiments],
        'modes': {'final': 'complete validated frozen outputs + explicit inventory-bound review approval',
                  'diagnostic': 'incomplete results visibly labelled; corruption still fatal'},
        'scientific_execution': False}


def code_provenance():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        p = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, check=True)
        return p.stdout.strip()
    try:
        revision, status = git('rev-parse', 'HEAD'), git('status', '--short')
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, 'Git unavailable; content hashes identify reporting code.'
    paths = [*Path(__file__).parent.glob('*.py'), root / 'experiments' / 'scientific_report.py']
    return {'git_revision': revision, 'working_tree_status': status,
            'reporting_code_sha256': {p.relative_to(root).as_posix(): file_hash(p) for p in paths if p.exists()}}


def require_final_approval(report, approval):
    if not report['complete'] or report['artificial_fixture']:
        raise ValueError('Final report ineligible: incomplete evidence or artificial fixtures.')
    if approval is None:
        raise ValueError('Final report requires explicit inventory-bound reporting approval.')
    record = read_json(approval)
    required = {'experiment', 'protocol_hash', 'source_inventory_hash', 'approved_for_final_reporting', 'reviewed_by'}
    if not required <= set(record) or record['approved_for_final_reporting'] is not True or not isinstance(record['reviewed_by'], str) or not record['reviewed_by'].strip():
        raise ValueError('Invalid final-report approval record.')
    if any(record[k] != report[k] for k in ('experiment', 'protocol_hash', 'source_inventory_hash')):
        raise ValueError('Approval is not bound to these scientific inputs.')
    return {'record': record, 'sha256': file_hash(approval)}


def generate_report(experiment, source, output, *, mode='final', approval=None, ledger_path=None, dpi=300):
    if mode not in ('final', 'diagnostic'):
        raise ValueError('Choose final or diagnostic explicitly.')
    if type(dpi) is not int or not 150 <= dpi <= 1200:
        raise ValueError('PNG dpi must be an integer in [150,1200].')
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError('Reporting never overwrites an existing output path.')
    if Path(source).is_symlink():
        raise ValueError('Symlink scientific input root is unsupported.')
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Report outputs must be disjoint from scientific inputs.')
    if output.exists():
        raise FileExistsError('Reporting never overwrites or resumes outputs.')
    if approval is not None and Path(approval).resolve().is_relative_to(source):
        raise ValueError('Keep reporting approval outside the immutable scientific input root.')
    report = validate_source(experiment, source, ledger_path=ledger_path)
    approval_record = require_final_approval(report, approval) if mode == 'final' else None
    tables = normalized_tables(report)
    for row in tables['eligibility']:
        row.update(report_mode=mode, final_eligible=mode=='final' and report['complete'] and not report['artificial_fixture'])
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': SCHEMA, 'status': 'generating', 'mode': mode, 'experiment': experiment,
        'scientific_final': False, 'artificial_fixture': report['artificial_fixture'],
        'completeness': {k: report[k] for k in ('complete', 'issues', 'planned_cells', 'observed_cells', 'planned_records', 'complete_records')},
        'protocol_hash': report['protocol_hash'], 'source_root': str(source),
        'source_inventory': report['source_inventory'], 'source_inventory_hash': report['source_inventory_hash'],
        'source_execution': report['source_manifest'], 'source_execution_ledger': report['source_ledger'],
        'reporting_code': code_provenance(), 'approval': approval_record,
        'table_inventory': {name: {'records': len(rows), 'columns': table_columns(rows),
                                  'empty_or_inapplicable': not rows, 'csv': 'tables/'+name+'.csv',
                                  'latex': 'tables/'+name+'.tex'} for name, rows in tables.items()},
        'configuration': {'dpi': dpi, 'vector_format': 'svg', 'raster_format': 'png', 'units': 'nats', 'log_base': 'e',
                          'figure_size_inches': [6.4, 4.0], 'font_family': 'DejaVu Sans',
                          'style_version': 'njee-static-v1', 'latex_significant_digits': 4,
                          'csv_precision': 'full round-trip', 'confidence_level_format': 'exact declared value'},
        'environment': {'python': platform.python_version(), 'interpreter': sys.executable, 'numpy': np.__version__,
                        'matplotlib': matplotlib.__version__, 'scipy': version('scipy'), 'torch': version('torch'),
                        'platform': platform.platform(), 'cpu': platform.processor()},
        'statistics': {'reused': ['aggregate_records', 'paired_intervals', 'summarize_information', 'summarize_transfer', 'ablation.summarize'],
                       'confirmatory': 'Experiment 1: fixed 16-comparison Bonferroni two-sided paired t; 99.6875%; all 30 pairs required',
                       'exploratory': 'Pointwise 95% replica bootstrap / exact binomial intervals, as frozen per experiment',
                       'replication_unit': 'independent dataset/process; no rows/orders/terms/windows as independent units',
                       'references': 'analytic vs stationary numerical; truncation bounds exclude floating-point roundoff',
                       'calibration': 'TE structural-null surrogate decisions are conditional on the configured known path laws',
                       'input_policy': 'Recalculate existing validated reporting statistics in memory; never rewrite source summaries'},
        'created_utc': datetime.now(timezone.utc).isoformat()}
    path = output / 'report_manifest.json'
    def save():
        manifest['manifest_digest'] = digest({k: v for k, v in manifest.items() if k != 'manifest_digest'})
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    save()
    try:
        td, fd = output / 'tables', output / 'figures'
        td.mkdir(); fd.mkdir()
        for name, rows in tables.items():
            export_table(td, name, rows)
        with (output / 'validated_analysis.json').open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, allow_nan=False)
        figures = generate_figures(fd, tables, experiment, mode=mode, artificial=report['artificial_fixture'], dpi=dpi)
        with (output / 'figure_index.json').open('x', encoding='utf-8') as f:
            json.dump(figures, f, indent=2, ensure_ascii=False, allow_nan=False)
        with (output / 'captions.md').open('x', encoding='utf-8') as f:
            f.write(('ARTIFICIAL REPORTING FIXTURE — NOT SCIENTIFIC EVIDENCE\n\n' if report['artificial_fixture'] else '')
                    + ('DIAGNOSTIC REPORT — NOT FINAL SCIENTIFIC RESULTS\n\n' if mode == 'diagnostic' else 'APPROVED COMPLETE INPUTS — USE ONLY WHEN REPORT MANIFEST STATUS IS COMPLETE\n\n'))
            for figure in figures:
                f.write(figure['recipe'] + ': ' + figure.get('caption', 'Unavailable: ' + figure.get('reason', '')) + '\n\n')
        if inventory(source) != report['source_inventory']:
            raise ValueError('Scientific inputs changed during report generation.')
        manifest['output_artifacts'] = {p.relative_to(output).as_posix(): file_hash(p)
                                        for p in sorted(output.rglob('*')) if p.is_file() and p != path}
        manifest['figure_index'] = 'figure_index.json'
        manifest['status'] = 'complete' if mode == 'final' else 'diagnostic_complete'
        manifest['scientific_final'] = mode == 'final'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        failure={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    return manifest


def verify_report(output, *, verify_inputs=False):
    output = Path(output).resolve()
    manifest = read_json(output / 'report_manifest.json')
    if manifest.get('manifest_digest') != digest({k: v for k, v in manifest.items() if k != 'manifest_digest'}):
        raise ValueError('Report manifest metadata checksum mismatch.')
    if manifest['protocol_hash'] != HASHES.get(manifest['experiment']):
        raise ValueError('Report scientific protocol hash is not approved.')
    if manifest['schema'] != SCHEMA or manifest['status'] not in ('complete', 'diagnostic_complete'):
        raise ValueError('Report is incomplete or unsupported.')
    expected = set(manifest['output_artifacts']) | {'report_manifest.json'}
    observed = set(inventory(output))
    if observed != expected:
        raise ValueError('Missing/unexpected report artifacts.')
    for name, sha in manifest['output_artifacts'].items():
        if file_hash(safe_artifact(output, name)) != sha:
            raise ValueError('Report artifact hash mismatch.')
    if digest(manifest['source_inventory']) != manifest['source_inventory_hash']:
        raise ValueError('Input inventory metadata changed.')
    if manifest['scientific_final'] and (manifest['mode'] != 'final' or not manifest['completeness']['complete'] or not manifest['approval'] or manifest['artificial_fixture']):
        raise ValueError('Invalid final-report eligibility declaration.')
    if verify_inputs and inventory(Path(manifest['source_root'])) != manifest['source_inventory']:
        raise ValueError('Archived scientific inputs have changed.')
    return {'status': 'verified', 'mode': manifest['mode'], 'scientific_final': manifest['scientific_final'],
            'artifact_count': len(manifest['output_artifacts']), 'protocol_hash': manifest['protocol_hash'],
            'limitation': 'Hashes detect accidental changes; this manifest is not a cryptographic signature.'}
