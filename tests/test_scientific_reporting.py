"""ARTIFICIAL reporting fixtures only. No fitted estimator or scientific campaign."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.reporting import adapters, figures, pipeline, tables
from experiments import scientific_report
from src.utils.split import make_iid_split, save_split
from src.utils.temporal import build_temporal_embedding, make_chronological_split


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def hashes(directory):
    return {p.relative_to(directory).as_posix(): adapters.file_hash(p) for p in directory.rglob('*')
            if p.is_file() and p.name != 'manifest.json'}


@pytest.fixture(autouse=True)
def no_scientific_execution(monkeypatch):
    from src.estimators.neural import NeuralJointEntropyEstimator
    from src.estimators.neural_information import NeuralConditionalMutualInformationEstimator
    from src.estimators.transfer_entropy import NeuralTransferEntropyEstimator
    from src.simulation.coupled_binary import CoupledBinaryModel
    from src.simulation.information import InformationModel
    def forbidden(*a, **k):
        pytest.fail('Reporting must never execute a scientific fit/generator.')
    for cls in (NeuralJointEntropyEstimator, NeuralConditionalMutualInformationEstimator, NeuralTransferEntropyEstimator):
        monkeypatch.setattr(cls, 'fit', forbidden)
    for cls in (CoupledBinaryModel, InformationModel):
        monkeypatch.setattr(cls, 'generate', forbidden)


@pytest.fixture(scope='module')
def plans():
    return {k: adapters.frozen_plan(k) for k in adapters.HASHES}


def fixture_plan(kind, plans):
    p = deepcopy(plans[kind])
    if kind == 'entropy':
        p['cells'] = p['cells'][:1]
        c = p['cells'][0]
        c['replicas'] = c['replicas'][:2]
        c['campaign']['aggregation']['bootstrap_draws'] = 3
        c['data']['n'] = c['baseline']['data']['n'] = 40
    elif kind == 'information':
        p['cells'] = [next(c for c in p['cells'] if c['truth']['is_null'] and c['model']['family'] == 'bsc')]
        p['cells'][0]['allocation'] = p['cells'][0]['allocation'][:2]
        p['cells'][0].update(replicas=2, n=40)
        p['definition']['bootstrap_draws'] = 3
    elif kind == 'transfer':
        p['cells'] = p['cells'][:1]
        p['cells'][0]['allocation'] = p['cells'][0]['allocation'][:2]
        p['cells'][0].update(replicas=2, n=40)
        p['definition']['bootstrap_draws'] = 3
    else:
        p['cells'] = [c for c in p['cells'] if c['group'] == 'discretization']
        p['groups'] = [next(g for g in p['groups'] if g['group_id'] == 'discretization')]
        p['groups'][0]['allocation'] = p['groups'][0]['allocation'][:2]
        p['comparisons'] = [r for r in p['comparisons'] if r['left'] in {c['cell_id'] for c in p['cells']}]
        p['definition'].update(n=40, replicas=2, bootstrap_draws=3)
    return p


def fake_root(root, kind, p):
    root.mkdir()
    manifest = {'status': 'complete', 'protocol_hash': p['protocol_hash'], 'git': {'commit': 'ARTIFICIAL', 'dirty': False},
                'environment': {'artificial_fixture': True}, 'artificial_fixture': True}
    if kind == 'entropy':
        ledger = {**manifest, 'git_commit': 'ARTIFICIAL', 'campaigns': [
            {'campaign_id': c['campaign_id'], 'status': 'complete'} for c in p['cells']]}
        write(root / 'control' / 'ledger.json', ledger)
    else:
        write(root / 'protocol.json', p)
        write(root / 'manifest.json', manifest)
        entries = [{'cell_id': c['cell_id'], 'status': 'complete'} for c in p['cells']]
        if kind == 'ablation':
            entries = [{'group': g['group_id'], 'replica_id': r['replica_id'], 'status': 'complete'}
                       for g in p['groups'] for r in g['allocation']]
        write(root / 'execution_ledger.json', entries)


def iid_artifacts(rd, values, split_seed):
    rd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rd / 'dataset.npz', values=values)
    split = make_iid_split(len(values), seed=split_seed)
    save_split(split, rd / 'split.npz')
    return {'full': len(values), **{k: len(getattr(split, k)) for k in ('train', 'validation', 'test')}}, hashlib.sha256(values.tobytes()).hexdigest()


def entropy_fixture(root, p):
    from experiments.neural_baseline import VARIANTS
    c = p['cells'][0]; cid = c['campaign_id']; directory = root / cid
    science = {'data': c['data'], 'neural': {k: v for k, v in c['baseline']['neural'].items() if k not in ('split_seed', 'training_seed')}}
    scenario = hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()
    write(directory / 'campaign_manifest.json', {'status': 'complete', 'baseline_configuration': c['baseline'],
        'replicas': [{k: r[k] for k in ('replica_id', 'data_seed', 'split_seed', 'training_seed')} for r in c['replicas']],
        'purpose': 'final', 'configuration': c['campaign']})
    rows = []
    for replica in c['replicas']:
        rid = replica['replica_id']; rd = directory / 'replicas' / rid
        values = np.tile(np.array([[0, 1, 0, 1], [1, 0, 1, 0]]), (20, 1))
        counts, sha = iid_artifacts(rd, values, replica['split_seed'])
        original = []
        for v in VARIANTS:
            estimate = c['ground_truth_nats'] + .125
            r = {'campaign_id': cid, 'replica_id': rid, 'variant': v, 'scenario_id': scenario,
                 **{k: replica[k] for k in ('data_seed', 'split_seed', 'training_seed')},
                 'status': 'complete', 'entropy_nats': estimate, 'ground_truth_nats': c['ground_truth_nats'],
                 'signed_error_nats': estimate - c['ground_truth_nats'], 'sample_counts': counts,
                 'runtime_seconds': 1., 'fit_runtime_seconds': .5, 'inference_runtime_seconds': .5, 'entropy_unit': 'nats'}
            rows.append(r); original.append(r)
        write(rd / 'results.json', original)
        write(rd / 'njee_diagnostics.json', {'terms': [{'position': 2, 'contribution_nats': .5, 'best_validation_ce_nats': .6}]})
        base = deepcopy(c['baseline'])
        base['neural'].update(split_seed=replica['split_seed'], training_seed=replica['training_seed'])
        write(rd / 'manifest.json', {'status': 'complete', 'sample_counts': counts, 'dataset_sha256': sha,
            'dataset_id': rid+':'+sha, 'split_id': rid+':iid-pcg64-largest-remainder-v1:'+str(replica['split_seed']),
            'resolved_configuration': base, 'seeds': {'data_generation': replica['data_seed'],
                'split': replica['split_seed'], 'training': replica['training_seed']}})
    write(directory / 'raw_results.json', rows)


def information_fixture(root, p):
    from experiments.information_validation import _empty_rows
    c = p['cells'][0]; directory = root / c['cell_id']; rows = _empty_rows(c)
    for allocation in c['allocation']:
        rid = allocation['replica_id']; rd = directory / rid
        counts, sha = iid_artifacts(rd, np.tile([[0, 1], [1, 0]], (20, 1)), allocation['split_seed'])
        identity = c['cell_id']+':'+rid+':'+sha
        split_id = identity+':iid-pcg64-largest-remainder-v1:'+str(allocation['split_seed'])
        for r in (r for r in rows if r['replica_id'] == rid):
            r.update(status='complete', entropy_nats=-.25, information_nats=-.25,
                signed_error_nats=-.25, first_entropy_nats=.5, second_entropy_nats=.75,
                runtime_seconds=1., fit_runtime_seconds=.5, inference_runtime_seconds=.5,
                dataset_id=identity, split_id=split_id, sample_counts=counts,
                test_status='complete' if r['variant'] == 'neural' else 'not_applicable',
                p_value=.001 if r['variant'] == 'neural' else None, reject=r['variant'] == 'neural')
        write(rd / 'neural_diagnostics.json', {'components': []})
        write(rd / 'independence_test.json', {'method': 'frozen-model-heldout-exact-strata-permutation-v1',
            'seed': allocation['permutation_seed'], 'draws': p['definition']['permutation_draws'], 'refits': 0,
            'observed_nats': -.25, 'p_value': .001, 'surrogate_statistics_nats': [-.5] * p['definition']['permutation_draws']})
        write(rd / 'manifest.json', {'seeds': allocation, 'truth': c['truth'], 'status': 'complete',
            'dataset_id': identity, 'split_id': split_id, 'sample_counts': counts, 'variable_ids': ['X','Y'],
            'dataset_sha256': sha, 'artifact_sha256': hashes(rd)})
    write(directory / 'raw_results.json', rows)


def reference(job):
    memory = max(1, *job['source_lags'], *job['target_lags'])
    return {'status': 'analytic_zero', 'version': 'pair-chain-power-coupling-v1', 'transfer_entropy_nats': 0.,
            'memory': memory, 'n_states': 4 ** memory, 'iterations': 1,
            'first_entropy_nats': .5, 'second_entropy_nats': .5, 'direction': job['direction'],
            'source_lags': job['source_lags'], 'target_lags': job['target_lags'], 'horizon': 1,
            'tv_truncation_bound': 0., 'stationarity_l1_residual': 0., 'entropy_truncation_error_bound_nats': 0.,
            'floating_point': 'ARTIFICIAL float64; truncation bound excludes roundoff'}


def transfer_fixture(root, p):
    from experiments.transfer_validation import empty_rows
    c = p['cells'][0]; directory = root / c['cell_id']
    refs = {j['job_id']: reference(j) for j in c['jobs']}
    write(directory / 'references.json', refs)
    for allocation in c['allocation']:
        rid = allocation['replica_id']; rd = directory / rid; rd.mkdir()
        x, y = np.tile([0, 1], 20), np.tile([1, 0], 20)
        identity = c['cell_id']+':'+rid+':'+hashlib.sha256(np.column_stack((x, y)).tobytes()).hexdigest()
        np.savez_compressed(rd / 'process.npz', X=x, Y=y, time=np.arange(40))
        rows = empty_rows(c, allocation)
        for r in rows:
            from experiments.transfer_validation import portable
            from src.utils.temporal import TemporalEmbeddingSpec
            spec = TemporalEmbeddingSpec(tuple(r['source_lags']), tuple(r['target_lags']))
            offset = 6 - max(spec.source_lags + spec.target_lags)
            source, target = (x, y) if r['direction'] == 'X_to_Y' else (y, x)
            e = build_temporal_embedding(source[offset:], target[offset:], spec=spec, dataset_id=identity,
                source_id='X' if r['direction']=='X_to_Y' else 'Y', target_id='Y' if r['direction']=='X_to_Y' else 'X',
                source_times=np.arange(offset, 40), target_times=np.arange(offset, 40))
            parts = make_chronological_split(e, gap=6, purge=True)
            r.update(status='complete', te_nats=-.125, reference_nats=0., signed_error_nats=-.125,
                reference_status='analytic_zero', reference_error_bound_nats=0., validation_score_nats=-.125,
                dataset_id=parts.test.dataset_id, split_id=parts.test.split_id, runtime_seconds=1.,
                test_timestamps=list(map(int, parts.provenance[2].target_timestamps)), test_status='complete' if r['embedding'] == 'canonical' else 'not_applicable',
                p_value=1. if r['embedding'] == 'canonical' else None)
            write(rd / r['job_id'] / 'diagnostics.json', {'cmi_report': {'components': []}, 'embedding':
                {'source_lags': r['source_lags'], 'target_lags': r['target_lags'], 'horizon': 1},
                'partitions': portable(parts.provenance), 'gap': 6, 'purge': True, 'split_fractions': [.6, .2, .2]})
        for direction in ('X_to_Y', 'Y_to_X'):
            selected = [r for r in rows if r['direction']==direction and r['embedding']=='canonical']
            write(rd / (direction + '-surrogates.json'), {'method': 'oracle-conditional-path-frozen-statistic-v1',
                'seed': allocation['surrogates'][direction]['seed'], 'draws': p['definition']['surrogate_draws'], 'refits': 0,
                'observed_nats': [r['te_nats'] for r in selected], 'p_values': [1.] * len(selected),
                'surrogate_statistics_nats': [[0.] * len(selected)] * p['definition']['surrogate_draws']})
        write(rd / 'raw_results.json', rows)
        write(rd / 'manifest.json', {'cell_id': c['cell_id'], 'replica': allocation, 'status': 'complete',
            'dataset_id': identity, 'dataset_sha256': hashlib.sha256(np.column_stack((x, y)).tobytes()).hexdigest(),
            'artifact_sha256': hashes(rd)})


def ablation_fixture(root, p):
    from experiments.ablation_validation import empty_rows
    from src.utils.ablation_data import array_digest, discretized_entropy
    rows = empty_rows(p)
    group = p['groups'][0]
    for allocation in group['allocation']:
        rid = allocation['replica_id']; rd = root / group['group_id'] / rid
        values = np.arange(240).reshape(40, 6) / 240.
        counts, _ = iid_artifacts(rd, values, allocation['split']['seed'])
        identity = group['group_id']+':'+rid+':'+array_digest(values)
        split_id = identity+':iid:'+str(allocation['split']['seed'])
        own = [r for r in rows if r['replica_id'] == rid]
        for r in own:
            c = next(c for c in p['cells'] if c['cell_id'] == r['cell_id'])
            truth = discretized_entropy(6, c['states'])
            r.update(status='complete', estimate_nats=truth-.125, truth_nats=truth, signed_error_nats=-.125,
                runtime_seconds=1., parent_dataset_id=identity, parent_split_id=split_id,
                seeds=allocation['fits'][c['cell_id']], parameter_count=10, optimizer_updates=2,
                term_batch_evaluations=2, n_train=24)
            write(rd / c['cell_id'] / 'diagnostics.json', {'diagnostics': {'terms': []}})
        write(rd / 'raw_results.json', own)
        write(rd / 'manifest.json', {'group': group['group_id'], 'allocation': allocation, 'status': 'complete',
            'dataset_id': identity, 'parent_split_id': split_id,
            'data_sha256': array_digest(values), 'artifact_sha256': hashes(rd)})


@pytest.fixture
def source_factory(tmp_path, plans, monkeypatch):
    def make(kind):
        p = fixture_plan(kind, plans)
        monkeypatch.setattr(adapters, 'frozen_plan', lambda e: p if e == kind else plans[e])
        root = tmp_path / ('ARTIFICIAL-' + kind)
        fake_root(root, kind, p)
        {'entropy': entropy_fixture, 'information': information_fixture, 'transfer': transfer_fixture,
         'ablation': ablation_fixture}[kind](root, p)
        return root, p
    return make


@pytest.mark.parametrize('kind', tuple(adapters.HASHES))
def test_adapters_read_only_signed_values_and_schema(source_factory, kind):
    root, p = source_factory(kind)
    before = adapters.inventory(root)
    report = adapters.validate_source(kind, root)
    assert adapters.inventory(root) == before
    assert report['complete'] and report['artificial_fixture']
    normalized = tables.normalized_tables(report)
    assert normalized['summaries'] and report['planned_cells'] == len(p['cells'])
    if kind in ('information', 'transfer'):
        assert normalized['summaries'][0]['mean_estimate_nats'] < 0
    if kind == 'information':
        fpr = next(r for r in normalized['accuracy'] if r['metric'] == 'false_positive_rate')
        assert fpr['n_evaluated'] == fpr['count'] == 2 and fpr['rate'] == 1.
        assert normalized['correlations'][0]['pearson_r'] is None  # explicitly degenerate
    if kind == 'transfer':
        assert all(r['roundoff_certified'] is False for r in normalized['references'])
    if kind == 'ablation':
        assert all(r['same_estimand'] is False for r in normalized['paired_comparisons'])


def test_missing_duplicate_failed_records_and_eligibility(source_factory):
    root, p = source_factory('entropy'); path = root / p['cells'][0]['campaign_id'] / 'raw_results.json'
    rows = adapters.read_json(path)
    write(path, rows + rows[:1])
    with pytest.raises(ValueError, match='Duplicate'): adapters.validate_source('entropy', root)
    rows.pop(); rows[0]['status'] = 'failed'; write(path, rows)
    report = adapters.validate_source('entropy', root)
    assert not report['complete'] and report['planned_records'] == 10 and report['complete_records'] == 8
    assert any('missing' in s for s in report['issues'])
    with pytest.raises(ValueError, match='ineligible'): pipeline.require_final_approval(report, None)


def test_protocol_artifact_units_and_nonfinite_rejection(source_factory):
    root, p = source_factory('information')
    path = root / 'protocol.json'; value = adapters.read_json(path); value['protocol_hash'] = 'bad'; write(path, value)
    with pytest.raises(ValueError, match='protocol'): adapters.validate_source('information', root)
    write(path, p)
    artifact = next(root.rglob('neural_diagnostics.json')); artifact.write_text('{}')
    with pytest.raises(ValueError, match='hash'): adapters.validate_source('information', root)
    with pytest.raises(ValueError, match='nats'): adapters.valid_value({'entropy_unit': 'bits'}, 'v', 't')
    bad = root / 'nonfinite.json'; bad.write_text('{"v":1e999}')
    with pytest.raises(ValueError, match='Nonfinite'): adapters.read_json(bad)


def test_final_approval_inventory_binding_and_artificial_rejection(source_factory, tmp_path):
    root, _ = source_factory('ablation'); r = adapters.validate_source('ablation', root)
    with pytest.raises(ValueError, match='ineligible'): pipeline.require_final_approval(r, None)
    # Eligibility logic fixture only: never emit artificial data as a final report.
    controlled = {**r, 'artificial_fixture': False}
    with pytest.raises(ValueError, match='approval'): pipeline.require_final_approval(controlled, None)
    approval = tmp_path / 'approval.json'
    record = {k: r[k] for k in ('experiment', 'protocol_hash', 'source_inventory_hash')}
    record.update(approved_for_final_reporting=True, reviewed_by='ARTIFICIAL TEST REVIEWER')
    write(approval, record)
    assert pipeline.require_final_approval(controlled, approval)['record'] == record
    record['source_inventory_hash'] = 'wrong'; write(approval, record)
    with pytest.raises(ValueError, match='bound'): pipeline.require_final_approval(controlled, approval)


def test_diagnostic_generation_exports_provenance_no_overwrite(source_factory, tmp_path, monkeypatch):
    root, _ = source_factory('information')
    real = figures.recipes
    monkeypatch.setattr(figures, 'recipes', lambda e: [r for r in real(e) if r.name in
        ('mean_estimate_nats-sample-size', 'null-false-positive-rate', 'paired-absolute-error')])
    before = adapters.inventory(root)
    output = tmp_path / 'ARTIFICIAL-REPORT'
    manifest = pipeline.generate_report('information', root, output, mode='diagnostic', dpi=150)
    assert manifest['status'] == 'diagnostic_complete' and not manifest['scientific_final']
    assert manifest['table_inventory']['confirmatory']['records'] == 0
    assert adapters.inventory(root) == before
    assert pipeline.verify_report(output, verify_inputs=True)['status'] == 'verified'
    assert 'ARTIFICIAL' in (output / 'captions.md').read_text(encoding='utf-8')
    index = adapters.read_json(output / 'figure_index.json')
    for f in index:
        if f['status'] != 'generated': continue
        with (output / f['source_csv']).open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        recipe = next(r for r in figures.recipes('information') if r.name == f['recipe'])
        assert all(float(rows[p['csv_row_index_zero_based']][recipe.y]) == p['y'] for p in f['points'])
        assert all((output / 'figures' / name).exists() for name in f['files'])
    assert '\\begin{longtable}' in (output / 'tables' / 'summaries.tex').read_text()
    with pytest.raises(FileExistsError): pipeline.generate_report('information', root, output, mode='diagnostic')
    with pytest.raises(ValueError, match='disjoint'): pipeline.generate_report('information', root, root / 'out', mode='diagnostic')
    second = tmp_path / 'ARTIFICIAL-SECOND-REPORT'
    pipeline.generate_report('information', root, second, mode='diagnostic', dpi=150)
    for image in (output / 'figures').iterdir():
        assert image.read_bytes() == (second / 'figures' / image.name).read_bytes()
    (output / 'tables' / 'summaries.csv').write_text('tampered')
    with pytest.raises(ValueError, match='hash'): pipeline.verify_report(output)


def test_confirmatory_intervals_are_not_pointwise_bootstrap(source_factory, tmp_path):
    root, _ = source_factory('entropy'); report = adapters.validate_source('entropy', root)
    t = tables.normalized_tables(report)
    assert len(t['confirmatory']) == 16
    assert all(r['confidence_level'] == .996875 and r['family_size'] == 16 for r in t['confirmatory'])
    assert all(r['lower_nats'] is None and r['status'] == 'incomplete' for r in t['confirmatory'])
    assert all(r['confidence_level'] == .95 for r in t['uncertainty'])
    directory = tmp_path / 'tables'; directory.mkdir()
    tables.export_table(directory, 'confirmatory', t['confirmatory'])
    assert 'confidence\\_level' in (directory / 'confirmatory.tex').read_text()


def test_reference_metadata_not_a_floating_point_certificate():
    job = {'direction': 'X_to_Y', 'source_lags': [2], 'target_lags': [1]}
    r = reference(job); r['status'] = 'numerical_stationary_enumeration'
    assert adapters.reference_metadata(r, {'coupling': .3}, job)['roundoff_certified'] is False
    r['floating_point'] = 'certified all errors'
    with pytest.raises(ValueError, match='reference'): adapters.reference_metadata(r, {'coupling': .3}, job)


def test_plan_cli_safety_and_all_figure_factors(plans, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, 'frozen_plan', lambda e: plans[e])
    p = pipeline.reporting_plan()
    assert len(p['experiments']) == 4 and p['scientific_execution'] is False
    assert len(p['experiments'][3]['figure_recipes']) >= 12
    assert {r.name for r in figures.recipes('ablation')} >= {'architecture-rmse', 'sharing-rmse', 'training_size-rmse',
        'early_stopping-rmse', 'discretization-rmse', 'embedding-rmse', 'ordering-variability'}
    scientific_report.main(['--plan', '--experiment', 'entropy'])
    assert 'plan_only_no_observations' in capsys.readouterr().out
    with pytest.raises(SystemExit): scientific_report.main(['--generate', 'missing'])
    with pytest.raises(ValueError): adapters.safe_artifact(Path('.'), 'C:/private/key')
    with pytest.raises(ValueError): adapters.safe_artifact(Path('.'), '../escape')


def test_table_precision_latex_escaping_missing_and_nonfinite(tmp_path):
    rows = [{'cell_id': 'a_b%&', 'variant': 'method', 'bias_nats': -.123456789123456, 'rmse_nats': None}]
    tables.export_table(tmp_path, 'summaries', rows)
    assert '-0.123456789123456' in (tmp_path / 'summaries.csv').read_text()
    assert 'a\\_b\\%\\&' in (tmp_path / 'summaries.tex').read_text()
    with pytest.raises(ValueError): tables.json_cell(float('nan'))
    tables.export_table(tmp_path, 'accuracy', [])
    with (tmp_path / 'accuracy.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['status', 'reason'] and list(reader) == []


def test_incomplete_null_test_denominator_is_not_planned_count(source_factory):
    root, p = source_factory('information')
    path = root / p['cells'][0]['cell_id'] / 'raw_results.json'
    rows = adapters.read_json(path)
    rows = [r for r in rows if not (r['replica_id']=='replica-000000' and r['variant']=='neural')]
    write(path, rows)
    report = adapters.validate_source('information', root)
    rate = next(r for r in tables.normalized_tables(report)['accuracy'] if r['metric']=='false_positive_rate')
    assert rate['n_evaluated'] == 1 and rate['n_planned'] == 2 and rate['n_missing'] == 1
    assert rate['rate'] == 1. and rate['planned_rate_bounds'] == [.5, 1.]
    assert not report['complete']


def test_semantic_pvalue_and_temporal_timestamp_audit(source_factory):
    root, p = source_factory('transfer')
    rd = root / p['cells'][0]['cell_id'] / p['cells'][0]['allocation'][0]['replica_id']
    file = rd / 'X_to_Y-surrogates.json'; value = adapters.read_json(file)
    value['p_values'][0] = .001; write(file, value)
    mp = rd / 'manifest.json'; manifest = adapters.read_json(mp); manifest['artifact_sha256'] = hashes(rd); write(mp, manifest)
    with pytest.raises(ValueError, match='Inference'): adapters.validate_source('transfer', root)
    value['p_values'][0] = 1.; write(file, value)
    diagnostic = next(rd.glob('*/diagnostics.json')); data = adapters.read_json(diagnostic)
    data['partitions'][2]['observation_timestamps'][0][1] = data['partitions'][2]['target_timestamps'][0]
    write(diagnostic, data); manifest['artifact_sha256'] = hashes(rd); write(mp, manifest)
    with pytest.raises(ValueError, match='leakage'): adapters.validate_source('transfer', root)


def test_adjusted_confirmatory_numeric_figure_fixture(source_factory, tmp_path, monkeypatch):
    from src.metrics.confirmatory import paired_intervals
    from experiments.final_protocol import cells, PRIMARY, REPLICAS
    from experiments.neural_baseline import VARIANTS
    rows = [{'campaign_id': c['campaign_id'], 'replica_id': rid, 'variant': v, 'status': 'complete',
             'entropy_nats': .125 if v=='njee' else .25, 'ground_truth_nats': 0.,
             'signed_error_nats': .125 if v=='njee' else .25}
            for c in cells() if c['cell'] in PRIMARY for rid in REPLICAS for v in VARIANTS]
    intervals = paired_intervals(rows)
    assert all(r['lower_nats']==r['upper_nats']==-.125 and r['n_pairs']==30 for r in intervals)
    recipe = next(r for r in figures.recipes('entropy') if r.name=='confirmatory-adjusted')
    monkeypatch.setattr(figures, 'recipes', lambda e: [recipe])
    tableset = {name: [] for name in tables.TABLE_NAMES}; tableset['confirmatory'] = intervals
    directory = tmp_path / 'ARTIFICIAL-CI'; directory.mkdir()
    index = figures.generate_figures(directory, tableset, 'entropy', mode='diagnostic', artificial=True, dpi=150)
    assert len(index)==4 and all('99.6875%' in f['caption'] for f in index)
    assert all(p['interval']==(-.125,-.125) for f in index for p in f['points'])
    with pytest.raises(FileExistsError): figures.generate_figures(directory, tableset, 'entropy', mode='diagnostic', artificial=True, dpi=150)


def test_duplicate_json_keys_and_manifest_metadata_tamper(tmp_path):
    path = tmp_path / 'invalid.json'; path.write_text('{"v":1,"v":2}')
    with pytest.raises(ValueError, match='Duplicate'): adapters.read_json(path)
    root = tmp_path / 'ARTIFICIAL-MANIFEST'; root.mkdir()
    write(root / 'report_manifest.json', {'schema': pipeline.SCHEMA, 'manifest_digest': 'invalid'})
    with pytest.raises(ValueError, match='checksum'): pipeline.verify_report(root)
