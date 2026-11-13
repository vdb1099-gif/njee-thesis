"""Frozen matrix, effective seed, pairing, safe controller and failure fixtures."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import ablation_protocol as protocol, ablation_validation as runner
from src.metrics.ablation import summarize
from src.utils.temporal import build_temporal_embedding, make_chronological_split


@pytest.fixture(scope='module')
def full_plan():
    return protocol.preview()


def test_frozen_matrix_ofat_orders_counts_and_effective_seeds(full_plan):
    p = full_plan
    assert p['protocol_hash'] == '878571f73c32fe166b5569ce113e6be6201df6ae21666266428598f5d8854eb1'
    assert p['expected'] == {'cells': 39, 'replicas_per_cell': 30, 'independent_datasets_or_processes': 120,
        'cell_replica_records': 1170, 'entropy_fits': 990, 'te_fits': 180, 'conditional_heads': 5310,
        'shared_trunk_fits': 60, 'surrogate_statistics': 0}
    assert protocol.load_definition() == protocol.definition()
    assert len(set(protocol.orders())) == 6 and protocol.orders()[0] == tuple(range(6))
    assert protocol.orders() == protocol.orders()
    for c in p['cells']:
        if c['group'] == 'independent' and c['name'] not in ('reference', 'shared', 'sharing-control'):
            ref = next(x for x in p['cells'] if x['group'] == c['group'] and x['name'] == 'reference')
            changed = {k for k in ('order', 'hidden_sizes', 'train_fraction', 'early_stopping', 'states', 'variant') if c[k] != ref[k]}
            assert len(changed) == 1
    seeds = []
    for g in p['groups']:
        for r in g['allocation']:
            for role in ('data', 'split', 'subset'):
                seeds.append(r[role]['seed'])
            for fit in r['fits'].values():
                seeds.extend([fit['seed'], *fit['effective_seeds']])
    assert len(seeds) == len(set(seeds)) and not set(seeds) & protocol.reserved_seeds()
    assert not Path(p['output_root']).exists()


@pytest.fixture
def tiny(monkeypatch):
    d = deepcopy(protocol.definition())
    d.update(n=40, replicas=2, bootstrap_draws=3)
    monkeypatch.setattr(protocol, 'definition', lambda: deepcopy(d))
    return protocol.preview()


def test_preparation_hashes_no_overwrite_and_portable_paths(tiny, tmp_path):
    directory = tmp_path / 'prep'
    protocol.materialize(directory)
    assert protocol.validate_bundle(directory) == tiny
    assert len(list(directory.iterdir())) == 40
    with pytest.raises(FileExistsError): protocol.materialize(directory)
    file = next(directory.glob('*.yaml')); file.write_text('{}')
    with pytest.raises(ValueError): protocol.validate_bundle(directory)
    for bad in ('../escape', '/private', 'C:/private', 'bad\\path'):
        with pytest.raises(ValueError): protocol.preview(bad)


def mock_fits(monkeypatch):
    calls = []
    def fit(train, validation, test, **kwargs):
        assert not set(train.row_indices) & (set(validation.row_indices) | set(test.row_indices))
        calls.append((train, validation, test, kwargs))
        from src.estimators.ablation import stream_ids
        from src.utils.reproducibility import derive_term_seed
        config = kwargs['config']
        effective = [derive_term_seed(config.training_seed, tid, purpose=p)
                     for tid in stream_ids(config.component_order, kwargs['variant'])
                     for p in ('initialization', 'shuffle')]
        return {'entropy_nats': -.25, 'parameter_count': 10, 'optimizer_updates': 2,
                'term_batch_evaluations': 2, 'effective_seeds': effective, 'terms': []}, {}
    monkeypatch.setattr(runner, 'estimate_entropy', fit)
    def temporal(values, cell, seed, identity):
        from experiments.transfer_protocol import effective_streams
        return {'estimate_nats': -.1, 'truth_nats': 0., 'effective_seeds': effective_streams(cell, seed)[0],
                'diagnostics': {'cmi_report': {'components': []}}, 'parameter_count': 10, 'optimizer_updates': 2,
                'term_batch_evaluations': 2, 'n_train': 20, 'alignment_support': [[3, 4], [20], [30]]}, {}
    monkeypatch.setattr(runner, 'fit_temporal', temporal)
    def generate(group, seed):
        if group == 'discretization': return np.arange(240).reshape(40, 6) / 240, {'fixture': True}
        return np.arange(240).reshape(40, 6) % 2, {'fixture': True}
    monkeypatch.setattr(runner, 'generate_group', generate)
    monkeypatch.setattr(runner, 'environment', lambda: {'fixture': True})
    monkeypatch.setattr(runner, '_git_provenance', lambda: {'dirty': False, 'commit': 'fixture'})
    return calls


def setup(tiny, tmp_path, monkeypatch):
    calls = mock_fits(monkeypatch)
    prep, output = tmp_path / 'prep', tmp_path / 'out'
    protocol.materialize(prep)
    return prep, output, calls


def test_mock_controller_pairing_subsets_reports_integrity(tiny, tmp_path, monkeypatch):
    prep, output, calls = setup(tiny, tmp_path, monkeypatch)
    runner.execute(prep, authorized=True, allow_local_windows=True, output_dir=output)
    assert len(calls) == 66
    first = calls[:15]
    assert len({a.dataset_id for a, _, _, _ in first}) == 1
    assert len({a.split_id for a, _, _, _ in first}) == 1
    assert all(np.array_equal(v.row_indices, first[0][1].row_indices) for _, v, _, _ in first)
    assert all(np.array_equal(t.row_indices, first[0][2].row_indices) for _, _, t, _ in first)
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    result = runner.inspect_results(output)
    assert result['manifest']['status'] == 'complete' and len(result['raw_records']) == 78
    assert all(r['n_complete'] == 2 for r in result['analysis']['summaries'])
    assert before == {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError): runner.execute(prep, authorized=True, allow_local_windows=True, output_dir=output)
    next(output.rglob('diagnostics.json')).write_text('{}')
    with pytest.raises(ValueError, match='hash'): runner.inspect_results(output)


def test_failures_no_retry_and_incomplete_pairs(tiny, tmp_path, monkeypatch):
    prep, output, _ = setup(tiny, tmp_path, monkeypatch)
    attempts = []
    def fail(*args, **kw): attempts.append(1); raise RuntimeError('fit failure')
    monkeypatch.setattr(runner, 'estimate_entropy', fail)
    runner.execute(prep, authorized=True, allow_local_windows=True, output_dir=output)
    assert len(attempts) == 66
    result = runner.inspect_results(output)
    assert result['manifest']['status'] == 'complete_with_failures'
    assert all(r['n_complete'] == 0 and r['n_failed'] == 2 for r in result['analysis']['summaries'][:33])
    assert result['analysis']['paired_comparisons'][0]['n_incomplete_pairs'] == 2


def test_interruption_and_generation_failure_accounting(tiny, tmp_path, monkeypatch):
    prep, output, _ = setup(tiny, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'estimate_entropy', lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt('stop')))
    with pytest.raises(KeyboardInterrupt): runner.execute(prep, authorized=True, allow_local_windows=True, output_dir=output)
    result = runner.inspect_results(output)
    assert result['manifest']['status'] == 'interrupted'
    assert sum(r['status'] == 'interrupted' for r in result['raw_records']) == 1
    assert sum(r['status'] == 'not_attempted' for r in result['raw_records']) == 77
    mock_fits(monkeypatch)
    attempts = []
    def fail(group, seed): attempts.append((group, seed)); raise RuntimeError('data failed')
    monkeypatch.setattr(runner, 'generate_group', fail)
    second = tmp_path / 'second'
    runner.execute(prep, authorized=True, allow_local_windows=True, output_dir=second)
    assert len(attempts) == len(set(attempts)) == 8
    assert all(r['status'] == 'failed' for r in runner.inspect_results(second)['raw_records'])


def test_safety_default_auth_cancel_and_dirty_tree(tiny, tmp_path, monkeypatch, capsys):
    prep, output, _ = setup(tiny, tmp_path, monkeypatch)
    monkeypatch.setattr(protocol, 'load_definition', lambda: protocol.definition())
    monkeypatch.setattr(runner, 'run_group_replica', lambda *a, **k: pytest.fail('unauthorized fit'))
    runner.main([])
    assert '1170' not in capsys.readouterr().out  # tiny plan is metadata only
    with pytest.raises(PermissionError): runner.execute(prep)
    with pytest.raises(SystemExit): runner.main(['--dry-run', '--authorize-execution'])
    monkeypatch.setattr(runner, '_git_provenance', lambda: {'dirty': True})
    with pytest.raises(ValueError): runner.execute(prep, authorized=True, allow_local_windows=True, output_dir=output)
    assert not output.exists()
    monkeypatch.setattr(runner, '_git_provenance', lambda: {'dirty': False})
    cancel = tmp_path / 'cancel'; cancel.touch()
    with pytest.raises(KeyboardInterrupt): runner.execute(prep, authorized=True, allow_local_windows=True, output_dir=output, cancel_file=cancel)
    assert all(r['status'] == 'not_attempted' for r in runner.inspect_results(output)['raw_records'])


def test_temporal_alignment_real_contract_mock_neural_fit(monkeypatch):
    seen = []
    class TE:
        def __init__(self, config, *, embedding, source_id, target_id, **kw):
            self.config, self.spec, self.sid, self.tid, self.kw = config, embedding, source_id, target_id, kw
        def fit(self, source, target, **kw):
            e = build_temporal_embedding(source, target, spec=self.spec, source_id=self.sid, target_id=self.tid, **kw)
            self._parts = make_chronological_split(e, gap=self.kw['gap'], purge=self.kw['purge'])
            self._model = SimpleNamespace(_components=[])
            seen.append(self)
        def estimate(self):
            return SimpleNamespace(transfer_entropy_nats=-.1, partitions=self._parts.provenance,
                                   cmi_report=SimpleNamespace(components=[]))
    monkeypatch.setattr(runner, 'NeuralTransferEntropyEstimator', TE)
    monkeypatch.setattr(runner, 'reference_te', lambda *a, **k: {'transfer_entropy_nats': 0.})
    values = np.arange(200).reshape(100, 2) % 2
    results = [runner.fit_temporal(values, c, 42, 'identity')[0] for c in protocol.cells() if c['kind'] == 'te']
    assert all(r['alignment_support'] == results[0]['alignment_support'] for r in results)
    assert len(seen) == 6
    for model in seen:
        assert model._parts.train.n_samples == 100 - 9
        for left, right in zip(model._parts.provenance, model._parts.provenance[1:]):
            assert min(min(r) for r in right.observation_timestamps) > left.target_timestamps[-1]


def test_aggregation_signed_errors_pairs_and_no_pseudoreplication(tiny):
    rows = runner.empty_rows(tiny)
    for i, r in enumerate(rows):
        estimate = -.5 if r['replica_id'].endswith('0') else .5
        r.update(status='complete', estimate_nats=estimate, truth_nats=0., signed_error_nats=estimate,
                 runtime_seconds=1., parent_dataset_id=r['group']+r['replica_id'], parent_split_id='s')
    result = summarize(rows, tiny)
    assert result['summaries'][0]['metrics']['bias_nats'] == 0
    assert result['summaries'][0]['metrics']['rmse_nats'] == .5
    assert result['paired_comparisons'][0]['n_pairs'] == 2
    assert len(result['ordering_variability']) == 4
    rows[0]['status'] = 'failed'
    assert summarize(rows, tiny)['paired_comparisons'][0]['n_pairs'] == 1
    with pytest.raises(ValueError): summarize(rows[:-1], tiny)


def test_nonce_collision_and_manifest_tampering(tiny, tmp_path, monkeypatch):
    first = tiny['groups'][0]['allocation'][0]['data']['seed']
    old = protocol.reserved_seeds()
    monkeypatch.setattr(protocol, 'reserved_seeds', lambda: old | {first})
    assert protocol.preview()['groups'][0]['allocation'][0]['data']['nonce'] > 0
    directory = tmp_path / 'prep'; protocol.materialize(directory)
    path = directory / 'protocol.json'; data = json.loads(path.read_text()); data['protocol_hash'] = 'bad'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): protocol.validate_bundle(directory)


def test_seed_mismatch_and_reporting_failure_not_published_complete(tiny, tmp_path, monkeypatch):
    prep, output, _ = setup(tiny, tmp_path, monkeypatch)
    def bad_fit(*a, **k):
        return {'entropy_nats': 0., 'parameter_count': 0, 'optimizer_updates': 0,
                'term_batch_evaluations': 0, 'effective_seeds': [], 'terms': []}, {}
    monkeypatch.setattr(runner, 'estimate_entropy', bad_fit)
    runner.execute(prep, authorized=True, allow_local_windows=True, output_dir=output)
    result = runner.inspect_results(output)
    assert result['manifest']['status'] == 'complete_with_failures'
    assert all(r['status'] == 'failed' for r in result['raw_records'] if r['group'] != 'temporal')
    mock_fits(monkeypatch)
    monkeypatch.setattr(runner, 'export_tables', lambda *a: (_ for _ in ()).throw(RuntimeError('export failed')))
    second = tmp_path / 'second'
    with pytest.raises(RuntimeError, match='export failed'):
        runner.execute(prep, authorized=True, allow_local_windows=True, output_dir=second)
    assert json.loads((second / 'manifest.json').read_text())['status'] == 'reporting_failed'


def test_windows_exception_and_log_overlap_guards(tiny, tmp_path, monkeypatch):
    prep, output, _ = setup(tiny, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'os', SimpleNamespace(name='nt'))
    with pytest.raises(PermissionError, match='Windows'):
        runner.execute(prep, authorized=True, output_dir=output)
    with pytest.raises(ValueError, match='log'):
        runner.execute(prep, authorized=True, allow_local_windows=True, output_dir=output, log_dir=tmp_path)
    assert not output.exists()


def test_environment_preflight_failure_creates_no_output(tiny, tmp_path, monkeypatch):
    prep, output, _ = setup(tiny, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'environment', lambda: (_ for _ in ()).throw(RuntimeError('environment unavailable')))
    with pytest.raises(RuntimeError, match='environment unavailable'):
        runner.execute(prep, authorized=True, allow_local_windows=True, output_dir=output)
    assert not output.exists()
