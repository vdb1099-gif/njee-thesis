"""Deterministic deployment checks with no server or scientific workload."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import deployment as d


def test_checkout_and_clean_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(d, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    answers = {'--show-toplevel': str(tmp_path), 'HEAD': 'a'*40, '--untracked-files=all': ''}
    monkeypatch.setattr(d, '_run', lambda argv: answers[argv[-1]])
    assert d.checkout('a'*40, clean_required=True) == {'commit': 'a'*40, 'dirty': False}
    with pytest.raises(ValueError, match='differs'):
        d.checkout('b'*40)
    answers['--untracked-files=all'] = '?? private-local-name'
    assert d.checkout()['dirty']
    with pytest.raises(ValueError, match='clean'):
        d.checkout('a'*40, clean_required=True)


def test_allowlisted_settings_exclude_secrets():
    env = dict.fromkeys(d.THREADS, '2')
    env.update(MPLBACKEND='Agg', PYTHONUTF8='1', PYTHONHASHSEED='0', API_TOKEN='PRIVATE')
    settings = d.settings(env, required=True)
    assert settings['OMP_NUM_THREADS'] == 2
    assert 'PRIVATE' not in json.dumps(settings)
    env['OMP_NUM_THREADS'] = '2; other command'
    with pytest.raises(ValueError, match='positive integer'):
        d.settings(env, required=True)


def test_preflight_config_and_missing_schedulers(monkeypatch, tmp_path):
    import src.reporting.adapters as adapters
    monkeypatch.setattr(d, 'ROOT', tmp_path)
    monkeypatch.setattr(d, 'checkout', lambda *a, **k: {'commit': 'a'*40, 'dirty': False})
    monkeypatch.setattr(d, 'package_versions', lambda **k: {'torch': '2.14.0'})
    monkeypatch.setattr(d.shutil, 'which', lambda name: None)
    monkeypatch.setattr(d, 'artifact_smoke', lambda path: {'fixture': True})
    monkeypatch.setattr(adapters, 'frozen_plan', lambda name: {'protocol_hash': name, 'expected': {'cells': 1}})
    checked = []
    def fake_module(name):
        if name == 'experiments.scientific_report':
            return SimpleNamespace(main=lambda: pytest.fail('Reporter called'))
        kind = next(k for k, v in d.RUNNERS.items() if v.split('_')[0] == name.split('.')[1].split('_')[0])
        return SimpleNamespace(main=lambda: pytest.fail('Runner called'), __name__=name,
                               load_definition=lambda: checked.append(kind),
                               validate_bundle=lambda path: {'protocol_hash': kind})
    monkeypatch.setattr(d, 'import_module', fake_module)
    report = d.preflight(bundles={'entropy': tmp_path / 'bundle with spaces'})
    assert len(checked) == 4 and not report['tools_present']['sbatch']
    assert report['protocols']['entropy']['bundle_checked']
    assert report['launch_authorized'] is False
    json.dumps(report, allow_nan=False)
    monkeypatch.setattr(adapters, 'frozen_plan', lambda name: {'protocol_hash': 'other', 'expected': {}})
    with pytest.raises(ValueError, match='bundle hash'):
        d.preflight(bundles={'entropy': tmp_path})


def test_target_rejects_wrong_platform_before_checkout(monkeypatch):
    monkeypatch.setattr(d.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(d, 'checkout', lambda *a, **k: pytest.fail('Git checked before platform gate'))
    with pytest.raises(ValueError, match='Linux x86_64'):
        d.preflight(target_linux=True)


def test_scratch_format_round_trips_are_isolated(tmp_path):
    assert d.artifact_smoke(tmp_path)['numeric_npz_and_split']
    assert not list(tmp_path.iterdir())


def test_artifact_transfer_tamper_no_overwrite_and_no_source_changes(tmp_path):
    root = tmp_path / 'reviewed results'
    root.mkdir()
    nested = root / 'cell 01'
    nested.mkdir()
    record = nested / 'raw_results.json'
    record.write_bytes(b'{}')
    manifest = tmp_path / 'transfer inventory.json'
    d.main(['inventory', str(root), str(manifest)])
    first = manifest.read_bytes()
    assert list(json.loads(first)['files']) == ['cell 01/raw_results.json']
    d.main(['verify', str(root), str(manifest)])
    assert record.read_bytes() == b'{}' and manifest.read_bytes() == first
    with pytest.raises(FileExistsError):
        d.main(['inventory', str(root), str(manifest)])
    with pytest.raises(ValueError, match='outside'):
        d.main(['inventory', str(root), str(root / 'inventory.json')])
    record.write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs'):
        d.main(['verify', str(root), str(manifest)])


@pytest.mark.parametrize('name', ['.env', 'credentials.json', 'model.pkl', 'payload.exe'])
def test_rejects_unreviewed_artifact_types_and_sensitive_names(tmp_path, name):
    (tmp_path / name).write_text('fixture')
    with pytest.raises(ValueError):
        d.inventory(tmp_path)


def test_cli_has_no_scientific_launch_action():
    with pytest.raises(SystemExit):
        d.main(['execute', '--authorize-execution'])
