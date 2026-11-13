import argparse
from datetime import datetime, timezone
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {
    'entropy': 'final_validation',
    'information': 'information_validation',
    'transfer': 'transfer_validation',
    'ablation': 'ablation_validation',
}
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
TOOLS = ('git', 'bash', 'ssh', 'scp', 'rsync', 'sbatch', 'srun', 'qsub', 'bsub')
PAYLOAD_SUFFIXES = {'.json', '.jsonl', '.yaml', '.yml', '.csv', '.npz', '.log', '.md', '.tex', '.svg', '.png'}
SENSITIVE_NAME = re.compile(r'credential|secret|token|password|private.key|^id_(rsa|ed25519|ecdsa)', re.I)


def write_new(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)


def _run(args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', check=True).stdout.strip()


def checkout(expected=None, *, clean_required=False):
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from the repository root.')
    top = Path(_run(['git', 'rev-parse', '--show-toplevel'])).resolve()
    commit = _run(['git', 'rev-parse', 'HEAD'])
    if top != ROOT or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Unexpected Git checkout.')
    dirty = bool(_run(['git', 'status', '--porcelain', '--untracked-files=all']))
    if expected is not None and commit != expected:
        raise ValueError('Checkout differs from the requested commit.')
    if clean_required and (expected is None or dirty):
        raise ValueError('Linux validation requires an explicit commit and a clean checkout.')
    return {'commit': commit, 'dirty': dirty}


def package_versions(*, target_linux=False, include_tests=False):
    path = ROOT / ('requirements-linux-test.txt' if include_tests else 'requirements-linux.txt')
    pins = (ROOT / 'requirements-linux.txt').read_text(encoding='utf-8').splitlines()
    if include_tests:
        pins.extend(path.read_text(encoding='utf-8').splitlines())
    wanted = dict(line.split('==', 1) for line in pins if '==' in line)
    wanted['torch'] = '2.14.0+cpu' if target_linux else '2.14.0'
    versions = {}
    for name, pin in wanted.items():
        actual = metadata.version(name)
        comparable = actual.split('+', 1)[0] if name == 'torch' and not target_linux else actual
        if comparable != pin:
            raise ValueError(f'Unexpected installed version for {name}: {actual}; expected {pin}.')
        versions[name] = actual
    return versions


def settings(env, *, required=False):
    result = {}
    for name in THREADS:
        value = env.get(name)
        if value is not None and not re.fullmatch(r'[1-9][0-9]{0,5}', value):
            raise ValueError(f'{name} must be a positive integer.')
        if required and value is None:
            raise ValueError(f'Explicit thread setting required: {name}.')
        result[name] = int(value) if value is not None else None
    for name, expected in {'MPLBACKEND': 'Agg', 'PYTHONUTF8': '1', 'PYTHONHASHSEED': '0'}.items():
        value = env.get(name)
        if required and value != expected:
            raise ValueError(f'{name} must be {expected}.')
        # Record compliance, never arbitrary environment values.
        result[name + '_matches'] = value == expected
    return result


def artifact_smoke(parent=None):
    """Isolated fixed-value CPU, SciPy, PNG/SVG, split and NPZ round trips."""
    import numpy as np
    from scipy.linalg import solve
    import torch
    from matplotlib.figure import Figure
    from src.utils.split import load_split, make_iid_split, save_split

    if torch.from_numpy(np.array([1., 2.])).sum().item() != 3.:
        raise ValueError('PyTorch CPU / NumPy interoperability failed.')
    if not np.array_equal(solve(np.eye(2), np.array([1., 2.])), [1., 2.]):
        raise ValueError('SciPy numerical compatibility failed.')
    if parent is not None and not Path(parent).is_dir():
        raise ValueError('Scratch parent must be an existing directory.')
    with TemporaryDirectory(prefix='njee-preflight-', dir=parent) as scratch:
        path = Path(scratch)
        split = make_iid_split(10, seed=1)
        save_split(split, path / 'split.npz')
        if not np.array_equal(load_split(path / 'split.npz').test, split.test):
            raise ValueError('Split round trip failed.')
        np.savez_compressed(path / 'weights.npz', weight=np.array([1., 2.]))
        with np.load(path / 'weights.npz', allow_pickle=False) as arrays:
            if not np.array_equal(arrays['weight'], [1., 2.]):
                raise ValueError('Numeric checkpoint round trip failed.')
        fig = Figure(figsize=(1, 1))
        fig.subplots().plot([0, 1], [0, 1])
        for suffix in ('png', 'svg'):
            fig.savefig(path / ('fixture.' + suffix))
    return {'cpu_numpy_scipy': True, 'numeric_npz_and_split': True, 'headless_svg_png': True,
            'isolated_write_check': True}


def preflight(*, expected=None, target_linux=False, include_tests=False, scratch=None, bundles=None):
    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 13):
        raise ValueError('Use CPython 3.13.')
    system, machine = platform.system(), platform.machine()
    if target_linux and (system != 'Linux' or machine.lower() not in ('x86_64', 'amd64')):
        raise ValueError('Target profile requires Linux x86_64.')
    git = checkout(expected, clean_required=target_linux)
    configured = settings(os.environ, required=target_linux)
    versions = package_versions(target_linux=target_linux, include_tests=include_tests)
    import torch
    if target_linux and (torch.version.cuda is not None or torch.version.hip is not None):
        raise ValueError('Expected a CPU-only PyTorch build.')
    if target_linux:
        if not sys.flags.utf8_mode or torch.get_num_threads() != configured['OMP_NUM_THREADS']:
            raise ValueError('Effective UTF-8 or PyTorch thread setting differs from the target profile.')
        _run([sys.executable, '-m', 'pip', 'check'])
    imported = []
    for path in sorted((ROOT / 'src').rglob('*.py')):
        parts = path.relative_to(ROOT).with_suffix('').parts
        if parts[-1] == '__init__':
            parts = parts[:-1]
        name = '.'.join(parts)
        import_module(name)
        imported.append(name)
    if not callable(import_module('experiments.scientific_report').main):
        raise ValueError('Missing reporting CLI entry point.')
    from src.reporting.adapters import frozen_plan
    protocols = {}
    for experiment, runner in RUNNERS.items():
        module = import_module('experiments.' + runner)
        if not callable(module.main):
            raise ValueError('Missing CLI entry point.')
        protocol = import_module('experiments.' + runner.replace('_validation', '_protocol'))
        if hasattr(protocol, 'load_definition'):
            protocol.load_definition()
        plan = frozen_plan(experiment)
        bundle = (bundles or {}).get(experiment)
        if bundle is not None and protocol.validate_bundle(bundle)['protocol_hash'] != plan['protocol_hash']:
            raise ValueError('Preparation bundle hash differs from frozen protocol.')
        protocols[experiment] = {'hash': plan['protocol_hash'], 'expected': plan['expected'],
                                 'bundle_checked': bundle is not None, 'cli': module.__name__}
    return {'schema': 'njee-deployment-preflight-v1', 'checked_utc': datetime.now(timezone.utc).isoformat(),
            'profile': 'linux_target' if target_linux else 'offline',
            'platform': {'system': system, 'machine': machine, 'python': platform.python_version(),
                         'libc': list(platform.libc_ver()) if system == 'Linux' else None},
            'git': git, 'packages': versions, 'threads': configured,
            'tools_present': {name: shutil.which(name) is not None for name in TOOLS},
            'imports': imported, 'protocols': protocols, 'formats': artifact_smoke(scratch),
            'launch_authorized': False}


def inventory(root):
    """Inventory an explicitly selected, reviewed artifact tree."""
    root = Path(root)
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        raise ValueError('Expected a real artifact directory.')
    files, names = {}, set()
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if (path.is_symlink() or path.is_junction()
                or any(part.startswith('.') or SENSITIVE_NAME.search(part)
                       or re.search(r'[<>:"\\|?*\x00-\x1f]', part)
                       or part.endswith((' ', '.'))
                       or re.fullmatch(r'CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]', part.split('.')[0], re.I)
                       for part in relative.parts)):
            raise ValueError('Unsafe artifact name or link.')
        name = relative.as_posix()
        if name.casefold() in names:
            raise ValueError('Artifact names collide on a case-insensitive filesystem.')
        names.add(name.casefold())
        if path.is_dir():
            continue
        mode = path.stat()
        if not stat.S_ISREG(mode.st_mode) or mode.st_nlink != 1 or path.suffix.lower() not in PAYLOAD_SUFFIXES:
            raise ValueError('Unsupported artifact type or hard link.')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = path.stat()
        if (mode.st_size, mode.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Artifact changed during inventory.')
        files[name] = {'bytes': mode.st_size, 'sha256': digest}
    if not files:
        raise ValueError('Artifact inventory is empty.')
    return {'schema': 'njee-transfer-inventory-v1', 'files': files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('preflight')
    check.add_argument('--expected-commit')
    check.add_argument('--target-linux', action='store_true')
    check.add_argument('--include-tests', action='store_true')
    check.add_argument('--scratch-parent', type=Path, required=True)
    check.add_argument('--manifest', type=Path)
    for name in RUNNERS:
        check.add_argument('--' + name + '-bundle', type=Path)
    create = commands.add_parser('inventory')
    verify = commands.add_parser('verify')
    for command in (create, verify):
        command.add_argument('root', type=Path)
        command.add_argument('manifest', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'preflight':
        if args.manifest is not None and (args.manifest.exists() or args.manifest.is_symlink()):
            raise FileExistsError('Preflight manifest already exists.')
        report = preflight(expected=args.expected_commit, target_linux=args.target_linux,
                           include_tests=args.include_tests, scratch=args.scratch_parent,
                           bundles={name: getattr(args, name + '_bundle') for name in RUNNERS})
        if args.manifest is not None:
            write_new(args.manifest, report)
    else:
        if args.manifest.resolve().is_relative_to(args.root.resolve()):
            raise ValueError('Keep the transfer manifest outside its payload.')
        report = inventory(args.root)
        if args.command == 'inventory':
            write_new(args.manifest, report)
        elif json.loads(args.manifest.read_text(encoding='utf-8')) != report:
            raise ValueError('Transferred artifact inventory differs.')
        else:
            report = {'status': 'verified', 'files': len(report['files'])}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
