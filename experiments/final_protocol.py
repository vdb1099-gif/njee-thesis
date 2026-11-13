from copy import deepcopy
import hashlib
import json
from pathlib import Path, PureWindowsPath

import yaml

from experiments.neural_campaign import derive_replica_seeds, load_campaign_config
from experiments.neural_baseline import VARIANTS, _run_id
from src.simulation.factory import get_ground_truth
from src.utils.neural_config import validate_neural_config

DEFAULT_OUTPUT = 'results/task013c_final_validation_20261106'
PRIMARY = ('F01', 'F02', 'F17', 'F18')
ORDER = PRIMARY + tuple(f'F{i:02d}' for i in range(3, 17)) + tuple(f'F{i:02d}' for i in range(19, 28))
REPLICAS = tuple(f'replica-{i:06d}' for i in range(30))
BUDGET = 18000
SHUTDOWN_RESERVE = 60
MIN_CAMPAIGN_WINDOW = 60


def digest(value):
    """Semantic hashes ignore YAML formatting, platform newlines and root paths."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def relative_path(value):
    """Bundle paths are portable, relative to the explicit execution workdir."""
    value = str(value)
    path = Path(value)
    if (not value or value == '.' or value.startswith('/') or path.is_absolute() or PureWindowsPath(value).drive
            or '\\' in value or '..' in path.parts):
        raise ValueError('Use a nonempty portable relative path without .. or backslashes.')
    return path.as_posix()


def cells():
    matrix = []
    for generator, parameter, value in (('independent_bernoulli', 'p', .5), ('binary_chain', 'epsilon', .15)):
        for d in (4, 8, 12):
            for n in (200, 800, 3200):
                matrix.append((generator, parameter, value, d, n))
    for epsilon in (.05, .30):
        for d in (4, 8, 12):
            matrix.append(('binary_chain', 'epsilon', epsilon, d, 800))
    for d in (4, 8, 12):
        matrix.append(('independent_bernoulli', 'p', .2, d, 800))
    result = []
    for i, (generator, parameter, value, d, n) in enumerate(matrix):
        key = f'F{i+1:02d}'
        label = ('ind-p' if parameter == 'p' else 'chain-e') + f'{round(value*100):03d}'
        result.append({'cell': key, 'campaign_id': f'{key}-{label}-d{d:02d}-n{n:05d}',
                       'master_seed': 130100+i, 'data': {'generator': generator, 'd': d, 'n': n,
                                                       'params': {parameter: value}}})
    return result


def preview(output_root=DEFAULT_OUTPUT):
    """Return all configurations, allocation and hashes, entirely in memory."""
    output_root = relative_path(output_root)
    entries = []
    role_seeds, term_seeds = [], []
    # The derivation utility is deterministic and does not construct a model.
    from src.utils.reproducibility import derive_term_seed
    for cell in cells():
        cid, d = cell['campaign_id'], cell['data']['d']
        replicas = [{'replica_id': rid, **derive_replica_seeds(cell['master_seed'], i)}
                    for i, rid in enumerate(REPLICAS)]
        neural = {'split_seed': replicas[0]['split_seed'], 'split_fractions': [.6, .2, .2],
                  'training_seed': replicas[0]['training_seed'], 'component_order': list(range(d)),
                  'alphabet_policy': 'declared', 'marginal_method': 'train_plugin', 'hidden_sizes': [16],
                  'activation': 'relu', 'optimizer': 'adam', 'learning_rate': .01, 'batch_size': 64,
                  'max_epochs': 20, 'patience': 5, 'min_delta': .0001, 'device': 'cpu',
                  'reproducibility_mode': 'strict', 'num_workers': 0}
        validate_neural_config(neural, variable_ids=tuple(range(d)), n_samples=cell['data']['n'])
        baseline = {'seed': replicas[0]['data_seed'], 'data': deepcopy(cell['data']), 'neural': neural,
                    'output_dir': output_root, 'run_id': cid}
        campaign = {'campaign_id': cid, 'purpose': 'final', 'n_replicas': 30,
                    'master_seed': cell['master_seed'], 'baseline_config': f'{cid}-baseline.yaml',
                    'output_dir': output_root, 'aggregation': {'bootstrap_seed': 271813, 'bootstrap_draws': 10000,
                    'paired_comparisons': [['njee', v] for v in VARIANTS[1:]]}}
        _run_id(cid)
        allocation = []
        for replica in replicas:
            effective = []
            role_seeds.extend(replica[k] for k in ('data_seed', 'split_seed', 'training_seed'))
            for j in range(1, d):
                tid = json.dumps(['njee-conditional-term-v1', j, list(range(j))], separators=(',', ':'))
                seeds = {p: derive_term_seed(replica['training_seed'], tid, purpose=p)
                         for p in ('initialization', 'shuffle')}
                term_seeds.extend(seeds.values())
                effective.append({'target': j, 'conditioning': list(range(j)), **seeds})
            allocation.append({**replica, 'terms': effective})
        scientific = {'data': cell['data'], 'neural': {k: v for k, v in neural.items()
                                                     if k not in ('split_seed', 'training_seed')}}
        entries.append({**cell, 'baseline': baseline, 'campaign': campaign, 'replicas': allocation,
                        'ground_truth_nats': float(get_ground_truth(baseline)),
                        'baseline_hash': digest(baseline), 'campaign_hash': digest(campaign),
                        'scientific_hash': digest(scientific)})
    if len(set(role_seeds + term_seeds)) != len(role_seeds + term_seeds):
        raise ValueError('Frozen effective seed collision; stop for protocol review.')
    scientific = [{'cell': e['cell'], 'scientific_hash': e['scientific_hash'],
                   'master_seed': e['master_seed'], 'replicas': e['replicas']} for e in entries]
    plan = {'schema': 'njee-final-protocol-v1', 'output_root': output_root, 'execution_order': list(ORDER),
            'budget_seconds': BUDGET, 'shutdown_reserve_seconds': SHUTDOWN_RESERVE,
            'minimum_campaign_window_seconds': MIN_CAMPAIGN_WINDOW,
            'expected': {'cells': 27, 'replicas': 810, 'conditional_fits': 5670,
                         'chain_contributions': 6480, 'estimator_records': 4050},
            'protocol_hash': digest(scientific), 'cells': entries}
    # Historical allocation metadata only; no prior observations or outcomes read.
    prior = []
    historical = [(120010,4,3),(120011,4,3),(120012,4,3),(120013,4,3),(120014,12,3),
                  (130010,4,8),(130011,4,8),(130012,12,8),(130013,12,8)]
    for master, d, count in historical:
        for i in range(count):
            seeds = derive_replica_seeds(master, i)
            prior.extend(seeds.values())
            for j in range(1, d):
                tid = json.dumps(['njee-conditional-term-v1', j, list(range(j))], separators=(',', ':'))
                prior.extend(derive_term_seed(seeds['training_seed'], tid, purpose=p)
                             for p in ('initialization', 'shuffle'))
    plan['historical_seed_check'] = check_prior_seeds(plan, prior)
    return plan


def check_prior_seeds(plan, prior_allocations=()):
    """Caller supplies metadata only; never open pilot test-result files here."""
    final = {s for e in plan['cells'] for r in e['replicas']
             for s in [r['data_seed'], r['split_seed'], r['training_seed'],
                       *[t[p] for t in r['terms'] for p in ('initialization', 'shuffle')]]}
    collisions = final.intersection(prior_allocations)
    if collisions:
        raise ValueError(f'Prior/final seed collisions: {sorted(collisions)}')
    return {'prior_values_checked': len(set(prior_allocations)), 'collisions': []}


def materialize(preparation_dir, output_root=DEFAULT_OUTPUT):
    """Create a new preparation bundle only, never a scientific output root."""
    preparation_dir = Path(preparation_dir)
    plan = preview(output_root)
    output = Path(output_root).resolve()
    prep = preparation_dir.resolve()
    if prep == output or output in prep.parents or prep in output.parents:
        raise ValueError('Preparation and scientific output directories must be disjoint.')
    preparation_dir.mkdir(parents=True, exist_ok=False)
    for cell in plan['cells']:
        for kind in ('baseline', 'campaign'):
            with (preparation_dir/f'{cell["campaign_id"]}-{kind}.yaml').open('x', encoding='utf-8', newline='\n') as f:
                yaml.safe_dump(cell[kind], f, sort_keys=True)
    with (preparation_dir/'protocol.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(plan, f, indent=2, allow_nan=False)
        f.write('\n')
    validate_bundle(preparation_dir)
    return plan


def validate_bundle(preparation_dir):
    """Reject changes to the matrix, policy, mapping, hashes or file contents."""
    preparation_dir = Path(preparation_dir)
    saved = json.loads((preparation_dir/'protocol.json').read_text(encoding='utf-8'))
    expected = preview(saved['output_root'])
    if saved != expected:
        raise ValueError('Preparation manifest differs from frozen protocol.')
    for cell in expected['cells']:
        for kind in ('baseline', 'campaign'):
            path = preparation_dir/f'{cell["campaign_id"]}-{kind}.yaml'
            if yaml.safe_load(path.read_text(encoding='utf-8')) != cell[kind]:
                raise ValueError(f'Frozen {kind} configuration differs: {path.name}')
        _, _, replicas = load_campaign_config(preparation_dir/f'{cell["campaign_id"]}-campaign.yaml')
        if replicas != [{k: v for k, v in r.items() if k != 'terms'} for r in cell['replicas']]:
            raise ValueError('Effective replica allocation differs.')
    return expected
