from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from experiments.final_protocol import digest, relative_path
from experiments.transfer_protocol import preview as transfer_preview, prior_seeds
from src.estimators.ablation import stream_ids
from src.utils.neural_config import validate_neural_config
from src.utils.reproducibility import derive_term_seed


SCHEMA = 'njee-ablation-v1'
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'ablation_validation.yaml'


def definition():
    return {'schema': SCHEMA, 'replicas': 30, 'n': 2000, 'd': 6,
            'epsilon': .15, 'training_fractions': [.25, .5, 1.],
            'architectures': [[8], [16], [32], [16, 16], [32, 32]],
            'permutation_seed': 190019, 'random_orders': 5, 'discretizations': [2, 3, 5],
            'temporal': {'coupling': .3, 'delay': 2, 'noise_x': .1, 'noise_y': .1,
                         'burn_in': 1000, 'initial': 'random', 'common_span': 3, 'gap': 3, 'purge': True,
                         'candidates': [2], 'horizon': 1},
            'master_seed': 190000, 'bootstrap_seed': 271819, 'bootstrap_draws': 10000,
            'output_root': 'results/ablation_final',
            'neural': {'split_seed': 0, 'split_fractions': [.6, .2, .2], 'training_seed': 0,
                       'component_order': list(range(6)), 'alphabet_policy': 'declared',
                       'marginal_method': 'train_plugin', 'hidden_sizes': [16], 'activation': 'relu',
                       'optimizer': 'adam', 'learning_rate': .01, 'batch_size': 64,
                       'max_epochs': 20, 'patience': 5, 'min_delta': .0001, 'device': 'cpu',
                       'reproducibility_mode': 'strict', 'num_workers': 0},
            'policies': {'sharing': 'fixed physical slots + term one-hot; shared trunk / distinct heads; joint mean CE checkpoint',
                         'sharing_control': 'same representation/pooled objective/checkpoint; separate trunks and heads',
                         'early_stopping_off': '20 epochs; last checkpoint; validation diagnostic only',
                         'subsets': 'nested prefixes of one TRAIN-only permutation; floor sizes; unchanged val/test',
                         'discretization': 'same independent Uniform[0,1) matrix; analytic equal-width bins; H=d ln K',
                         'inference': 'exploratory pointwise replica bootstrap; no confirmatory significance claims',
                         'failure': 'no retry/replacement; incomplete pairs excluded with counts',
                         'reference': 'finite-history stationary TE; truncation bound excludes roundoff'}}


def load_definition(path=CONFIG_PATH):
    if yaml.safe_load(Path(path).read_text(encoding='utf-8')) != definition():
        raise ValueError('Ablation configuration differs from frozen policy.')
    return definition()


def orders():
    d = definition()
    rng = np.random.default_rng(d['permutation_seed'])
    result = [tuple(range(d['d']))]
    while len(result) <= d['random_orders']:
        candidate = tuple(map(int, rng.permutation(d['d'])))
        if candidate not in result:
            result.append(candidate)
    return result


def cells():
    d = definition()
    result = []
    def add(group, name, factor, **overrides):
        cell = {'cell_id': f'A{len(result)+1:03d}-{group}-{name}', 'group': group,
                'name': name, 'factor': factor, 'kind': 'entropy', 'order': list(orders()[0]),
                'hidden_sizes': [16], 'variant': 'native', 'train_fraction': 1.,
                'early_stopping': True, 'states': 2, **overrides}
        cell['configuration_hash'] = digest(cell)
        result.append(cell)
    for group in ('independent', 'binary_chain'):
        add(group, 'reference', 'reference')
        for i, order in enumerate(orders()[1:], 1):
            add(group, f'order-{i}', 'ordering', order=list(order))
        for sizes in d['architectures']:
            if sizes != [16]:
                add(group, 'hidden-' + '-'.join(map(str, sizes)), 'architecture', hidden_sizes=sizes)
        add(group, 'sharing-control', 'sharing', variant='sharing_control')
        add(group, 'shared', 'sharing', variant='shared')
        for fraction in d['training_fractions']:
            if fraction != 1.:
                add(group, f'train-{int(100*fraction)}', 'training_size', train_fraction=fraction)
        add(group, 'no-early-stopping', 'early_stopping', early_stopping=False)
    for states in d['discretizations']:
        add('discretization', f'K{states}', 'discretization', states=states)
    for direction in ('X_to_Y', 'Y_to_X'):
        for name, source, target in (('correct', [2], [1]), ('insufficient', [2], [2]),
                                     ('excessive', [2, 3], [1, 2])):
            add('temporal', f'{direction}-{name}', 'embedding', kind='te', direction=direction,
                source_lags=source, target_lags=target, horizon=1)
    return result


def config_for(cell, seed, split_seed=0):
    data = deepcopy(definition()['neural'])
    data.update(training_seed=seed, split_seed=split_seed, component_order=cell['order'],
                hidden_sizes=cell['hidden_sizes'])
    return validate_neural_config(data)


def comparison_pairs(matrix):
    pairs = []
    for group in ('independent', 'binary_chain', 'discretization', 'temporal'):
        own = [c for c in matrix if c['group'] == group]
        for cell in own:
            if group in ('independent', 'binary_chain'):
                if cell['name'] in ('reference', 'sharing-control'):
                    continue
                name = 'sharing-control' if cell['name'] == 'shared' else 'reference'
                ref = next(c for c in own if c['name'] == name)
            elif group == 'discretization':
                if cell['states'] == 2:
                    continue
                ref = own[0]
            else:
                if cell['name'].endswith('-correct'):
                    continue
                ref = next(c for c in own if c['direction'] == cell['direction'] and c['name'].endswith('-correct'))
            pairs.append({'left': cell['cell_id'], 'right': ref['cell_id'],
                          'same_estimand': group not in ('discretization', 'temporal'),
                          'interpretation': 'paired errors relative to each cell-specific truth; not a common-quantity contrast'
                          if group in ('discretization', 'temporal') else 'same-quantity paired error contrast'})
    return pairs


@lru_cache(maxsize=1)
def reserved_seeds():
    reserved = set(prior_seeds())
    for cell in transfer_preview()['cells']:
        reserved.update((cell['master_seed'], cell['bootstrap']['seed']))
        for r in cell['allocation']:
            reserved.add(r['data']['seed'])
            reserved.update(x['seed'] for x in r['surrogates'].values())
            for job in r['jobs'].values():
                reserved.add(job['seed'])
                for term in job['components']:
                    reserved.update(term[p] for p in ('training_seed', 'initialization', 'shuffle'))
    return frozenset(reserved)


def preview(output_root=None):
    d = definition()
    matrix = cells()
    reserved = set(reserved_seeds()) | {d['master_seed'], d['bootstrap_seed'], d['permutation_seed']}
    def allocate(identity, cell=None):
        for nonce in range(10000):
            seed = int.from_bytes(hashlib.sha256(json.dumps([SCHEMA, d['master_seed'], identity, nonce],
                                   separators=(',', ':')).encode()).digest()[:4], 'big')
            effective = []
            if cell and cell['kind'] == 'entropy':
                for tid in stream_ids(tuple(cell['order']), cell['variant']):
                    effective.extend(derive_term_seed(seed, tid, purpose=p) for p in ('initialization', 'shuffle'))
            elif cell:
                from experiments.transfer_protocol import effective_streams
                effective, _ = effective_streams(cell, seed)
            group = [seed, *effective]
            if len(set(group)) != len(group) or reserved.intersection(group):
                continue
            reserved.update(group)
            return {'seed': seed, 'nonce': nonce, 'effective_seeds': effective}
        raise ValueError('Seed allocation exhausted.')
    groups = []
    for name in ('independent', 'binary_chain', 'discretization', 'temporal'):
        allocation = []
        for i in range(d['replicas']):
            rid = f'replica-{i:06d}'
            identity = [name, rid]
            allocation.append({'replica_id': rid, **{role: allocate([*identity, role])
                               for role in ('data', 'split', 'subset')},
                               'fits': {c['cell_id']: allocate([*identity, c['cell_id']], c)
                                        for c in matrix if c['group'] == name}})
        groups.append({'group_id': name, 'allocation': allocation})
    scientific = {'schema': SCHEMA, 'definition': {k: v for k, v in d.items() if k != 'output_root'},
                  'cells': matrix, 'groups': groups, 'comparisons': comparison_pairs(matrix),
                  'execution_order': [g['group_id'] for g in groups],
                  'seed_policy': 'sha256-uint32-global-nonce-reservation-v1'}
    entropy = sum(c['kind'] == 'entropy' for c in matrix) * d['replicas']
    te = sum(c['kind'] == 'te' for c in matrix) * d['replicas']
    return {**scientific, 'protocol_hash': digest(scientific),
            'output_root': relative_path(d['output_root'] if output_root is None else output_root),
            'expected': {'cells': len(matrix), 'replicas_per_cell': d['replicas'],
                         'independent_datasets_or_processes': len(groups) * d['replicas'],
                         'cell_replica_records': len(matrix) * d['replicas'],
                         'entropy_fits': entropy, 'te_fits': te,
                         'conditional_heads': entropy * (d['d'] - 1) + 2 * te,
                         'shared_trunk_fits': 2 * d['replicas'], 'surrogate_statistics': 0}}


def materialize(directory, *, output_root=None):
    directory = Path(directory)
    plan = preview(output_root)
    root, prep = Path(plan['output_root']).resolve(), directory.resolve()
    if root == prep or root in prep.parents or prep in root.parents:
        raise ValueError('Preparation and output must be disjoint.')
    directory.mkdir(parents=True, exist_ok=False)
    for cell in plan['cells']:
        with (directory / (cell['cell_id'] + '.yaml')).open('x', encoding='utf-8') as f:
            yaml.safe_dump(cell, f, sort_keys=True)
    with (directory / 'protocol.json').open('x', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, allow_nan=False)
    return plan


def validate_bundle(directory):
    directory = Path(directory)
    saved = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    expected = preview(saved['output_root'])
    if saved != expected:
        raise ValueError('Ablation protocol/hash mismatch.')
    names = {'protocol.json', *[c['cell_id'] + '.yaml' for c in expected['cells']]}
    if {p.name for p in directory.iterdir()} != names:
        raise ValueError('Unexpected/missing bundle files.')
    for cell in expected['cells']:
        if yaml.safe_load((directory / (cell['cell_id'] + '.yaml')).read_text(encoding='utf-8')) != cell:
            raise ValueError('Cell configuration changed.')
    return expected
