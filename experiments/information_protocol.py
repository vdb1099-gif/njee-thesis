"""Frozen, outcome-independent Experiment 2 configuration and seed allocation."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from experiments.final_protocol import digest, relative_path, preview as entropy_preview
from src.estimators.neural_information import (
    NeuralMutualInformationEstimator, NeuralConditionalMutualInformationEstimator,
)
from src.simulation.information import InformationModel
from src.utils.neural_config import NeuralConfig


SCHEMA = 'njee-experiment2-v1'
CONFIG_PATH = Path(__file__).resolve().parents[1]/'configs'/'information_validation.yaml'


def definition():
    """Fixed before execution, chosen from binary-channel theory, not pilot outcomes."""
    return {'schema': SCHEMA, 'sample_sizes': [250, 500, 1000, 5000, 10000],
            'replicas': 50, 'mi_epsilon': [0., .05, .15, .30, .50],
            'chain_epsilon_x': [.15, .50], 'chain_epsilon_y': [0., .05, .15, .30, .50],
            'fork_epsilon_x': [.15], 'fork_epsilon_y': [.15, .50],
            'master_seed_start': 160000, 'bootstrap_seed': 271816, 'bootstrap_draws': 10000,
            'permutation_draws': 999, 'test_alpha': .05,
            'output_root': 'results/experiment2_final',
            'analysis': {'sampling_unit':'independent_replica', 'inference':'pointwise_exploratory',
                         'neural_direction':'H(X)-H(X|Y); H(X|Z)-H(X|Y,Z)',
                         'classical_variants':['plug_in_full','miller_madow_full','plug_in_train','miller_madow_train'],
                         'bootstrap':'95-percentile-paired-replica-10000-linear-quantiles',
                         'correlation':'Pearson-component-errors-joint-replica-bootstrap',
                         'frequency_interval':'95-Clopper-Pearson', 'secondary_clipping':True,
                         'null_test':'frozen-model-heldout-exact-strata-permutation-v1',
                         'p_value':'(1+count(T_surrogate>=T_observed))/(B+1)',
                         'classical_significance_tests':False},
            'neural': {'split_seed': 0, 'split_fractions': [.6, .2, .2], 'training_seed': 0,
                       'component_order': ['X'], 'alphabet_policy': 'declared',
                       'marginal_method': 'train_plugin', 'hidden_sizes': [16], 'activation': 'relu',
                       'optimizer': 'adam', 'learning_rate': .01, 'batch_size': 64,
                       'max_epochs': 20, 'patience': 5, 'min_delta': .0001, 'device': 'cpu',
                       'reproducibility_mode': 'strict', 'num_workers': 0}}


def load_definition(path=CONFIG_PATH):
    saved = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if saved != definition():
        raise ValueError('Configuration differs from frozen Experiment 2 definition.')
    return saved


def neural_config(split_seed, training_seed):
    from src.utils.neural_config import validate_neural_config
    config = definition()['neural']
    config.update(split_seed=split_seed, training_seed=training_seed)
    return validate_neural_config(config)


def make_estimator(model, config):
    alphabets = {v: np.array([0, 1], dtype=np.int64) for v in model.variable_ids}
    if model.family == 'bsc':
        return NeuralMutualInformationEstimator(config, y_order=('Y',), alphabets=alphabets)
    return NeuralConditionalMutualInformationEstimator(config, y_order=('Y',), z_order=('Z',),
                                                      alphabets=alphabets)


def cells():
    d = definition()
    settings = [('bsc', 0., ey) for ey in d['mi_epsilon']]
    settings += [('chain', ex, ey) for ex in d['chain_epsilon_x'] for ey in d['chain_epsilon_y']]
    settings += [('fork', ex, ey) for ex in d['fork_epsilon_x'] for ey in d['fork_epsilon_y']]
    result = []
    for family, ex, ey in settings:
        model = InformationModel(family, ex, ey)
        for n in d['sample_sizes']:
            index = len(result)
            result.append({'cell_id': f'I{index+1:03d}-{family}-ex{round(ex*100):03d}-ey{round(ey*100):03d}-n{n:05d}',
                           'model': asdict(model), 'n': n, 'replicas': d['replicas'],
                           'master_seed': d['master_seed_start']+index, 'truth': model.truths()})
    return result


@lru_cache(maxsize=1)
def _prior_seeds():
    """Read allocation metadata only, not scientific observations or result files."""
    prior = entropy_preview()
    seeds = {271812, 271813, 271816, 120010, 120011, 120012, 120013, 120014,
             130010, 130011, 130012, 130013}
    from experiments.neural_campaign import derive_replica_seeds
    from src.utils.reproducibility import derive_term_seed
    for master, dim, count in [(120010,4,3),(120011,4,3),(120012,4,3),(120013,4,3),(120014,12,3),
                               (130010,4,8),(130011,4,8),(130012,12,8),(130013,12,8)]:
        for i in range(count):
            r = derive_replica_seeds(master, i)
            seeds.update(r.values())
            for target in range(1, dim):
                tid = json.dumps(['njee-conditional-term-v1', target, list(range(target))], separators=(',', ':'))
                seeds.update(derive_term_seed(r['training_seed'], tid, purpose=p)
                             for p in ('initialization', 'shuffle'))
    for cell in prior['cells']:
        seeds.add(cell['master_seed'])
        for r in cell['replicas']:
            seeds.update(r[k] for k in ('data_seed', 'split_seed', 'training_seed'))
            for t in r['terms']:
                seeds.update(t[p] for p in ('initialization', 'shuffle'))
    return frozenset(seeds)


def effective_streams(model, training_seed):
    """Existing composition/trainer derivations, without constructing or fitting an MLP."""
    estimator = make_estimator(model, neural_config(0, training_seed))
    from src.utils.reproducibility import derive_term_seed
    streams, components = [], []
    for name, context, config in estimator._plans():
        streams.append(config.training_seed)
        terms = []
        for i, target in enumerate(config.component_order):
            if not context and not i:
                continue
            condition = context + config.component_order[:i]
            tid = json.dumps(['njee-conditional-term-v1', target, list(condition)], separators=(',', ':'))
            seeds = {p: derive_term_seed(config.training_seed, tid, purpose=p)
                     for p in ('initialization', 'shuffle')}
            streams.extend(seeds.values())
            terms.append({'target': target, 'conditioning': list(condition), **seeds})
        components.append({'name': name, 'training_seed': config.training_seed, 'terms': terms})
    return streams, components


def preview(output_root=None):
    """Pure plan generation; globally disjoint effective seed allocations.

    SHA256 compact JSON [schema, master, cell, replica, role, nonce], first four
    bytes big-endian. Deterministic nonce probing rejects any collision against
    earlier allocations and reserved Experiment 1/pilot/development streams.
    Training candidates reserve all downstream component/term seeds together.
    No collision is silently accepted. Nonces and resolved seeds are persisted.
    """
    d = definition()
    root = relative_path(d['output_root'] if output_root is None else output_root)
    reserved = set(_prior_seeds())
    prior_count = len(reserved)
    matrix = cells()
    reserved.update(c['master_seed'] for c in matrix)
    for cell in matrix:
        model = InformationModel(**cell['model'])
        allocations = []
        for i in range(d['replicas']):
            r = {'replica_id': f'replica-{i:06d}', 'nonces': {}}
            for role in ('data_seed', 'split_seed', 'training_seed', 'permutation_seed', 'bootstrap_seed'):
                for nonce in range(10000):
                    domain = d['bootstrap_seed'] if role == 'bootstrap_seed' else cell['master_seed']
                    payload = json.dumps([SCHEMA, domain, cell['cell_id'], r['replica_id'], role, nonce],
                                         separators=(',', ':')).encode()
                    seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')
                    streams, components = [], []
                    try:
                        if role == 'training_seed':
                            streams, components = effective_streams(model, seed)
                    except ValueError:
                        continue
                    candidates = [seed, *streams]
                    if len(set(candidates)) != len(candidates) or reserved.intersection(candidates):
                        continue
                    reserved.update(candidates)
                    r[role], r['nonces'][role] = seed, nonce
                    if role == 'training_seed':
                        r['components'] = components
                    break
                else:
                    raise ValueError('Seed allocation exhausted; protocol review required.')
            allocations.append(r)
        cell['allocation'] = allocations
        cell['configuration_hash'] = digest({k: v for k, v in cell.items() if k != 'allocation'})
    # Null cells first so failure/FPR evidence is preserved under interruption.
    order = [c['cell_id'] for c in sorted(matrix, key=lambda c: (not c['truth']['is_null'], c['n'], c['cell_id']))]
    scientific = {'definition': {k:v for k,v in d.items() if k != 'output_root'}, 'cells': matrix,
                  'execution_order': order, 'seed_policy': 'sha256-uint32-global-collision-rejection-v1'}
    null_replicas = sum(c['replicas'] for c in matrix if c['truth']['is_null'])
    plan = {'schema': SCHEMA, 'output_root': root, **scientific,
            'expected': {'cells': len(matrix), 'replicas': sum(c['replicas'] for c in matrix),
                         'null_replicas': null_replicas,
                         'conditional_fits': sum(c['replicas']*(1 if c['model']['family']=='bsc' else 2) for c in matrix),
                         'estimator_records': sum(c['replicas']*5 for c in matrix),
                         'null_surrogate_evaluations': null_replicas*d['permutation_draws']},
            'seed_audit': {'historical_values_reserved': prior_count, 'collisions_accepted': 0},
            'protocol_hash': digest(scientific)}
    return plan


def materialize(directory, *, output_root=None):
    directory = Path(directory)
    plan = preview(output_root)
    root, prep = Path(plan['output_root']).resolve(), directory.resolve()
    if root == prep or root in prep.parents or prep in root.parents:
        raise ValueError('Preparation and execution directories must be disjoint.')
    directory.mkdir(parents=True, exist_ok=False)
    for cell in plan['cells']:
        with (directory/f"{cell['cell_id']}.yaml").open('x', encoding='utf-8') as stream:
            yaml.safe_dump(cell, stream, sort_keys=True)
    with (directory/'protocol.json').open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
    return plan


def validate_bundle(directory):
    directory = Path(directory)
    saved = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    expected = preview(saved['output_root'])
    if saved != expected:
        raise ValueError('Experiment 2 manifest/hash differs from frozen protocol.')
    names = {'protocol.json'} | {f"{c['cell_id']}.yaml" for c in expected['cells']}
    if {p.name for p in directory.iterdir()} != names:
        raise ValueError('Unexpected/missing preparation files.')
    for cell in expected['cells']:
        if yaml.safe_load((directory/f"{cell['cell_id']}.yaml").read_text(encoding='utf-8')) != cell:
            raise ValueError('Cell configuration mismatch.')
    return expected
