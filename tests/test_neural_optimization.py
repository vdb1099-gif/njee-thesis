"""Leakage barriers, paired inputs, persistence, budget and frozen decision rule."""
from dataclasses import replace
import json
import math
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from experiments import neural_optimization as study
from src.utils.split import make_iid_split


class GuardedValues:
    """Only approved train/validation indexing is possible; full scans fail."""
    def __init__(self, values, split):
        self.shape = values.shape
        self.values = values
        self.allowed = (split.train, split.validation)
        self.calls = []

    def __getitem__(self, key):
        assert isinstance(key, np.ndarray)
        assert any(np.array_equal(key, a) for a in self.allowed), 'Test/full values accessed'
        self.calls.append(key.copy())
        return self.values[key]

    def __array__(self, *args, **kwargs):
        raise AssertionError('Full dataset coercion forbidden')


def batches():
    split = make_iid_split(30, seed=42)
    values = np.array([[i % 2, i % 2] for i in range(30)])
    values[split.test] = 99  # Unencodable nominal test labels must never be seen.
    guard = GuardedValues(values, split)
    return guard, study.prepare_train_validation(guard, split, dataset_id='data', split_id='split')


def fake_result(config, target, conditioning):
    return SimpleNamespace(model=object(), target=target, conditioning=conditioning,
                           output_cardinality=2, term_id=str(target), initialization_seed=target+10,
                           shuffle_seed=target+20, epochs_trained=1, best_epoch=1,
                           best_validation_ce_nats=.6, final_training_ce_nats=.5,
                           runtime_seconds=.01, stop_reason='early_stopping', warnings=(),
                           epoch_training_ce_nats=(.5,), epoch_validation_ce_nats=(.6,),
                           environment={'device':'cpu','reproducibility_mode':'strict'})


def test_guarded_prepare_and_real_training_never_access_test(tmp_path, monkeypatch):
    guard, (train, validation, encoder) = batches()
    assert len(guard.calls) == 2
    original = study.train_conditional_term
    seen = []
    def checked(train, validation, **kwargs):
        assert train.role == 'train' and validation.role == 'validation'
        assert np.max(train.values) == np.max(validation.values) == 1
        seen.append((train, validation))
        return original(train, validation, **kwargs)
    monkeypatch.setattr(study, 'train_conditional_term', checked)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = replace(study._configuration({'split_seed':42,'training_seed':202}, 2, 'A'), max_epochs=2, patience=1)
        record = {}
        study.evaluate_candidate(train, validation, encoder, config, record=record,
                                 path=tmp_path/'A.json', deadline=time.perf_counter()+30)
    finally:
        torch.set_num_threads(threads)
    assert len(seen) == 1 and len(guard.calls) == 2
    saved = json.loads((tmp_path/'A.json').read_text())
    assert saved['status'] == 'complete'
    assert saved['mean_best_validation_ce_nats'] == saved['terms'][0]['best_validation_ce_nats']
    assert len(saved['terms'][0]['epoch_validation_ce_nats']) == saved['terms'][0]['epochs_trained']
    assert 'model' not in saved['terms'][0]
    assert saved['terms'][0]['environment']['deterministic_algorithms']


def test_fixed_protocol_pairing_and_full_study_without_test(tmp_path, monkeypatch):
    from experiments import neural_campaign, neural_baseline
    from src.estimators import classical, neural
    def forbidden(*args, **kwargs):
        raise AssertionError('Test/entropy/campaign access forbidden')
    monkeypatch.setattr(neural_campaign, 'run_campaign', forbidden)
    monkeypatch.setattr(neural_baseline, 'run_baseline', forbidden)
    monkeypatch.setattr(neural.NeuralJointEntropyEstimator, 'estimate', forbidden)
    monkeypatch.setattr(classical.PlugInEstimator, 'estimate', forbidden)
    monkeypatch.setattr(classical.MillerMadowEstimator, 'estimate', forbidden)
    real_generate = study.generate_dataset
    generated = []
    def generate(config):
        values = real_generate(config)
        cell = next(c for c in study.CELLS if c[1] == config['data']['generator'] and c[4] == config['data']['d'] and c[5] == config['data']['n'])
        seeds = next(study.derive_replica_seeds(cell[-1], r) for r in range(8)
                     if study.derive_replica_seeds(cell[-1], r)['data_seed'] == config['seed'])
        split = make_iid_split(len(values), seed=seeds['split_seed'])
        generated.append(GuardedValues(values, split))
        return generated[-1]
    monkeypatch.setattr(study, 'generate_dataset', generate)
    calls = {}
    def train(train, validation, *, encoder, config, target, conditioning):
        key = train.dataset_id, target
        item = (id(train), id(validation), id(encoder), config.training_seed, train.split_id)
        calls.setdefault(key, []).append(item)
        return fake_result(config, target, conditioning)
    monkeypatch.setattr(study, 'train_conditional_term', train)
    out = study.run_diagnostic(tmp_path/'study')
    assert len(generated) == 32 and all(len(g.calls) == 2 for g in generated)
    assert sum(len(c) for c in calls.values()) == 672
    assert all(len(c) == 3 and c[0] == c[1] == c[2] for c in calls.values())
    records = json.loads((out/'evaluations.json').read_text())
    assert len(records) == 96 and all(r['status'] == 'complete' for r in records)
    assert json.loads((out/'selection.json').read_text())['recommendation'] == 'A'
    for r in records:
        config = r['resolved_configuration']
        assert (config['max_epochs'],config['patience']) == study.CANDIDATES[r['candidate']]
        assert config['hidden_sizes'] == [16] and config['min_delta'] == .0001
    with pytest.raises(FileExistsError):
        study.run_diagnostic(out)


@pytest.mark.parametrize('failure', [RuntimeError('nonfinite'), KeyboardInterrupt()])
def test_partial_term_failure_preserved_and_reraised(tmp_path, monkeypatch, failure):
    _, (train, validation, encoder) = batches()
    def fail(*args, **kwargs): raise failure
    monkeypatch.setattr(study, 'train_conditional_term', fail)
    record = {}
    with pytest.raises(type(failure)):
        study.evaluate_candidate(train, validation, encoder, study._configuration({'split_seed':42,'training_seed':202},2,'A'),
                                 record=record,path=tmp_path/'A.json',deadline=time.perf_counter()+30)
    saved = json.loads((tmp_path/'A.json').read_text())
    assert saved['status'] == ('interrupted' if isinstance(failure, KeyboardInterrupt) else 'failed')
    assert saved['failure']['type'] == type(failure).__name__ and saved['terms'] == []


def test_expired_budget_never_calls_trainer(tmp_path, monkeypatch):
    _, (train, validation, encoder) = batches()
    monkeypatch.setattr(study,'train_conditional_term',lambda *a,**k: pytest.fail('Trainer after deadline'))
    with pytest.raises(TimeoutError):
        study.evaluate_candidate(train,validation,encoder,study._configuration({'split_seed':42,'training_seed':202},2,'A'),
                                 record={},path=tmp_path/'A.json',deadline=0)
    assert json.loads((tmp_path/'A.json').read_text())['status'] == 'interrupted'


def selection_records(b=.0,c=.0):
    records=[]
    for cell,*_ in study.CELLS:
        for replica in range(8):
            for candidate in study.CANDIDATES:
                improvement = 0 if cell=='T1' else {'A':0,'B':b,'C':c}[candidate]
                records.append({'cell':cell,'replica':replica,'candidate':candidate,'status':'complete',
                                'mean_best_validation_ce_nats':.7-improvement,'dataset_id':f'{cell}-{replica}',
                                'split_id':'split','seeds':{'training_seed':replica},'categorical_mappings':{'0':{0:0,1:1}},
                                'terms':[{'initialization_seed':replica,'shuffle_seed':replica+10}]})
    return records


@pytest.mark.parametrize('b,c,expected',[(0,0,'A'),(.01,0,'B'),(0,.01,'C'),(.01,.012,'B'),(.01,.02,'C')])
def test_frozen_selection_branches(b,c,expected):
    result=study.select_policy(selection_records(b,c))
    assert result['recommendation']==expected
    assert result['comparisons']['C_versus_B']['cells']['T2']['n_strictly_positive']==(8 if c>b else 0)


def test_consistency_and_independent_guard():
    records=selection_records(.01,0)
    for r in records:
        if r['candidate']=='B' and r['replica']>=5: r['mean_best_validation_ce_nats']=.7
    assert study.select_policy(records)['recommendation']=='A' # only five positive replicas
    records=selection_records(.01,0)
    for r in records:
        if r['candidate']=='B' and r['cell']=='T1':r['mean_best_validation_ce_nats']=.706
    assert study.select_policy(records)['recommendation']=='A'


def test_incomplete_duplicate_and_pairing_mismatch():
    records=selection_records()
    assert study.select_policy(records[:-1])['recommendation'] is None
    records[0]['status']='failed'
    assert study.select_policy(records)['recommendation'] is None
    with pytest.raises(ValueError,match='Duplicate'):study.select_policy(records+[records[0]])
    records=selection_records();records[1]['split_id']='wrong'
    with pytest.raises(ValueError,match='pairing'):study.select_policy(records)


def test_nonfinite_trainer_record_is_failure(tmp_path,monkeypatch):
    _,(train,validation,encoder)=batches()
    def bad(*args,**kwargs):
        result=fake_result(kwargs['config'],kwargs['target'],kwargs['conditioning'])
        result.best_validation_ce_nats=math.nan
        return result
    monkeypatch.setattr(study,'train_conditional_term',bad)
    with pytest.raises(RuntimeError,match='Nonfinite'):
        study.evaluate_candidate(train,validation,encoder,study._configuration({'split_seed':42,'training_seed':202},2,'A'),
                                 record={},path=tmp_path/'A.json',deadline=time.perf_counter()+30)
    assert json.loads((tmp_path/'A.json').read_text())['status']=='failed'
