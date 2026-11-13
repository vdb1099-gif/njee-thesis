"""Deterministic composition contracts; exact fixtures make no convergence claim."""
from dataclasses import FrozenInstanceError, replace
import math
import random

import numpy as np
import pytest
import torch

from src.estimators import neural_information as info
from src.estimators.neural import NeuralEntropyReport
from src.estimators.conditional_neural import ConditionalNeuralEntropyReport
from src.utils.categorical import CategoricalDataset
from src.utils.neural_config import NeuralConfig
from src.utils.split import make_iid_split


@pytest.fixture(autouse=True)
def preserve_state():
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
    torch.set_num_threads(threads)


def scenario(x=('x1', 'x0'), singleton=False, n=23):
    ids = ('x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'unused')
    values = np.zeros((n, len(ids)), dtype=int) if singleton else np.array([
        [(i+j) % 2 for j in range(len(ids))] for i in range(n)])
    dataset = CategoricalDataset(values, 'data', ids)
    split = make_iid_split(n, seed=42)
    batches = tuple(dataset.batch(split, role, split_id='split') for role in ('train', 'validation', 'test'))
    config = NeuralConfig(42, (.6,.2,.2), 202, x, 'declared', 'train_plugin',
                          (4,), 'relu', 'adam', .01, 2, 1, 1, 0., 'cpu', 'strict', 0)
    alphabets = {v: np.array([0]) if singleton else np.array([0,1]) for v in ids}
    return batches, config, alphabets


def controlled(monkeypatch, values=(.25,.75), failure=None):
    calls = []
    class Component:
        def __init__(self, config, *, alphabets, conditioning_order=()):
            self.config, self.context = config, conditioning_order
            self.alphabets = alphabets
            self.index = len(calls)
            calls.append(self)
        def fit(self, train, validation):
            self.train, self.validation = train, validation
            assert train.role == 'train' and validation.role == 'validation'
            if failure and self.index == 1:
                raise RuntimeError('second fit failed')
            return self
        def estimate(self, test):
            self.test = test
            assert test.role == 'test'
            args = (values[self.index], self.config.component_order)
            tail = ((), self.config.reproducibility_mode, test.dataset_id, test.split_id)
            if self.context:
                return ConditionalNeuralEntropyReport(*args, self.context, *tail)
            return NeuralEntropyReport(*args, *tail)
    monkeypatch.setattr(info, 'NeuralJointEntropyEstimator', Component)
    # Constructor validation uses the real parent (the composition intentionally reuses it).
    Component._alphabets = None
    original = info.NeuralMutualInformationEstimator._initialize
    def initialize(self, config, y_order, z_order, alphabets):
        # Temporarily use real constructor validation, then use mocks only for fits.
        from src.estimators.neural import NeuralJointEntropyEstimator
        monkeypatch.setattr(info, 'NeuralJointEntropyEstimator', NeuralJointEntropyEstimator)
        try:
            original(self, config, y_order, z_order, alphabets)
        finally:
            monkeypatch.setattr(info, 'NeuralJointEntropyEstimator', Component)
    monkeypatch.setattr(info.NeuralMutualInformationEstimator, '_initialize', initialize)
    monkeypatch.setattr(info, 'ConditionalNeuralJointEntropyEstimator', Component)
    return calls


@pytest.mark.parametrize('cmi', [False, True])
def test_arithmetic_order_shared_partitions_and_immutable_reports(monkeypatch, cmi):
    (train,val,test), config, alphabets = scenario()
    calls = controlled(monkeypatch)
    kwargs = dict(y_order=('y1','y0'), alphabets=alphabets)
    cls = info.NeuralMutualInformationEstimator
    if cmi:
        cls = info.NeuralConditionalMutualInformationEstimator
        kwargs['z_order'] = ('z1','z0')
    model = cls(config, **kwargs)
    before = [b.values.copy() for b in (train,val,test)]
    assert model.fit(train,val) is model
    report = model.estimate(test)
    assert report.information_nats == -.5  # explicitly retained signed result
    assert report.kind == ('cmi' if cmi else 'mi')
    contexts = [('z1','z0'), ('y1','y0','z1','z0')] if cmi else [(), ('y1','y0')]
    assert [c.context for c in calls] == contexts
    for call, context, component in zip(calls, contexts, report.components):
        assert call.config.component_order == ('x1','x0')
        assert call.config == component.config
        for original, projected in zip((train,val,test), (call.train,call.validation,call.test)):
            assert projected.variable_ids == config.component_order + context
            assert projected.dataset_id == original.dataset_id and projected.split_id == original.split_id
            np.testing.assert_array_equal(projected.row_indices, original.row_indices)
            np.testing.assert_array_equal(projected.values, original.values[:, [original.variable_ids.index(v) for v in projected.variable_ids]])
        assert set(call.alphabets) == set(call.train.variable_ids)
    assert calls[0].config.training_seed != calls[1].config.training_seed
    assert report.components[0].entropy.entropy_nats == .25
    assert report.components[1].entropy.entropy_nats == .75
    assert report.base_config == config and report.dataset_id == 'data' and report.split_id == 'split'
    assert report.caller_environment['device'] == 'cpu'
    assert model.estimate(test).information_nats == report.information_nats
    for b,v in zip((train,val,test),before): np.testing.assert_array_equal(b.values,v)
    with pytest.raises(FrozenInstanceError): report.information_nats = 0
    with pytest.raises(FrozenInstanceError): report.components[0].name = 'bad'
    with pytest.raises(TypeError): report.caller_environment['device'] = 'bad'
    with pytest.raises(RuntimeError): model.fit(train,val)


@pytest.mark.parametrize('cmi',[False,True])
@pytest.mark.parametrize('case',['independent','copy','xor'])
def test_exact_probability_identities(monkeypatch,cmi,case):
    # Exact uniform binary probability tables. For XOR, X=Y xor Z:
    # I(X;Y)=0 and I(X;Y|Z)=ln(2). Copy X=Y gives ln(2) in either direction.
    rows = [(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)
            if case == 'independent' or x == (y if case == 'copy' else y^z)]
    p = 1 / len(rows)
    def h(columns):
        counts = {}
        for row in rows:
            key = tuple(row[i] for i in columns)
            counts[key] = counts.get(key,0) + p
        return -math.fsum(q*math.log(q) for q in counts.values())
    a,b = (h((0,2))-h((2,)), h((0,1,2))-h((1,2))) if cmi else (h((0,)),h((0,1))-h((1,)))
    controlled(monkeypatch,(a,b))
    batches,config,alphabets = scenario(x=('x0',))
    cls = info.NeuralConditionalMutualInformationEstimator if cmi else info.NeuralMutualInformationEstimator
    kwargs = {'z_order':('z0',)} if cmi else {}
    report = cls(config,y_order=('y0',),alphabets=alphabets,**kwargs).fit(*batches[:2]).estimate(batches[2])
    expected = math.log(2) if case == 'copy' or (cmi and case == 'xor') else 0
    assert report.information_nats == pytest.approx(expected,abs=1e-14)


@pytest.mark.parametrize('block',[(),[],('y0','y0'),('x0',),(True,),('unknown',)])
def test_invalid_y(block):
    batches,config,alphabets = scenario()
    with pytest.raises(ValueError):
        info.NeuralMutualInformationEstimator(config,y_order=block,alphabets=alphabets).fit(*batches[:2])


@pytest.mark.parametrize('block',[(),[],('z0','z0'),('x0',),('y0',),('unknown',)])
def test_invalid_z(block):
    batches,config,alphabets = scenario()
    with pytest.raises(ValueError):
        info.NeuralConditionalMutualInformationEstimator(config,y_order=('y0',),z_order=block,alphabets=alphabets).fit(*batches[:2])


@pytest.mark.parametrize('phase',['fit','estimate'])
@pytest.mark.parametrize('change',[{'dataset_id':'bad'},{'split_id':'bad'},{'n_samples':24}, {'variable_ids':('y0','x0','z0','x1','y1','z1','unused')}, {'role':'train'}])
def test_identity_rejected_before_component_calls(monkeypatch,phase,change):
    (train,val,test),config,alphabets = scenario()
    calls = controlled(monkeypatch)
    model = info.NeuralMutualInformationEstimator(config,y_order=('y0',),alphabets=alphabets)
    if phase == 'fit':
        with pytest.raises(ValueError): model.fit(train,replace(val,**change))
        assert not calls
    else:
        model.fit(train,val)
        with pytest.raises(ValueError): model.estimate(replace(test,**change))
        assert all(not hasattr(c,'test') for c in calls)


def test_failure_atomicity_and_retry(monkeypatch):
    batches,config,alphabets = scenario()
    calls = controlled(monkeypatch,failure=True)
    model = info.NeuralMutualInformationEstimator(config,y_order=('y0',),alphabets=alphabets)
    with pytest.raises(RuntimeError): model.estimate(batches[2])
    with pytest.raises(TypeError): model.fit(*batches[:2],test=batches[2])
    with pytest.raises(RuntimeError,match='second fit'): model.fit(*batches[:2])
    assert not model._fitted and not hasattr(model,'_components')
    model.fit(*batches[:2])  # later mock indices do not fail; retry publishes only complete state
    assert model._fitted and len(model._components) == 2


@pytest.mark.parametrize('values',[(float('nan'),1.),(1.,float('inf')),(1e308,-1e308)])
def test_nonfinite_components_and_difference(monkeypatch,values):
    batches,config,alphabets = scenario()
    controlled(monkeypatch,values)
    model = info.NeuralMutualInformationEstimator(config,y_order=('y0',),alphabets=alphabets).fit(*batches[:2])
    with pytest.raises(RuntimeError,match='Nonfinite'): model.estimate(batches[2])


def test_stable_stream_separation_and_collision_rejection(monkeypatch):
    batches,config,alphabets = scenario()
    a = info.NeuralConditionalMutualInformationEstimator(config,y_order=('y0',),z_order=('z0',),alphabets=alphabets)
    b = info.NeuralConditionalMutualInformationEstimator(config,y_order=('y0',),z_order=('z0',),alphabets=alphabets)
    assert a._plans() == b._plans()
    assert a._plans() != info.NeuralMutualInformationEstimator(config,y_order=('y0',),alphabets=alphabets)._plans()
    plans = a._plans()
    assert plans[0][2].training_seed == 1398942160  # independently calculated compact-JSON SHA-256 vector
    monkeypatch.setattr(info,'derive_term_seed',lambda *args,**kwargs: 7)
    with pytest.raises(ValueError,match='collision'): a.fit(*batches[:2])
    assert not a._fitted


@pytest.mark.parametrize('cmi',[False,True])
def test_minimal_cpu_integration_and_frozen_state(cmi):
    batches,config,alphabets = scenario(x=('x0',),singleton=True,n=5)
    cls = info.NeuralConditionalMutualInformationEstimator if cmi else info.NeuralMutualInformationEstimator
    kwargs = {'z_order':('z0',)} if cmi else {}
    model = cls(config,y_order=('y0',),alphabets=alphabets,**kwargs).fit(*batches[:2])
    report = model.estimate(batches[2])
    assert report.information_nats == 0.
    assert all(t.epochs_trained == 0 for c in report.components for t in c.entropy.terms if t.method != 'train_plugin')
    snapshots = [(r.model, {k:v.clone() for k,v in r.model.state_dict().items()})
                 for entry in model._components for r in entry[4]._results]
    assert model.estimate(batches[2]).information_nats == report.information_nats
    for net,state in snapshots:
        assert not net.training
        for k,v in net.state_dict().items(): assert torch.equal(v,state[k])


def test_one_epoch_integration_unobserved_classes_and_provenance():
    batches,config,alphabets = scenario(x=('x0',))
    alphabets['x0'] = np.array([0,1,2])
    test = batches[2]; values = test.values.copy(); values[0,0] = 2
    model = info.NeuralMutualInformationEstimator(config,y_order=('y0',),alphabets=alphabets).fit(*batches[:2])
    report = model.estimate(replace(test,values=values))
    assert math.isfinite(report.information_nats)
    assert report.components[0].entropy.terms[0].method == 'train_plugin'
    assert report.components[0].entropy.terms[0].evaluation_role == 'train'
    term = report.components[1].entropy.terms[0]
    assert term.epochs_trained == 1 and term.output_cardinality == 3
    assert term.initialization_seed != term.shuffle_seed
    assert any('absent from training' in w for w in term.warnings)
    assert term.environment['reproducibility_mode'] == 'strict'


def test_malformed_alphabet_overlap_coverage_and_unknown_categories():
    (train,val,test),config,alphabets = scenario(x=('x0',),singleton=True)
    model = info.NeuralMutualInformationEstimator(config,y_order=('y0',),alphabets={**alphabets,'unused':np.array([0,0])})
    with pytest.raises(ValueError): model.fit(train,val)
    model = info.NeuralMutualInformationEstimator(config,y_order=('y0',),alphabets=alphabets)
    with pytest.raises(ValueError): model.fit(train,replace(val,row_indices=train.row_indices[:len(val.values)]))
    model.fit(train,val)
    for bad in (replace(test,row_indices=val.row_indices[:len(test.values)]),
                replace(test,values=test.values[:-1],row_indices=test.row_indices[:-1])):
        with pytest.raises(ValueError): model.estimate(bad)
    v=test.values.copy();v[0,1]=99
    with pytest.raises(ValueError,match='Unknown category'): model.estimate(replace(test,values=v))

def test_component_report_provenance_and_evaluation_failure(monkeypatch):
    batches,config,alphabets = scenario()
    calls = controlled(monkeypatch)
    model = info.NeuralMutualInformationEstimator(config,y_order=('y0',),alphabets=alphabets).fit(*batches[:2])
    original = calls[1].estimate
    calls[1].estimate = lambda test: replace(original(test),split_id='foreign')
    with pytest.raises(RuntimeError,match='provenance'): model.estimate(batches[2])
    calls[1].estimate = original
    assert model.estimate(batches[2]).information_nats == -.5


def test_actual_effective_seed_streams_and_rng_isolation():
    # Singleton targets create real frozen classifiers without any optimization.
    batches,config,alphabets = scenario(singleton=True)
    model = info.NeuralConditionalMutualInformationEstimator(
        config,y_order=('y1','y0'),z_order=('z1','z0'),alphabets=alphabets).fit(*batches[:2])
    state = random.getstate(),np.random.get_state(),torch.get_rng_state()
    report = model.estimate(batches[2])
    seeds = [c.config.training_seed for c in report.components]
    for c in report.components:
        assert len(c.entropy.terms) == 2
        for t in c.entropy.terms:
            assert t.target not in t.conditioning and t.method == 'neural_test_ce'
            seeds.extend((t.initialization_seed,t.shuffle_seed))
    assert len(set(seeds)) == len(seeds)
    assert random.getstate() == state[0]
    np.testing.assert_array_equal(np.random.get_state()[1],state[1][1])
    assert np.random.get_state()[2:] == state[1][2:]
    assert torch.equal(torch.get_rng_state(),state[2])
    assert report.information_nats == 0.


@pytest.mark.parametrize('seed',[0,2**32-1])
def test_seed_boundaries(seed):
    _,config,alphabets = scenario()
    model = info.NeuralMutualInformationEstimator(replace(config,training_seed=seed),y_order=('y0',),alphabets=alphabets)
    assert all(0 <= p[2].training_seed <= 2**32-1 for p in model._plans())
