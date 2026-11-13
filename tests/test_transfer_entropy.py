"""TE composition mocks and one minimal singleton CPU integration; no campaigns."""
from dataclasses import FrozenInstanceError, replace
import math
import random

import numpy as np
import pytest
import torch

from src.estimators import transfer_entropy as te
from src.estimators.neural_information import NeuralInformationReport, InformationComponent
from src.estimators.conditional_neural import ConditionalNeuralEntropyReport
from src.utils.neural_config import NeuralConfig
from src.utils.temporal import TemporalEmbeddingSpec


@pytest.fixture(autouse=True)
def preserve_state():
    state=random.getstate(),np.random.get_state(),torch.get_rng_state()
    threads=torch.get_num_threads();torch.set_num_threads(1)
    yield
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
    torch.set_num_threads(threads)


def config(spec=TemporalEmbeddingSpec(),target='Y'):
    return NeuralConfig(42,(.6,.2,.2),202,spec.target_order(target),'declared','train_plugin',
                        (4,),'relu','adam',.01,4,1,1,0.,'cpu','strict',0)


def estimator(**kwargs):
    spec=kwargs.pop('embedding',TemporalEmbeddingSpec())
    target=kwargs.get('target_id','Y')
    return te.NeuralTransferEntropyEstimator(config(spec,target),embedding=spec,
                                            source_alphabet=np.array([0,1]),target_alphabet=np.array([0,1]),**kwargs)


def mock_cmi(monkeypatch,fail=False,values=(.125,.375)):
    calls=[]
    class CMI:
        def __init__(self,config,*,y_order,z_order,alphabets):
            self.config,self.y_order,self.z_order,self.alphabets=config,y_order,z_order,alphabets
            calls.append(self)
        def fit(self,train,validation):
            self.train,self.validation=train,validation
            assert train.role=='train' and validation.role=='validation'
            assert train.dataset_id==validation.dataset_id and train.split_id==validation.split_id
            if fail:raise RuntimeError('component failed')
            return self
        def estimate(self,test):
            assert test.role=='test' and test.dataset_id==self.train.dataset_id and test.split_id==self.train.split_id
            assert not set(test.row_indices)&(set(self.train.row_indices)|set(self.validation.row_indices))
            self.test=test
            components=[]
            for name,context,value in (('h_x_given_z',self.z_order,values[0]),
                                       ('h_x_given_yz',self.y_order+self.z_order,values[1])):
                entropy=ConditionalNeuralEntropyReport(value,self.config.component_order,context,(),
                                                       'strict',test.dataset_id,test.split_id)
                components.append(InformationComponent(name,self.config,context,entropy,0.,0.))
            return NeuralInformationReport(values[0]-values[1],'cmi',self.config.component_order,
                                            self.y_order,self.z_order,test.dataset_id,test.split_id,
                                            self.config,'fixture',{},tuple(components))
    monkeypatch.setattr(te,'NeuralConditionalMutualInformationEstimator',CMI)
    return calls


def test_composition_order_negative_report_and_no_mutation(monkeypatch):
    calls=mock_cmi(monkeypatch)
    spec=TemporalEmbeddingSpec((3,1),(2,1),2)
    source=np.arange(60)%2;target=1-source
    before=source.copy(),target.copy()
    model=estimator(embedding=spec,gap=1)
    assert model.fit(source,target,dataset_id='d') is model
    cmi=calls[0]
    assert cmi.config.component_order==spec.target_order('Y')
    assert cmi.y_order==spec.past_order('X',source=True)
    assert cmi.z_order==spec.past_order('Y',source=False)
    assert not set(cmi.config.component_order)&set(cmi.y_order+cmi.z_order)
    assert not hasattr(cmi,'test')
    report=model.estimate()
    assert report.transfer_entropy_nats==-.25 and report.direction==('X','Y')
    assert report.transfer_entropy_nats==report.cmi_report.components[0].entropy.entropy_nats-report.cmi_report.components[1].entropy.entropy_nats
    assert report.cmi_report.components[1].conditioning_order==cmi.y_order+cmi.z_order
    assert report.partitions[0].target_positions[-1]<report.partitions[1].target_positions[0]
    assert model.estimate().transfer_entropy_nats==report.transfer_entropy_nats
    np.testing.assert_array_equal(source,before[0]);np.testing.assert_array_equal(target,before[1])
    with pytest.raises(FrozenInstanceError):report.direction=('Y','X')
    with pytest.raises(FrozenInstanceError):report.partitions[0].role='test'
    with pytest.raises(RuntimeError):model.fit(source,target,dataset_id='d')


def test_explicit_reversal_changes_target_history_and_seed_namespaces(monkeypatch):
    from src.estimators.neural_information import NeuralConditionalMutualInformationEstimator
    calls=mock_cmi(monkeypatch)
    source=np.arange(30)%2;target=1-source
    f=estimator().fit(source,target,dataset_id='d').estimate()
    r=estimator(source_id='Y',target_id='X').fit(target,source,dataset_id='d').estimate()
    assert f.direction==('X','Y') and r.direction==('Y','X')
    assert calls[0].config.component_order!=calls[1].config.component_order
    def seeds(c):
        return NeuralConditionalMutualInformationEstimator(c.config,y_order=c.y_order,z_order=c.z_order,
                                                          alphabets=c.alphabets)._plans()
    assert seeds(calls[0])!=seeds(calls[1])
    assert seeds(calls[0])==seeds(calls[0])
    assert f.cmi_report.split_id!=r.cmi_report.split_id


def test_exact_delayed_copy_table_has_one_bit_te_with_controlled_entropies(monkeypatch):
    # Periodic exact frequency fixture, not an IID/time-series performance experiment.
    # Every TEST (X_{t-1},Y_{t-1}) state occurs twice and Y_t=X_{t-1}.
    source=np.resize(np.array([0,0,1,1]),41)
    target=np.concatenate(([1],source[:-1]))
    calls=mock_cmi(monkeypatch,values=(math.log(2),0.))
    report=estimator(purge=False).fit(source,target,dataset_id='exact-delayed-copy').estimate()
    table=calls[0].test.values
    np.testing.assert_array_equal(table[:,0],table[:,1])
    assert np.unique(table[:,1:],axis=0,return_counts=True)[1].tolist()==[2,2,2,2]
    assert report.transfer_entropy_nats==math.log(2)


def test_failure_atomicity_retry_and_nonfinite(monkeypatch):
    model=estimator();x=np.arange(30)%2
    with pytest.raises(RuntimeError):model.estimate()
    mock_cmi(monkeypatch,fail=True)
    with pytest.raises(RuntimeError,match='component'):model.fit(x,x,dataset_id='d')
    assert not model._fitted and not hasattr(model,'_model') and not hasattr(model,'_parts')
    mock_cmi(monkeypatch);model.fit(x,x,dataset_id='d')
    assert model.estimate().transfer_entropy_nats==-.25
    mock_cmi(monkeypatch,values=(float('nan'),0.))
    bad=estimator().fit(x,x,dataset_id='d')
    with pytest.raises(RuntimeError,match='Nonfinite'):bad.estimate()


def test_config_identity_and_invalid_alphabet():
    spec=TemporalEmbeddingSpec()
    with pytest.raises(ValueError):
        te.NeuralTransferEntropyEstimator(replace(config(),component_order=('arbitrary',)),
                                         source_alphabet=np.array([0,1]),target_alphabet=np.array([0,1]))
    with pytest.raises(ValueError):estimator(source_id='Y')
    with pytest.raises(ValueError):estimator(gap=True)
    with pytest.raises(ValueError):
        te.NeuralTransferEntropyEstimator(config(),source_alphabet=[0,1],target_alphabet=np.array([0,1]))
    m=te.NeuralTransferEntropyEstimator(config(),source_alphabet=np.array([0,0]),target_alphabet=np.array([0,1]))
    with pytest.raises(ValueError):m.fit(np.zeros(30,dtype=int),np.zeros(30,dtype=int),dataset_id='d')


def test_minimal_singleton_cpu_integration_repeatable_rng_and_models():
    state=random.getstate(),np.random.get_state(),torch.get_rng_state()
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        model=te.NeuralTransferEntropyEstimator(config(),source_alphabet=np.array([0]),target_alphabet=np.array([0]))
        source=target=np.zeros(12,dtype=int)
        model.fit(source,target,dataset_id='singleton')
        snapshots=[(r.model,{k:v.clone() for k,v in r.model.state_dict().items()})
                   for entry in model._model._components for r in entry[4]._results]
        rng=random.getstate(),np.random.get_state(),torch.get_rng_state()
        first=model.estimate();second=model.estimate()
        assert first.transfer_entropy_nats==second.transfer_entropy_nats==0.
        assert len(first.cmi_report.components)==2
        seeds=[]
        for c in first.cmi_report.components:
            seeds.append(c.config.training_seed)
            for term in c.entropy.terms:
                assert term.epochs_trained==0 and term.stop_reason=='singleton_target'
                seeds.extend((term.initialization_seed,term.shuffle_seed))
                assert term.environment['reproducibility_mode']=='strict'
        assert len(set(seeds))==len(seeds)
        assert random.getstate()==rng[0]
        np.testing.assert_array_equal(np.random.get_state()[1],rng[1][1])
        assert torch.equal(torch.get_rng_state(),rng[2])
        for net,before in snapshots:
            assert not net.training
            for k,v in net.state_dict().items():assert torch.equal(v,before[k])
    finally:
        random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
        torch.set_num_threads(threads)


def test_future_categories_not_used_for_encoder_fitting():
    # No optimizer is needed: future target class is declared, train target singleton.
    x=np.zeros(12,dtype=int);y=np.zeros(12,dtype=int);y[-1]=1
    model=te.NeuralTransferEntropyEstimator(config(),source_alphabet=np.array([0]),target_alphabet=np.array([0]))
    model.fit(x,y,dataset_id='d')
    with pytest.raises(ValueError,match='Unknown category'):model.estimate()
    assert model._fitted
