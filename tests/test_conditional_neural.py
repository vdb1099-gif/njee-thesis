"""Exact conditional CE fixtures, identity/leakage checks and tiny CPU integration."""
from dataclasses import FrozenInstanceError, replace
import math
import random

import numpy as np
import pytest
import torch

from src.estimators import conditional_neural as conditional
from src.estimators import neural_training as training
from src.estimators.classical import PlugInEstimator
from src.estimators.neural_training import TermTrainingResult
from src.utils.categorical import CategoricalBatch, CategoricalDataset, CategoricalEncoder
from src.utils.neural_config import NeuralConfig
from src.utils.reproducibility import derive_term_seed
from src.utils.split import make_iid_split


@pytest.fixture(autouse=True)
def preserve_state():
    state=random.getstate(),np.random.get_state(),torch.get_rng_state()
    threads=torch.get_num_threads();torch.set_num_threads(1)
    yield
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
    torch.set_num_threads(threads)


def scenario(x=('x0','x1'),y=('y0','y1'),singleton=False):
    ids=x+y+('unused',)
    values=np.array([[(i+j)%2 for j in range(len(ids))] for i in range(23)])
    if singleton:values[:,:len(x)]=0
    ds=CategoricalDataset(values,'data',ids);split=make_iid_split(23,seed=42)
    batches=tuple(ds.batch(split,role,split_id='split') for role in ('train','validation','test'))
    config=NeuralConfig(42,(.6,.2,.2),202,x,'declared','train_plugin',(4,),'relu','adam',.01,2,1,1,0.,'cpu','strict',0)
    alphabets={v:np.array([0]) if singleton and v in x else np.array([0,1]) for v in ids}
    return batches,config,alphabets


class KnownLogits(torch.nn.Module):
    def __init__(self,cardinality,kind='uniform'):
        super().__init__();self.kind=kind
        self.register_buffer('logits',torch.zeros(cardinality,dtype=torch.float64))
    def forward(self,features):
        if self.kind=='uniform':return self.logits.expand(len(features),-1)
        # First binary Y block: p(X=1|Y=0)=.25, p(X=1|Y=1)=.75.
        p=torch.where(features[:,0].bool(),.25,.75).double()
        if self.kind=='copy':
            return torch.stack((features[:,0],features[:,1]),dim=1).double()*1000
        return torch.stack((torch.log1p(-p),torch.log(p)),dim=1)


def controlled(calls,kind='uniform'):
    def fit(train,validation,*,encoder,config,target,conditioning,context):
        assert train.role=='train' and validation.role=='validation'
        assert conditioning==context+config.component_order[:config.component_order.index(target)]
        assert target not in conditioning
        model=KnownLogits(encoder.cardinalities[target],kind).eval().requires_grad_(False)
        calls.append((target,conditioning,context,encoder,model))
        return TermTrainingResult(model,target,conditioning,encoder.cardinalities[target],str(target),
                                  101+len(calls),201+len(calls),2,1,.61,.6,.01,'early_stopping',
                                  ('fixture',),(.7,.6),(.61,.62),{'device':'cpu','reproducibility_mode':'strict'})
    return fit


def test_all_terms_learned_permutation_feature_order_and_diagnostics(monkeypatch):
    (train,val,test),config,alphabets=scenario()
    config=replace(config,component_order=('x1','x0'));calls=[]
    monkeypatch.setattr(conditional,'train_conditional_term',controlled(calls))
    monkeypatch.setattr(PlugInEstimator,'estimate',lambda *a,**k:pytest.fail('No training marginal in C-NJEE'))
    model=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y1','y0'),alphabets=alphabets)
    assert model.fit(train,val) is model
    assert [(t,c,y) for t,c,y,_,_ in calls]==[('x1',('y1','y0'),('y1','y0')),('x0',('y1','y0','x1'),('y1','y0'))]
    assert calls[0][3] is calls[1][3] and calls[0][4] is not calls[1][4]
    encoder=calls[0][3]
    features=encoder.conditioning_features(test,('y1','y0','x1'),target='x0')
    expected=np.concatenate([np.eye(2,dtype=np.float32)[test.values[:,test.variable_ids.index(v)]] for v in ('y1','y0','x1')],axis=1)
    np.testing.assert_array_equal(features,expected)
    report=model.estimate(test)
    assert report.entropy_nats==pytest.approx(2*math.log(2),abs=1e-12)
    assert report.entropy_nats==math.fsum(t.contribution_nats for t in report.terms)
    assert report.component_order==('x1','x0') and report.conditioning_order==('y1','y0')
    assert [t.position for t in report.terms]==[1,2]
    assert all(t.method=='neural_test_ce' and t.evaluation_role=='test' for t in report.terms)
    first=report.terms[0]
    assert first.n_evaluation_observations==len(test.values) and first.best_validation_ce_nats==.61
    assert first.epochs_trained==2 and first.best_epoch==1 and first.final_training_ce_nats==.6
    assert first.epoch_training_ce_nats==(.7,.6) and first.epoch_validation_ce_nats==(.61,.62)
    assert first.initialization_seed==102 and first.shuffle_seed==202 and first.warnings==('fixture',)
    with pytest.raises(FrozenInstanceError):report.entropy_nats=0
    with pytest.raises(FrozenInstanceError):first.contribution_nats=0
    with pytest.raises(TypeError):first.environment['device']='cuda'


def test_dx1_known_conditional_distribution(monkeypatch):
    (train,val,test),config,alphabets=scenario(x=('x',),y=('y',));calls=[]
    monkeypatch.setattr(conditional,'train_conditional_term',controlled(calls,'binary'))
    model=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y',),alphabets=alphabets).fit(train,val)
    report=model.estimate(test)
    x=test.values[:,0];y=test.values[:,1];p=np.where(y==0,.25,.75)
    reference=float(np.mean(-np.log(np.where(x==1,p,1-p))))
    assert report.entropy_nats==pytest.approx(reference,abs=1e-12)
    assert len(calls)==len(report.terms)==1 and report.terms[0].conditioning==('y',)


def test_exact_known_conditional_entropy_from_balanced_frequency_table(monkeypatch):
    # For each Y, exactly 3/4 copy Y and 1/4 differ. q equals this conditional PMF.
    table=np.array([[0,0]]*3+[[1,0]]+[[0,1]]+[[1,1]]*3)
    train=CategoricalBatch(np.tile(table,(3,1)),'exact',('x','y'),np.arange(24),'train','split',40)
    val=CategoricalBatch(table,'exact',('x','y'),np.arange(24,32),'validation','split',40)
    test=CategoricalBatch(table,'exact',('x','y'),np.arange(32,40),'test','split',40)
    _,config,_=scenario(x=('x',),y=('y',));alphabets={v:np.array([0,1]) for v in ('x','y')}
    monkeypatch.setattr(conditional,'train_conditional_term',controlled([],'binary'))
    report=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y',),alphabets=alphabets).fit(train,val).estimate(test)
    truth=-.25*math.log(.25)-.75*math.log(.75)
    assert report.entropy_nats==pytest.approx(truth,abs=1e-12)


def test_constant_conditioning_and_declared_unobserved_context(monkeypatch):
    (train,val,test),config,alphabets=scenario(x=('x',),y=('y',))
    def constant(batch):
        values=batch.values.copy();values[:,1]=0;return replace(batch,values=values)
    train,val,test=map(constant,(train,val,test));alphabets['y']=np.array([0])
    calls=[];monkeypatch.setattr(conditional,'train_conditional_term',controlled(calls))
    model=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y',),alphabets=alphabets).fit(train,val)
    assert model.estimate(test).entropy_nats==pytest.approx(math.log(2),abs=1e-12)
    assert model._encoder.conditioning_features(test,('y',),target='x').shape==(len(test.values),1)
    alphabets['y']=np.array([0,1,2]);values=test.values.copy();values[:,1]=2
    model=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y',),alphabets=alphabets).fit(train,val)
    assert model.estimate(replace(test,values=values)).entropy_nats==pytest.approx(math.log(2),abs=1e-12)


def test_exact_accumulation_controlled_ce(monkeypatch):
    (train,val,test),config,alphabets=scenario();calls=[]
    monkeypatch.setattr(conditional,'train_conditional_term',controlled(calls))
    values=iter([.125,.375]);monkeypatch.setattr(conditional,'_test_ce',lambda *a,**k:next(values))
    report=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y0',),alphabets=alphabets).fit(train,val).estimate(test)
    assert report.entropy_nats==.5 and [t.contribution_nats for t in report.terms]==[.125,.375]


def test_deterministic_copy_has_zero_conditional_ce(monkeypatch):
    (train,val,test),config,alphabets=scenario(x=('x',),y=('y',))
    def copy(batch):
        v=batch.values.copy();v[:,0]=v[:,1];return replace(batch,values=v)
    calls=[];monkeypatch.setattr(conditional,'train_conditional_term',controlled(calls,'copy'))
    report=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y',),alphabets=alphabets).fit(copy(train),copy(val)).estimate(copy(test))
    assert report.entropy_nats==0.


def test_singleton_actual_trainer_skips_optimizer(monkeypatch):
    (train,val,test),config,alphabets=scenario(x=('x',),y=('y',),singleton=True)
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('Singleton does not optimize'))
    report=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y',),alphabets=alphabets).fit(train,val).estimate(test)
    assert report.entropy_nats==0. and report.terms[0].epochs_trained==0
    assert report.terms[0].stop_reason=='singleton_target'


def test_minimal_cpu_integration_declared_unobserved_target_and_context_seeds():
    (train,val,test),config,alphabets=scenario(x=('x',),y=('y',))
    alphabets['x']=np.array([0,1,2]) # third declared target class absent in training
    v=test.values.copy();v[0,0]=2;test=replace(test,values=v)
    model=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y',),alphabets=alphabets).fit(train,val)
    report=model.estimate(test);term=report.terms[0]
    assert math.isfinite(report.entropy_nats) and term.epochs_trained==1 and term.output_cardinality==3
    assert any('absent from training' in s for s in term.warnings)
    tid='["njee-conditional-term-v1","x",["y"]]'
    assert term.initialization_seed==derive_term_seed(202,tid,purpose='initialization')
    assert term.shuffle_seed==derive_term_seed(202,tid,purpose='shuffle')


def test_repeatable_estimate_does_not_mutate_models_mappings_rng_or_backend(monkeypatch):
    (train,val,test),config,alphabets=scenario();calls=[]
    monkeypatch.setattr(conditional,'train_conditional_term',controlled(calls))
    model=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y0','y1'),alphabets=alphabets).fit(train,val)
    snapshots=[{k:v.clone() for k,v in call[4].state_dict().items()} for call in calls]
    mappings={k:dict(v) for k,v in model._encoder.mappings.items()}
    def forbidden(*a,**k):pytest.fail('Mutation/training during estimate')
    monkeypatch.setattr(conditional,'train_conditional_term',forbidden)
    monkeypatch.setattr(CategoricalEncoder,'fit',forbidden);monkeypatch.setattr(torch.optim,'Adam',forbidden)
    state=random.getstate(),np.random.get_state(),torch.get_rng_state()
    flags=torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.deterministic
    a=model.estimate(test);b=model.estimate(test)
    assert a.entropy_nats==b.entropy_nats
    assert [t.contribution_nats for t in a.terms]==[t.contribution_nats for t in b.terms]
    assert random.getstate()==state[0];np.testing.assert_array_equal(np.random.get_state()[1],state[1][1])
    assert np.random.get_state()[2:]==state[1][2:] and torch.equal(torch.get_rng_state(),state[2])
    assert flags==(torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.deterministic)
    assert mappings=={k:dict(v) for k,v in model._encoder.mappings.items()}
    for call,before in zip(calls,snapshots):
        assert not call[4].training and all(p.grad is None for p in call[4].parameters())
        for k,v in call[4].state_dict().items():assert torch.equal(v,before[k])


@pytest.mark.parametrize('order',[(),[],('y0','y0'),('x0',),('bad',),(True,)])
def test_invalid_context(order):
    (train,val,_),config,alphabets=scenario()
    with pytest.raises(ValueError):
        conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=order,alphabets=alphabets).fit(train,val)


def test_unknown_x_and_fit_roles_refit_and_failure_atomicity(monkeypatch):
    (train,val,test),config,alphabets=scenario();calls=[]
    bad=replace(config,component_order=('absent',))
    with pytest.raises(ValueError):conditional.ConditionalNeuralJointEntropyEstimator(bad,conditioning_order=('y0',),alphabets=alphabets).fit(train,val)
    model=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y0',),alphabets=alphabets)
    with pytest.raises(RuntimeError):model.estimate(test)
    with pytest.raises(ValueError):model.fit(train,test)
    with pytest.raises(TypeError):model.fit(train,val,test=test)
    partial=[];first=controlled(partial)
    def fail(*a,**k):
        if partial:raise RuntimeError('failed second term')
        return first(*a,**k)
    monkeypatch.setattr(conditional,'train_conditional_term',fail)
    with pytest.raises(RuntimeError):model.fit(train,val)
    assert not model._fitted and not hasattr(model,'_results')
    assert len(partial)==1
    monkeypatch.setattr(conditional,'train_conditional_term',controlled(calls));model.fit(train,val)
    with pytest.raises(RuntimeError):model.fit(train,val)


@pytest.mark.parametrize('change',[{'dataset_id':'wrong'},{'split_id':'wrong'},{'n_samples':24},{'variable_ids':('y0','x1','x0','y1','unused')},{'role':'validation'}])
def test_test_identity_mismatch(change,monkeypatch):
    (train,val,test),config,alphabets=scenario();monkeypatch.setattr(conditional,'train_conditional_term',controlled([]))
    model=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y0',),alphabets=alphabets).fit(train,val)
    with pytest.raises(ValueError):model.estimate(replace(test,**change))


def test_overlap_coverage_and_unknown_y_rejected_before_forward(monkeypatch):
    (train,val,test),config,alphabets=scenario();calls=[]
    monkeypatch.setattr(conditional,'train_conditional_term',controlled(calls))
    model=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y0',),alphabets=alphabets)
    with pytest.raises(ValueError):model.fit(train,replace(val,row_indices=train.row_indices[:len(val.values)]))
    model.fit(train,val)
    for bad in [replace(test,row_indices=train.row_indices[:len(test.values)]),replace(test,row_indices=val.row_indices[:len(test.values)]),
                replace(test,values=test.values[:-1],row_indices=test.row_indices[:-1])]:
        with pytest.raises(ValueError):model.estimate(bad)
    values=test.values.copy();values[0,2]=99
    with pytest.raises(ValueError,match='Unknown category'):model.estimate(replace(test,values=values))


def test_fit_mappings_train_only_and_test_values_not_used_for_selection(monkeypatch):
    (train,val,test),config,alphabets=scenario(x=('x',),y=('y',));calls=[]
    monkeypatch.setattr(conditional,'train_conditional_term',controlled(calls,'binary'))
    model=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y',),alphabets=alphabets).fit(train,val)
    assert model._encoder._train_rows==frozenset(train.row_indices)
    changed=test.values.copy();changed[:,0]=0;changed[:,1]=0
    low=model.estimate(replace(test,values=changed))
    changed[:,0]=1;high=model.estimate(replace(test,values=changed))
    assert low.entropy_nats<high.entropy_nats and len(calls)==1


def test_estimate_failure_restores_rng_and_backend(monkeypatch):
    (train,val,test),config,alphabets=scenario(x=('x',),y=('y',))
    monkeypatch.setattr(conditional,'train_conditional_term',controlled([]))
    model=conditional.ConditionalNeuralJointEntropyEstimator(config,conditioning_order=('y',),alphabets=alphabets).fit(train,val)
    def fail(*a,**k):
        random.random();np.random.random();torch.rand(1)
        raise RuntimeError('evaluation failure')
    monkeypatch.setattr(conditional,'_test_ce',fail)
    state=random.getstate(),np.random.get_state(),torch.get_rng_state()
    flags=torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.deterministic
    with pytest.raises(RuntimeError,match='evaluation failure'):model.estimate(test)
    assert random.getstate()==state[0];np.testing.assert_array_equal(np.random.get_state()[1],state[1][1])
    assert np.random.get_state()[2:]==state[1][2:] and torch.equal(torch.get_rng_state(),state[2])
    assert flags==(torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.deterministic)


@pytest.mark.parametrize('context,conditioning',[(('y',),('y',)),(('y',),('x',)),(('x',),('x',)),(('bad',),('bad',)),(('y','y'),('y','y')),([],('y',))])
def test_trainer_external_context_contract(context,conditioning):
    (train,val,_),config,alphabets=scenario(x=('x',),y=('y',))
    encoder=CategoricalEncoder.fit(train,policy='declared',alphabets=alphabets)
    if context==('y',) and conditioning==('y',):
        result=training.train_conditional_term(train,val,encoder=encoder,config=config,target='x',conditioning=conditioning,context=context)
        assert result.conditioning==('y',) and result.epochs_trained==1
    else:
        with pytest.raises(ValueError):training.train_conditional_term(train,val,encoder=encoder,config=config,target='x',conditioning=conditioning,context=context)
