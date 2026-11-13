"""Exact temporal equations and finite-state references, no Monte Carlo accuracy tests."""
import math
import numpy as np
import pytest

from src.simulation.coupled_binary import (CoupledBinaryModel,transition_matrix,stationary_reference,
                                          reference_te,coupling_bound)
from src.utils.temporal import TemporalEmbeddingSpec


@pytest.mark.parametrize('c',[0.,1.])
@pytest.mark.parametrize('delay',[1,3])
def test_exact_equations_with_controlled_switches_noises(monkeypatch,c,delay):
    calls=[]
    class RNG:
        def binomial(self,n,p,size=None):
            calls.append((n,p,size))
            if len(calls)==1:return np.array([0,1,0,1,0,1,0])[:size]
            if len(calls)==2:return np.array([1,0,0,1,0,0,1])[:size]
            return np.full(size,int(c),dtype=int)
    monkeypatch.setattr(np.random,'default_rng',lambda seed:RNG())
    model=CoupledBinaryModel(c,delay,.1,.2)
    x,y,meta=model.generate(8,seed=0,burn_in=0,initial='zeros')
    assert calls==[(1,.1,7),(1,.2,7),(1,c,7)]
    ex=np.array([0,1,0,1,0,1,0]);ey=np.array([1,0,0,1,0,0,1])
    for t in range(1,8):
        assert x[t]==x[t-1]^ex[t-1]
        parent=(x[t-delay] if t>=delay else 0) if c else y[t-1]
        assert y[t]==parent^ey[t-1]
    assert meta['physical_time_start']==0 and meta['initial_x_history']==(0,)*delay


def test_reproduction_burn_in_initialization_and_rng_isolation():
    m=CoupledBinaryModel(.3,2)
    state=np.random.get_state()
    a,b,meta=m.generate(6,seed=42,burn_in=4)
    c,d,again=m.generate(6,seed=42,burn_in=4)
    np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    assert meta==again and meta['physical_time_start']==4 and meta['physical_time_stop']==9
    whole_x,whole_y,_=m.generate(10,seed=42,burn_in=0)
    np.testing.assert_array_equal(a,whole_x[4:]);np.testing.assert_array_equal(b,whole_y[4:])
    assert not a.flags.writeable and not b.flags.writeable
    np.testing.assert_array_equal(np.random.get_state()[1],state[1])
    assert coupling_bound(m,1000,6)<1e-50


def test_transition_probabilities_and_stationarity():
    m=CoupledBinaryModel(.3,1,.1,.2);p=transition_matrix(m,1).toarray()
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-15)
    # State (X0,Y0)=(1,0): X1~Bern(.9), Y1~Bern(.2+.6*.3).
    assert p[2,3]==pytest.approx(.9*.38)
    pi,meta=stationary_reference(m,1)
    np.testing.assert_allclose(pi@p,pi,atol=1e-12)
    assert meta['tv_truncation_bound']<=1e-13 and meta['stationarity_l1_residual']<1e-10


@pytest.mark.parametrize('delay',[1,2,3])
def test_analytic_iid_driver_copy_channel_reference(delay):
    m=CoupledBinaryModel(1.,delay,.5,.1)
    r=reference_te(m,TemporalEmbeddingSpec((delay,),(1,)))
    truth=math.log(2)+.1*math.log(.1)+.9*math.log(.9)
    assert r['transfer_entropy_nats']==pytest.approx(truth,abs=1e-11)
    assert r['status']=='numerical_stationary_enumeration' and r['entropy_truncation_error_bound_nats']<1e-9
    reverse=reference_te(m,TemporalEmbeddingSpec((delay,),(1,)),direction='Y_to_X')
    assert reverse['transfer_entropy_nats']==0 and reverse['status']=='analytic_zero'


def test_null_and_misspecified_reverse_reference_are_distinct():
    spec=TemporalEmbeddingSpec((1,),(2,))
    assert reference_te(CoupledBinaryModel(0.,1),spec)['transfer_entropy_nats']==0
    m=CoupledBinaryModel(.8,1)
    assert reference_te(m,spec,direction='Y_to_X')['transfer_entropy_nats']>0
    assert reference_te(m,TemporalEmbeddingSpec(),direction='Y_to_X')['transfer_entropy_nats']==0
    with pytest.raises(RuntimeError):stationary_reference(m,1,1e-13,1)
    with pytest.raises(ValueError):transition_matrix(m,9)


def test_largest_frozen_embedding_reference_is_tractable_and_labelled_numerical():
    r=reference_te(CoupledBinaryModel(.8,5),TemporalEmbeddingSpec((5,6),(1,2)))
    assert r['n_states']==4096 and r['memory']==6
    assert 0<r['transfer_entropy_nats']<math.log(2)
    assert r['status']=='numerical_stationary_enumeration'
    assert r['entropy_truncation_error_bound_nats']<1e-9


@pytest.mark.parametrize('kwargs',[{'coupling':True,'delay':1},{'coupling':-1,'delay':1},
                                  {'coupling':.1,'delay':True},{'coupling':.1,'delay':0},
                                  {'coupling':.1,'delay':1,'noise_x':0},
                                  {'coupling':.1,'delay':1,'noise_y':float('nan')}])
def test_invalid_models(kwargs):
    with pytest.raises(ValueError):CoupledBinaryModel(**kwargs)


@pytest.mark.parametrize('kwargs',[{'n':0,'seed':0},{'n':True,'seed':0},{'n':3,'seed':True},
                                  {'n':3,'seed':0,'burn_in':-1},{'n':3,'seed':0,'initial':'stationary_claim'}])
def test_invalid_generation(kwargs):
    with pytest.raises(ValueError):CoupledBinaryModel(.1,1).generate(**kwargs)
