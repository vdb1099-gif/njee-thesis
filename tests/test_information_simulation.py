"""Exact binary formulas and deterministic generator contracts, no convergence tests."""
from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from src.simulation.information import InformationModel


def h(p):
    return -sum(q*math.log(q) for q in (p,1-p) if q)


@pytest.mark.parametrize('epsilon',[0.,.05,.15,.30,.5,1.])
def test_binary_channel_analytic_truth(epsilon):
    model = InformationModel('bsc',0.,epsilon)
    assert sum(model.pmf().values()) == pytest.approx(1,abs=1e-15)
    assert model.truths()['information_nats'] == pytest.approx(math.log(2)-h(epsilon),abs=1e-14)
    assert set(model.pmf()) == {(0,0),(0,1),(1,0),(1,1)}
    for (x,y),p in model.pmf().items():
        assert p == pytest.approx(.5*(1-epsilon if x==y else epsilon),abs=1e-15)


@pytest.mark.parametrize('ex',[0.,.15,.5,1.])
@pytest.mark.parametrize('ey',[0.,.15,.5,1.])
def test_chain_enumeration_and_conditional_formula(ex,ey):
    model = InformationModel('chain',ex,ey)
    truth = model.truths()
    # I(X;Y|Z)=H(Y|Z)-H(Y|X,Z)=h(ex+ey-2ex*ey)-h(ey).
    assert truth['information_nats'] == pytest.approx(h(ex+ey-2*ex*ey)-h(ey),abs=1e-14)
    assert truth['first_entropy_nats'] == pytest.approx(h(ex),abs=1e-14)
    assert sum(model.pmf().values()) == pytest.approx(1)
    assert len(model.pmf()) == 8


def test_conditional_independence_is_not_marginal_independence():
    model = InformationModel('fork',.15,.15)
    assert model.truths()['information_nats'] == 0.
    marginal_mi = model.entropy(('X',))+model.entropy(('Y',))-model.entropy(('X','Y'))
    assert marginal_mi == pytest.approx(math.log(2)-h(.255),abs=1e-14)
    assert marginal_mi > 0
    pmf = model.pmf()
    for x in (0,1):
        for y in (0,1):
            for z in (0,1):
                pxz = sum(p for (a,b,c),p in pmf.items() if a==x and c==z)
                pyz = sum(p for (a,b,c),p in pmf.items() if b==y and c==z)
                assert pmf[x,y,z]*.5 == pytest.approx(pxz*pyz,abs=1e-15)
    assert InformationModel('fork',.15,.5).truths()['information_nats'] == 0.


@pytest.mark.parametrize('family',['bsc','chain','fork'])
def test_generator_local_rng_support_and_reproduction(family):
    model = InformationModel(family,0 if family=='bsc' else .15,.3)
    state = np.random.get_state()
    a = model.generate(8,seed=42)
    np.testing.assert_array_equal(a,model.generate(8,seed=42))
    assert not np.array_equal(a,model.generate(8,seed=43))
    np.testing.assert_array_equal(np.random.get_state()[1],state[1])
    assert np.random.get_state()[2:] == state[2:]
    assert a.shape == (8,len(model.variable_ids)) and a.dtype==np.int64
    assert set(np.unique(a)) <= {0,1}
    with pytest.raises(FrozenInstanceError): model.epsilon_x = .2


def test_exact_rng_calls_show_independent_iid_noise_draws(monkeypatch):
    calls = []
    class RNG:
        def binomial(self,n,p,size):
            calls.append((n,p,size))
            return np.array([0,1,0,1]) if len(calls)==1 else np.array([0,0,1,1])
    monkeypatch.setattr(np.random,'default_rng',lambda seed:RNG())
    a = InformationModel('chain',.15,.3).generate(4,seed=1)
    assert calls == [(1,.5,4),(1,.15,4),(1,.3,4)]
    np.testing.assert_array_equal(a[:,0],a[:,2]^np.array([0,0,1,1]))
    np.testing.assert_array_equal(a[:,1],a[:,0]^np.array([0,0,1,1]))


@pytest.mark.parametrize('family,ex,ey',[('bad',.1,.1),('bsc',.1,.1),('chain',True,.1),
                                      ('fork',float('nan'),.1),('chain',.1,float('inf')),('fork',-.1,.2)])
def test_invalid_parameters(family,ex,ey):
    with pytest.raises(ValueError): InformationModel(family,ex,ey)


@pytest.mark.parametrize('n,seed',[(0,1),(True,1),(3.5,1),(4,True),(4,-1),(4,2**32)])
def test_invalid_generation(n,seed):
    with pytest.raises(ValueError): InformationModel('bsc',0,.5).generate(n,seed=seed)
