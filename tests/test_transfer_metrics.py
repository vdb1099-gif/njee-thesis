"""Exact path-surrogate, multiplicity and replica reporting fixtures."""
from types import SimpleNamespace
import numpy as np
import pytest

from src.utils.temporal_surrogates import conditional_source_path,circular_shift_source,block_permute_source
from src.simulation.coupled_binary import CoupledBinaryModel
from src.metrics.transfer import family_decisions,select_lag,summarize_transfer,frozen_path_test
from src.utils.categorical import CategoricalBatch
from src.utils.temporal import TemporalEmbeddingSpec


def test_conditional_path_prefix_target_and_reverse_coupled_kernel():
    source=np.zeros(10,dtype=int);target=np.ones(10,dtype=int)
    class RNG:
        def random(self):return .5
    m=CoupledBinaryModel(1.,2,.1,.1)
    reverse=conditional_source_path(source,target,model=m,direction='Y_to_X',cutoff=3,rng=RNG())
    np.testing.assert_array_equal(reverse[:4],source[:4])
    np.testing.assert_array_equal(reverse[4:],np.ones(6))
    forward=conditional_source_path(source,target,model=m,direction='X_to_Y',cutoff=3,rng=RNG())
    assert not forward.any()  # forward null is no-coupling, NOT the alternative Y kernel
    np.testing.assert_array_equal(target,np.ones(10))


def test_exploratory_shifts_and_blocks_never_iid_rows():
    a=np.arange(12)
    shifted=circular_shift_source(a,shift=4,minimum_shift=3)
    np.testing.assert_array_equal(shifted,np.roll(a,4))
    shuffled=block_permute_source(a,block_length=3,seed=42).reshape(-1,3)
    assert {tuple(row) for row in shuffled}=={tuple(row) for row in a.reshape(-1,3)}
    assert np.all(np.diff(shuffled,axis=1)==1)
    with pytest.raises(ValueError):block_permute_source(a[:-1],block_length=3,seed=42)
    with pytest.raises(ValueError):circular_shift_source(a,shift=1,minimum_shift=3)


def test_fixed_family_no_shrinking_and_exact_p_thresholds():
    r=family_decisions([.001,.01]+[1.]*6)
    assert r['reject']==[True,False]+[False]*6 and r['adjusted_p_values'][0]==.008
    assert not family_decisions([None]+[.001]*7)['complete']
    assert family_decisions([None]+[.001]*7)['any_reject'] is None
    with pytest.raises(ValueError):family_decisions([.001])


def test_conditional_path_null_tail_probability_exact_enumeration():
    # Given X_V=0 and flip probability .1, enumerate the two-step path law.
    states=[(0,0),(0,1),(1,0),(1,1)]
    masses=[.9*.9,.9*.1,.1*.1,.1*.9]
    assert sum(masses)==pytest.approx(1)
    statistic=lambda path:2*path[0]+path[1]
    tails=[sum(p for s,p in zip(states,masses) if statistic(s)>=statistic(observed)) for observed in states]
    for alpha in (.05,.1,.25,.5):
        assert sum(p for p,tail in zip(masses,tails) if tail<=alpha)<=alpha+1e-15


def test_lag_selection_uses_validation_not_largest_test_te():
    rows=[{'embedding':'canonical','lag':lag,'status':'complete','validation_score_nats':score,'te_nats':test}
          for lag,score,test in [(1,.2,99),(2,.5,-.25),(3,.5,100)]]
    assert select_lag(rows,[1,2,3])['selected_lag']==2
    assert select_lag(rows,[1,2,3])['test_te_nats']==-.25
    rows[0]['status']='failed'
    assert select_lag(rows,[1,2,3])['status']=='incomplete'


def test_frozen_test_changes_only_source_columns_and_never_refits():
    a=np.zeros(15,dtype=int);target=np.ones(15,dtype=int)
    batch=CategoricalBatch(np.array([[1,0,1],[1,0,1]]),'d',('future','source','own'),np.array([3,4]),'test','s',5)
    seen=[]
    def evaluate(b):
        np.testing.assert_array_equal(b.values[:,[0,2]],batch.values[:,[0,2]])
        assert b.dataset_id=='d' and b.split_id=='s'
        seen.append(b)
        return SimpleNamespace(information_nats=-.5)
    model=SimpleNamespace(_parts=SimpleNamespace(test=batch),_model=SimpleNamespace(estimate=evaluate))
    report=SimpleNamespace(transfer_entropy_nats=.5,embedding=TemporalEmbeddingSpec(),
                           partitions=(None,SimpleNamespace(target_timestamps=(5,)),
                                       SimpleNamespace(target_timestamps=(10,11),observation_timestamps=((10,9,9),(11,10,10)))))
    r=frozen_path_test([model],[report],a,target,process=CoupledBinaryModel(.1,1),direction='X_to_Y',seed=42,draws=19)
    assert r['p_values']==[.05] and r['refits']==0 and len(seen)==19
    assert batch.values[:,1].tolist()==[0,0]
    with pytest.raises(KeyboardInterrupt):
        frozen_path_test([model],[report],a,target,process=CoupledBinaryModel(.1,1),direction='X_to_Y',seed=42,draws=19,cancelled=lambda:True)


def fixture_rows():
    jobs=[{'job_id':f'{d}-{lag}','direction':d,'embedding':'canonical','lag':lag}
          for d in ('X_to_Y','Y_to_X') for lag in (1,2)]
    rows=[]
    for i in range(3):
        for j in jobs:
            v=(-.5,0.,1.)[i] if j['direction']=='X_to_Y' else -.1
            rows.append({**j,'replica_id':str(i),'status':'complete','te_nats':v,'reference_nats':0.,
                         'signed_error_nats':v,'runtime_seconds':1.,'validation_score_nats':j['lag'],
                         'test_status':'complete','p_value':.001 if i==0 else 1.,'test_timestamps':[10,11]})
    return rows,jobs


def test_replica_metrics_signed_errors_and_missing_outcomes():
    rows,jobs=fixture_rows()
    r=summarize_transfer(rows,['0','1','2'],jobs,true_delay=2,coupling=.3,candidates=[1,2],draws=5,seed=1)
    s=r['summaries'][0]
    assert s['metrics']['bias_nats']==pytest.approx(1/6)
    assert s['negative_frequency']==1/3
    assert r['accuracy']['lag_correct']['rate']==1.
    assert r['accuracy']['direction_correct']['rate']==2/3
    assert r['accuracy']['false_link']['rate'] is None
    assert r['direction_difference']['n_pairs']==3
    r=summarize_transfer(rows,['0','1','2'],jobs,true_delay=1,coupling=0.,candidates=[1,2],draws=5,seed=1)
    assert r['accuracy']['false_link']['rate']==1/3
    assert r['accuracy']['lag_correct']['n_eligible']==0
    rows[0].update(status='failed',test_status='failed')
    r=summarize_transfer(rows,['0','1','2'],jobs,true_delay=1,coupling=0.,candidates=[1,2],draws=5,seed=1)
    assert r['accuracy']['false_link']['n_evaluated']==2
    assert r['accuracy']['false_link']['planned_bounds']==[0,1/3]
    with pytest.raises(ValueError):summarize_transfer(rows[:-1],['0','1','2'],jobs,true_delay=1,coupling=0.,candidates=[1,2],draws=5,seed=1)
