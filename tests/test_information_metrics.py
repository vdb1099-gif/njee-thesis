"""Exact statistics, paired units and frozen-model permutation calibration."""
from itertools import combinations
from types import SimpleNamespace
import math

import numpy as np
import pytest

from src.metrics import information as metrics
from src.utils.categorical import CategoricalBatch


def test_classical_formulas_and_signed_miller_madow():
    table = np.array([[0,0],[0,1],[1,0],[1,1]])
    assert metrics.classical_information(table,('X','Y'),conditional=False,method='plug_in')['information_nats'] == pytest.approx(0.)
    result = metrics.classical_information(table,('X','Y'),conditional=False,method='miller_madow')
    assert result['information_nats'] == pytest.approx(-1/8)
    copy = np.array([[0,0],[1,1]])
    assert metrics.classical_information(copy,('X','Y'),conditional=False,method='plug_in')['information_nats'] == pytest.approx(math.log(2))
    xyz = np.array([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)])
    assert metrics.classical_information(xyz,('X','Y','Z'),conditional=True,method='plug_in')['information_nats'] == pytest.approx(0.)
    # Observed-K correction: (Kxz+Kyz-Kz-Kxyz)/(2n)=(4+4-2-8)/16.
    assert metrics.classical_information(xyz,('X','Y','Z'),conditional=True,method='miller_madow')['information_nats'] == pytest.approx(-.125)


@pytest.mark.parametrize('conditional',[False,True])
def test_permutation_respects_strata_frozen_state_and_signed_statistic(conditional):
    values = np.array([[0,0,0],[1,1,0],[0,1,1],[1,0,1]])
    test = CategoricalBatch(values,'d',('X','Y','Z'),np.arange(4),'test','s',4)
    calls = []
    class Frozen:
        def estimate(self,batch):
            calls.append(batch)
            assert batch.dataset_id=='d' and batch.split_id=='s' and batch.role=='test'
            np.testing.assert_array_equal(batch.values[:,[0,2]],values[:,[0,2]])
            for z in ((0,1) if conditional else (None,)):
                rows = values[:,2]==z if z is not None else np.ones(4,dtype=bool)
                np.testing.assert_array_equal(np.sort(batch.values[rows,1]),np.sort(values[rows,1]))
            return SimpleNamespace(information_nats=-.25)
    before = test.values.copy()
    a = metrics.heldout_permutation_test(Frozen(),test,conditional=conditional,seed=4,draws=9,observed=-.25)
    assert a['p_value']==1 and not a['reject'] and a['refits']==0 and len(calls)==9
    b = metrics.heldout_permutation_test(Frozen(),test,conditional=conditional,seed=4,draws=9,observed=-.25)
    assert a==b
    np.testing.assert_array_equal(test.values,before)


def test_plus_one_upper_tail_and_cancellation():
    test=CategoricalBatch(np.array([[0,0],[1,1]]),'d',('X','Y'),np.arange(2),'test','s',2)
    frozen=SimpleNamespace(estimate=lambda b:SimpleNamespace(information_nats=-1.))
    r=metrics.heldout_permutation_test(frozen,test,conditional=False,seed=0,draws=19,observed=1.)
    assert r['p_value']==.05 and r['reject']
    with pytest.raises(KeyboardInterrupt):
        metrics.heldout_permutation_test(frozen,test,conditional=False,seed=0,draws=19,observed=1.,cancelled=lambda:True)


def test_exact_exchangeability_reference_does_not_use_positive_threshold():
    # Six equally likely arrangements of two ones under a finite exchangeable null.
    arrangements=[]
    for ones in combinations(range(4),2):
        y=np.zeros(4,dtype=int);y[list(ones)]=1;arrangements.append(y)
    statistic=lambda y: int(y[0]+2*y[1]+4*y[2]+8*y[3])
    ts=[statistic(y) for y in arrangements]
    ps=[sum(t>=observed for t in ts)/len(ts) for observed in ts]
    assert sum(p<=1/3 for p in ps)/6 <= 1/3
    assert all(t>0 for t in ts)  # positivity would reject every null arrangement


def records():
    rows=[]
    variants=('neural','plug_in_full','miller_madow_full','plug_in_train','miller_madow_train')
    for i,v in [(i,v) for i in range(3) for v in variants]:
        estimate=(-1.,0.,2.)[i]
        rows.append({'campaign_id':'cell','scenario_id':'hash','replica_id':str(i),'variant':v,
                     'entropy_nats':estimate,'ground_truth_nats':0.,'signed_error_nats':estimate,
                     'first_entropy_nats':estimate+3.,'second_entropy_nats':3.,
                     'first_truth_nats':1.,'second_truth_nats':1.,'status':'complete',
                     'runtime_seconds':1.,'fit_runtime_seconds':.5,'inference_runtime_seconds':.5,
                     'is_null':True,'test_status':'complete' if v=='neural' else 'not_attempted',
                     'p_value':.01 if i==0 else .8,'reject':i==0})
    return rows


def test_metrics_exact_intervals_failures_and_paired_units():
    rows=records()
    r=metrics.summarize_information(rows,['0','1','2'],bootstrap_draws=10,bootstrap_seed=3)
    s=next(s for s in r['summaries'] if s['variant']=='neural')
    assert s['bias_nats']==pytest.approx(1/3) and s['mae_nats']==1
    assert s['rmse_nats']==pytest.approx(math.sqrt(5/3))
    assert s['sample_std_nats']==pytest.approx(np.std([-1,0,2],ddof=1))
    assert s['negative_frequency']==1/3
    assert s['secondary_clipped_only']['metrics']['bias_nats']==pytest.approx(2/3)
    assert r['false_positive_analysis']['rate']==1/3
    assert all(p['n_paired_successful']==3 for p in r['paired_comparisons'])
    assert all(p['bootstrap']['absolute_error_difference_nats']['unit']=='paired_replica_id' for p in r['paired_comparisons'])
    rows[0].update(status='failed',test_status='failed')
    r=metrics.summarize_information(rows,['0','1','2'],bootstrap_draws=10,bootstrap_seed=3)
    s=next(s for s in r['summaries'] if s['variant']=='neural')
    assert s['n_failed']==1 and s['n_successful']==2
    assert r['false_positive_analysis']['n_untested']==1
    assert r['false_positive_analysis']['planned_rate_bounds']==[0,1/3]
    assert all(p['n_paired_successful']==2 for p in r['paired_comparisons'])


def test_error_correlations_and_undefined_cases():
    a=metrics.error_correlation([(1,2),(2,4),(3,6),(4,8)],draws=10,seed=4)
    assert a['pearson_r']==pytest.approx(1)
    assert metrics.error_correlation([(1,8),(2,6),(3,4),(4,2)],draws=10,seed=4)['pearson_r']==pytest.approx(-1)
    assert metrics.error_correlation([(1,3),(1,4),(1,5)],draws=10,seed=4)['pearson_r'] is None
    assert metrics.error_correlation([],draws=10,seed=4)['pearson_r'] is None
    assert metrics.error_correlation([(1,2)],draws=10,seed=4)['lower'] is None
    assert metrics.binomial_interval(0,50)['lower']==0
    assert metrics.binomial_interval(50,50)['upper']==1


@pytest.mark.parametrize('kind',['missing','duplicate','nonfinite','arithmetic','decision'])
def test_invalid_aggregate_records(kind):
    rows=records()
    if kind=='missing': rows.pop()
    if kind=='duplicate': rows.append(rows[0])
    if kind=='nonfinite': rows[0]['entropy_nats']=float('nan')
    if kind=='arithmetic': rows[0]['first_entropy_nats']=999
    if kind=='decision': rows[0]['reject']=False
    with pytest.raises(ValueError): metrics.summarize_information(rows,['0','1','2'],bootstrap_draws=2)
