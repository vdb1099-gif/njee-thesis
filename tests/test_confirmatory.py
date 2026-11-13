"""Synthetic numerical fixtures only; no real final outcomes are loaded."""
import copy
import math
import statistics

import pytest

from experiments.final_protocol import PRIMARY, cells
from experiments.neural_baseline import VARIANTS
from src.metrics.confirmatory import paired_intervals


def records():
    result=[]
    for cell in cells():
        if cell['cell'] not in PRIMARY:continue
        for i in range(30):
            for variant in VARIANTS:
                # NJEE abs error i+1, classical abs error 1 => D=i.
                estimate=2+(i+1 if variant=='njee' else 1)
                result.append({'campaign_id':cell['campaign_id'],'replica_id':f'replica-{i:06d}',
                               'variant':variant,'status':'complete','entropy_nats':estimate,
                               'ground_truth_nats':2,'signed_error_nats':estimate-2})
    return result


def test_numerical_fixture_and_bonferroni():
    rows=records();results=paired_intervals(list(reversed(rows)))
    assert len(results)==16
    sd=statistics.stdev(range(30));se=sd/math.sqrt(30)
    # Independently frozen SciPy reference t.ppf(.9984375,29).
    critical=3.2234827906278363
    for r in results:
        assert r['status']=='complete' and r['n_pairs']==30 and r['df']==29
        assert r['family_size']==16 and r['family_alpha']==.05
        assert r['individual_alpha']==.003125 and r['confidence_level']==.996875
        assert r['mean_difference_nats']==14.5
        assert r['sample_std_nats']==pytest.approx(sd)
        assert r['critical_value']==pytest.approx(critical,abs=1e-10)
        assert r['lower_nats']==pytest.approx(14.5-critical*se,abs=1e-10)
        assert r['upper_nats']==pytest.approx(14.5+critical*se,abs=1e-10)


def test_missing_failed_and_empty_pairs_never_shrink_family():
    rows=records();rows.pop(0)
    result=paired_intervals(rows)
    assert sum(r['status']=='incomplete' for r in result)==4
    for r in result[:4]:
        assert r['n_pairs']==29 and r['df'] is None and r['lower_nats'] is None
        assert r['family_size']==16 and r['missing_replica_ids']==['replica-000000']
    rows=records();rows[1]['status']='failed'
    assert paired_intervals(rows)[0]['status']=='incomplete'
    assert len(paired_intervals([]))==16
    assert all(r['n_pairs']==0 and r['lower_nats'] is None for r in paired_intervals([]))


def test_absolute_errors_direction_zero_variance():
    rows=records()
    for r in rows:
        r['signed_error_nats']=-.25 if r['variant']=='njee' else .5
        r['entropy_nats']=r['ground_truth_nats']+r['signed_error_nats']
    for r in paired_intervals(rows):
        assert r['mean_difference_nats']==-.25
        assert r['lower_nats']==r['upper_nats']==-.25
        assert r['zero_observed_variance']


@pytest.mark.parametrize('failure',['duplicate','nonfinite','error','truth','seed','scenario','unplanned'])
def test_malformed_pairing_fails(failure):
    rows=records()
    if failure=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif failure=='nonfinite':rows[0]['entropy_nats']=math.nan
    elif failure=='error':rows[0]['signed_error_nats']=999
    elif failure=='truth':rows[0]['ground_truth_nats']=3;rows[0]['signed_error_nats']=0
    elif failure=='seed':rows[0]['data_seed']=123;rows[1]['data_seed']=456
    elif failure=='scenario':rows[0]['scenario_id']='x';rows[1]['scenario_id']='y'
    elif failure=='unplanned':rows[0]['replica_id']='replica-000030'
    with pytest.raises(ValueError):paired_intervals(rows)
