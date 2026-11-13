"""Frozen matrix/allocation and portable preparation: never launch training."""
import json
from pathlib import Path

import pytest
import yaml

from experiments import final_protocol as protocol


@pytest.fixture(scope='module')
def plan():
    return protocol.preview()


def test_exact_matrix_seed_mapping_and_counts(plan):
    expected=[]
    for generator,param,value in [('independent_bernoulli','p',.5),('binary_chain','epsilon',.15)]:
        for d in [4,8,12]:
            for n in [200,800,3200]:expected.append((generator,param,value,d,n))
    for epsilon in [.05,.30]:
        for d in [4,8,12]:expected.append(('binary_chain','epsilon',epsilon,d,800))
    for d in [4,8,12]:expected.append(('independent_bernoulli','p',.2,d,800))
    assert len(plan['cells'])==27
    identities=[];fits=0
    for i,(cell,e) in enumerate(zip(plan['cells'],expected)):
        generator,param,value,d,n=e
        assert cell['cell']==f'F{i+1:02d}' and cell['master_seed']==130100+i
        assert cell['data']=={'generator':generator,'params':{param:value},'d':d,'n':n}
        assert len(cell['replicas'])==30
        for r,replica in enumerate(cell['replicas']):
            assert replica['replica_id']==f'replica-{r:06d}'
            identities.append((cell['campaign_id'],replica['replica_id']))
            assert len(replica['terms'])==d-1
            fits+=len(replica['terms'])
    assert len(identities)==len(set(identities))==810 and fits==5670
    assert plan['expected']=={'cells':27,'replicas':810,'conditional_fits':5670,'chain_contributions':6480,'estimator_records':4050}
    assert plan['historical_seed_check']=={'prior_values_checked':727,'collisions':[]}
    assert plan['execution_order']==['F01','F02','F17','F18',*[f'F{i:02d}' for i in range(3,17)],*[f'F{i:02d}' for i in range(19,28)]]


def test_frozen_policy_and_stable_hashes(plan):
    for cell in plan['cells']:
        neural=cell['baseline']['neural']
        expected={'split_fractions':[.6,.2,.2],'alphabet_policy':'declared','marginal_method':'train_plugin',
                  'hidden_sizes':[16],'activation':'relu','optimizer':'adam','learning_rate':.01,'batch_size':64,
                  'max_epochs':20,'patience':5,'min_delta':.0001,'device':'cpu','reproducibility_mode':'strict','num_workers':0}
        assert {k:neural[k] for k in expected}==expected
        assert neural['component_order']==list(range(cell['data']['d']))
        assert cell['baseline_hash']==protocol.digest(cell['baseline'])
        assert cell['campaign_hash']==protocol.digest(cell['campaign'])
        assert cell['campaign']['purpose']=='final'
        assert cell['campaign']['aggregation']['bootstrap_seed']==271813
        assert cell['campaign']['aggregation']['bootstrap_draws']==10000
        assert len(cell['campaign']['aggregation']['paired_comparisons'])==4
    assert plan==protocol.preview()
    moved=protocol.preview('elsewhere/final')
    assert moved['protocol_hash']==plan['protocol_hash']
    assert [e['scientific_hash'] for e in moved['cells']]==[e['scientific_hash'] for e in plan['cells']]
    assert moved['cells'][0]['baseline_hash']!=plan['cells'][0]['baseline_hash']
    assert protocol.digest({'a':1,'b':2})==protocol.digest({'b':2,'a':1})


@pytest.mark.parametrize('path',['/home/user/final','C:/Users/user/final','C:\\Users\\user\\final','../results','.',''])
def test_no_nonportable_paths(path):
    with pytest.raises(ValueError):protocol.preview(path)


def test_preparation_no_outcome_access_or_training(tmp_path,monkeypatch):
    from experiments import neural_campaign, neural_baseline
    from src.estimators import neural_training
    from src.simulation import factory
    def forbidden(*a,**k):raise AssertionError('Outcome/training access during preparation')
    monkeypatch.setattr(neural_campaign,'run_campaign',forbidden)
    monkeypatch.setattr(neural_baseline,'run_baseline',forbidden)
    monkeypatch.setattr(neural_training,'train_conditional_term',forbidden)
    monkeypatch.setattr(factory,'generate_dataset',forbidden)
    original=Path.read_text
    def read(path,*a,**k):
        assert 'results.json' not in path.name and path.name not in ['evaluations.json','dataset.npz']
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'read_text',read)
    prep=tmp_path/'preparation'
    plan=protocol.materialize(prep,'results/new-final')
    assert len(list(prep.glob('*.yaml')))==54
    assert protocol.validate_bundle(prep)==plan
    assert not (tmp_path/'results/new-final').exists()
    with pytest.raises(FileExistsError):protocol.materialize(prep)
    path=next(prep.glob('*baseline.yaml'));cfg=yaml.safe_load(path.read_text());cfg['neural']['max_epochs']=60
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='differs'):protocol.validate_bundle(prep)


def test_manifest_tampering_and_prior_seed_collisions(tmp_path,plan):
    prep=tmp_path/'prep';protocol.materialize(prep)
    manifest=json.loads((prep/'protocol.json').read_text());manifest['execution_order'].reverse()
    (prep/'protocol.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='differs'):protocol.validate_bundle(prep)
    assert protocol.check_prior_seeds(plan,[1,2,3])['collisions']==[]
    with pytest.raises(ValueError,match='collisions'):
        protocol.check_prior_seeds(plan,[plan['cells'][0]['replicas'][0]['data_seed']])


def test_preparation_cannot_be_scientific_output(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):protocol.materialize('results/new/prep','results/new')
