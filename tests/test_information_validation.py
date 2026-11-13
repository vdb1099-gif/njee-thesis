"""Experiment 2 control tests use exact fixtures and mock fits, never scientific runs."""
from copy import deepcopy
from dataclasses import fields, replace
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from experiments import information_protocol as protocol
from experiments import information_validation as runner
from experiments.final_protocol import digest
from src.estimators.neural import NeuralEntropyReport, TermDiagnostic
from src.estimators.conditional_neural import ConditionalNeuralEntropyReport
from src.estimators.neural_information import InformationComponent, NeuralInformationReport
from src.simulation.information import InformationModel


@pytest.fixture(scope='module')
def full_plan():
    return protocol.preview()


def test_frozen_matrix_scope_effective_seeds_and_hashes(full_plan):
    p=full_plan
    assert p['expected']=={'cells':85,'replicas':4250,'null_replicas':1250,'conditional_fits':7250,
                           'estimator_records':21250,'null_surrogate_evaluations':1248750}
    assert protocol.load_definition()==protocol.definition()
    assert {c['n'] for c in p['cells']}=={250,500,1000,5000,10000}
    assert {c['master_seed'] for c in p['cells']}==set(range(160000,160085))
    assert len({c['cell_id'] for c in p['cells']})==85
    assert len(p['execution_order'])==len(set(p['execution_order']))==85
    seeds=[]
    for c in p['cells']:
        assert c['replicas']==50 and len(c['allocation'])==50
        assert c['truth']==InformationModel(**c['model']).truths()
        assert c['configuration_hash']==digest({k:v for k,v in c.items() if k not in ('allocation','configuration_hash')})
        for r in c['allocation']:
            seeds.extend(r[k] for k in ('data_seed','split_seed','training_seed','permutation_seed','bootstrap_seed'))
            for component in r['components']:
                seeds.append(component['training_seed'])
                for t in component['terms']:
                    seeds.extend((t['initialization'],t['shuffle']))
    assert len(seeds)==len(set(seeds)) and not set(seeds)&protocol._prior_seeds()
    for c in p['cells']:
        r=c['allocation'][0]
        _,components=protocol.effective_streams(InformationModel(**c['model']),r['training_seed'])
        assert components==r['components']
        config=protocol.neural_config(r['split_seed'],r['training_seed'])
        assert config.max_epochs==20 and config.patience==5 and config.device=='cpu'
    assert not Path(p['output_root']).exists()


@pytest.fixture
def tiny(monkeypatch):
    # Test-only synthetic protocol; production load/validate always uses the full definition.
    original=protocol.definition()
    d=deepcopy(original)
    d.update(sample_sizes=[10],replicas=2,mi_epsilon=[.5],chain_epsilon_x=[],fork_epsilon_x=[],
             bootstrap_draws=5,permutation_draws=3)
    monkeypatch.setattr(protocol,'definition',lambda:deepcopy(d))
    return protocol.preview()


def test_protocol_reproduction_preparation_integrity_no_overwrite(tiny,tmp_path):
    assert tiny==protocol.preview()
    bundle=tmp_path/'preparation'
    plan=protocol.materialize(bundle)
    assert protocol.validate_bundle(bundle)==plan
    assert len(list(bundle.iterdir()))==2
    with pytest.raises(FileExistsError): protocol.materialize(bundle)
    path=next(bundle.glob('*.yaml'))
    path.write_text('corrupt: true',encoding='utf-8')
    with pytest.raises(ValueError): protocol.validate_bundle(bundle)


def test_declared_collision_probing_and_definition_tamper(tiny,tmp_path,monkeypatch):
    first=tiny['cells'][0]['allocation'][0]
    assert first['nonces']['data_seed']==0
    monkeypatch.setattr(protocol,'_prior_seeds',lambda:frozenset({first['data_seed']}))
    plan=protocol.preview()
    resolved=plan['cells'][0]['allocation'][0]
    assert resolved['nonces']['data_seed']==1 and resolved['data_seed']!=first['data_seed']
    assert plan==protocol.preview()
    import yaml
    path=tmp_path/'bad.yaml';bad=protocol.definition();bad['neural']['max_epochs']=1
    path.write_text(yaml.safe_dump(bad),encoding='utf-8')
    with pytest.raises(ValueError):protocol.load_definition(path)


def test_manifest_tamper_and_extra_files(tiny,tmp_path):
    p=tmp_path/'prep';protocol.materialize(p)
    path=p/'protocol.json';saved=json.loads(path.read_text())
    saved['protocol_hash']='bad';path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):protocol.validate_bundle(p)
    path.write_text(json.dumps(tiny))
    (p/'extra.txt').write_text('extra')
    with pytest.raises(ValueError):protocol.validate_bundle(p)


@pytest.mark.parametrize('path',['C:/private/output','/home/user/output','../results','a\\b'])
def test_portable_path_rejection(tiny,path):
    with pytest.raises(ValueError):protocol.preview(path)


def mock_neural(monkeypatch):
    original=protocol.make_estimator
    calls=[]
    class Frozen:
        def __init__(self,model,config):
            self.model,self.config=model,config
            self.plans=original(model,config)._plans()
            self._components=tuple((None,None,None,None,SimpleNamespace(_results=())) for _ in self.plans)
        def _plans(self):
            return self.plans
        def fit(self,train,validation):
            assert train.role=='train' and validation.role=='validation'
            assert train.dataset_id==validation.dataset_id and train.split_id==validation.split_id
            assert not set(train.row_indices)&set(validation.row_indices)
            self.train,self.validation=train,validation
            calls.append(self)
            return self
        def estimate(self,test):
            assert test.role=='test' and test.dataset_id==self.train.dataset_id and test.split_id==self.train.split_id
            assert not set(test.row_indices)&(set(self.train.row_indices)|set(self.validation.row_indices))
            components=[]
            for index,(name,context,config) in enumerate(self.plans):
                value=(.125,.375)[index]
                args={f.name:None for f in fields(TermDiagnostic)}
                args.update(position=1,target='X',conditioning=context,contribution_nats=value,
                            method='neural_test_ce' if context else 'train_plugin',environment=MappingProxyType({'device':'cpu'}))
                if context:
                    from src.utils.reproducibility import derive_term_seed
                    tid=json.dumps(['njee-conditional-term-v1','X',list(context)],separators=(',',':'))
                    args.update(initialization_seed=derive_term_seed(config.training_seed,tid,purpose='initialization'),
                                shuffle_seed=derive_term_seed(config.training_seed,tid,purpose='shuffle'))
                term=TermDiagnostic(**args)
                if context:
                    report=ConditionalNeuralEntropyReport(value,('X',),context,(term,),'strict',test.dataset_id,test.split_id)
                else:
                    report=NeuralEntropyReport(value,('X',),(term,),'strict',test.dataset_id,test.split_id)
                components.append(InformationComponent(name,config,context,report,.01,.001))
            return NeuralInformationReport(-.25,'cmi' if self.model.family!='bsc' else 'mi',('X',),('Y',),
                                           ('Z',) if self.model.family!='bsc' else (),test.dataset_id,test.split_id,
                                           self.config,'fixture',MappingProxyType({}),tuple(components))
    monkeypatch.setattr(protocol,'make_estimator',lambda model,config:Frozen(model,config))
    # Deterministic table, not a scientific generated replica.
    monkeypatch.setattr(InformationModel,'generate',lambda self,n,seed:np.array(
        [[i%2,(i//2)%2] + ([i%2] if self.family!='bsc' else []) for i in range(n)],dtype=np.int64))
    monkeypatch.setattr(runner,'environment',lambda:{'fixture':True})
    return calls


@pytest.mark.parametrize('family',['bsc','chain','fork'])
def test_worker_mock_orchestration_signed_outputs_and_npz(tiny,tmp_path,monkeypatch,family):
    cell=deepcopy(tiny['cells'][0]);cell['model']={'family':family,'epsilon_x':0 if family=='bsc' else .15,'epsilon_y':.5}
    cell['truth']=InformationModel(**cell['model']).truths()
    r=cell['allocation'][0]
    _,r['components']=protocol.effective_streams(InformationModel(**cell['model']),r['training_seed'])
    calls=mock_neural(monkeypatch)
    result,inference,meta=runner.run_replica(cell,r,tmp_path)
    assert result['neural']['information_nats']==-.25
    assert inference['p_value']==1. and not inference['reject']
    assert set(result)==set(runner.VARIANTS) and len(calls)==1
    assert meta['status']=='complete' and meta['sample_counts']=={'full':10,'train':6,'validation':2,'test':2}
    with np.load(tmp_path/'dataset.npz',allow_pickle=False) as saved:assert saved['values'].shape[0]==10
    from src.utils.split import load_split
    assert load_split(tmp_path/'split.npz').n_samples==10
    report=json.loads((tmp_path/'neural_diagnostics.json').read_text())
    assert report['information_nats']==-.25 and len(report['components'])==2
    assert 'dataset.npz' in meta['artifact_sha256']


def test_execution_guards_and_dry_run_never_fit(tiny,tmp_path,monkeypatch,capsys):
    def forbidden(*a,**k):pytest.fail('Scientific worker called by safe interface')
    monkeypatch.setattr(runner,'run_replica',forbidden)
    monkeypatch.setattr(protocol,'load_definition',lambda:protocol.definition())
    runner.main(['--dry-run'])
    assert 'conditional_fits' in capsys.readouterr().out
    assert not Path(tiny['output_root']).exists()
    with pytest.raises(PermissionError):runner.execute(tmp_path/'missing')
    with pytest.raises(SystemExit):runner.main(['--dry-run','--authorize-execution'])
    if runner.os.name=='nt':
        with pytest.raises(PermissionError):runner.execute(tmp_path/'missing',authorized=True)


def execution_setup(tiny,tmp_path,monkeypatch):
    prep=tmp_path/'prep';protocol.materialize(prep)
    root=tmp_path/'output'
    monkeypatch.setattr(runner,'_git_provenance',lambda:{'dirty':False,'commit':'fixture'})
    monkeypatch.setattr(runner,'environment',lambda:{'fixture':True})
    return prep,root


def test_mock_campaign_failure_accounting_and_no_retry(tiny,tmp_path,monkeypatch):
    prep,root=execution_setup(tiny,tmp_path,monkeypatch)
    calls=[]
    def fail(cell,replica,directory,**kwargs):
        calls.append(replica['replica_id']);raise RuntimeError('controlled failure')
    monkeypatch.setattr(runner,'run_replica',fail)
    runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    assert calls==['replica-000000','replica-000001']
    cell=root/tiny['cells'][0]['cell_id']
    rows=json.loads((cell/'raw_results.json').read_text())
    assert len(rows)==10 and all(r['status']=='failed' for r in rows)
    summary=json.loads((cell/'summary.json').read_text())
    assert all(s['n_failed']==2 for s in summary['summaries'])
    assert json.loads((root/'manifest.json').read_text())['status']=='complete_with_failures'
    with pytest.raises(FileExistsError):runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    assert len(calls)==2


def test_mock_campaign_success_reports_and_read_only_inspection(tiny,tmp_path,monkeypatch):
    prep,root=execution_setup(tiny,tmp_path,monkeypatch)
    calls=mock_neural(monkeypatch)
    runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    assert len(calls)==2
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    report=runner.inspect_results(root)
    assert report['cells'][0]['summary']['false_positive_analysis']['n_tested']==2
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    for file in ('components.csv','term_diagnostics.csv','raw_results.csv','paired_comparisons.csv','summary.csv'):
        assert (root/tiny['cells'][0]['cell_id']/file).exists()


def test_interruption_and_pending_evidence(tiny,tmp_path,monkeypatch):
    prep,root=execution_setup(tiny,tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'run_replica',lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    rows=json.loads((root/tiny['cells'][0]['cell_id']/'raw_results.json').read_text())
    assert sum(r['status']=='interrupted' for r in rows)==5
    assert sum(r['status']=='not_attempted' for r in rows)==5
    assert json.loads((root/'manifest.json').read_text())['status']=='interrupted'


def test_inference_failure_retains_completed_estimates(tiny,tmp_path,monkeypatch):
    cell=tiny['cells'][0];r=cell['allocation'][0]
    mock_neural(monkeypatch)
    monkeypatch.setattr(runner,'heldout_permutation_test',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('test failed')))
    results,inference,meta=runner.run_replica(cell,r,tmp_path)
    assert results['neural']['information_nats']==-.25 and inference['status']=='failed'
    assert (tmp_path/'estimates.json').exists() and meta['status']=='complete'


def test_inference_interruption_preserves_estimates_and_stops_next_replica(tiny,tmp_path,monkeypatch):
    prep,root=execution_setup(tiny,tmp_path,monkeypatch)
    calls=mock_neural(monkeypatch)
    monkeypatch.setattr(runner,'heldout_permutation_test',lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    assert len(calls)==1
    rows=json.loads((root/tiny['cells'][0]['cell_id']/'raw_results.json').read_text())
    assert sum(r['status']=='complete' for r in rows)==5
    assert sum(r['status']=='not_attempted' for r in rows)==5
    assert rows[0]['test_status']=='interrupted' and rows[0]['entropy_nats']==-.25
    summary=json.loads((root/tiny['cells'][0]['cell_id']/'summary.json').read_text())
    assert summary['false_positive_analysis']['n_tested']==0
    assert summary['false_positive_analysis']['n_untested']==2


def test_raw_pair_identity_and_artifact_tampering_detected(tiny,tmp_path,monkeypatch):
    prep,root=execution_setup(tiny,tmp_path,monkeypatch)
    mock_neural(monkeypatch)
    runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    directory=root/tiny['cells'][0]['cell_id']
    path=directory/'raw_results.json';rows=json.loads(path.read_text())
    rows[1]['split_id']='foreign'
    with pytest.raises(ValueError):runner.validate_rows(tiny['cells'][0],rows)
    rows[1]['split_id']=rows[0]['split_id']
    rows[0]['signed_error_nats']=999
    from src.metrics.information import summarize_information
    with pytest.raises(ValueError):summarize_information(rows,['replica-000000','replica-000001'],bootstrap_draws=2)
    artifact=directory/'replica-000000'/'estimates.json';artifact.write_text('{}')
    with pytest.raises(ValueError,match='hash'):runner.inspect_results(root)


def test_cancellation_before_launch_and_clean_git_gate(tiny,tmp_path,monkeypatch):
    prep,root=execution_setup(tiny,tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'_git_provenance',lambda:{'dirty':True})
    with pytest.raises(ValueError):runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    assert not root.exists()
    monkeypatch.setattr(runner,'_git_provenance',lambda:{'dirty':False})
    cancel=tmp_path/'cancel';cancel.touch()
    with pytest.raises(KeyboardInterrupt):
        runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root,cancel_file=cancel)
    assert all(r['status']=='not_attempted' for r in json.loads((root/'execution_ledger.json').read_text()))


def test_export_interruption_cannot_publish_complete_manifest(tiny,tmp_path,monkeypatch):
    prep,root=execution_setup(tiny,tmp_path,monkeypatch)
    mock_neural(monkeypatch)
    monkeypatch.setattr(runner,'export_tables',lambda *a:(_ for _ in ()).throw(KeyboardInterrupt('export interrupted')))
    with pytest.raises(KeyboardInterrupt):
        runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='interrupted'
    assert manifest['reporting_failure']['type']=='KeyboardInterrupt'
