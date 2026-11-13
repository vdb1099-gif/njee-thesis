"""Protocol/control fixtures with mocked generation and fits; no scientific run."""
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
from types import MappingProxyType,SimpleNamespace

import numpy as np
import pytest

from experiments import transfer_protocol as protocol
from experiments import transfer_validation as runner
from src.estimators.neural import TermDiagnostic
from src.estimators.conditional_neural import ConditionalNeuralEntropyReport
from src.estimators.neural_information import NeuralConditionalMutualInformationEstimator,InformationComponent,NeuralInformationReport
from src.estimators.transfer_entropy import NeuralTransferEntropyReport
from src.utils.temporal import build_temporal_embedding,make_chronological_split


@pytest.fixture(scope='module')
def full_plan():return protocol.preview()


def test_full_matrix_counts_and_effective_namespaces(full_plan):
    p=full_plan
    assert p['expected']=={'cells':51,'process_replicas':2550,'te_fits':30600,'conditional_fits':61200,
                           'primary_tests':20400,'surrogate_te_statistics':20379600}
    assert len(p['execution_order'])==len(set(p['execution_order']))==51
    assert {c['n'] for c in p['cells']}=={1000,5000,10000}
    assert {c['master_seed'] for c in p['cells']}==set(range(180000,180051))
    assert sum(c['coupling']==0 for c in p['cells'])==3
    assert {(c['coupling'],c['delay']) for c in p['cells'] if c['coupling']}=={(c,lag) for c in (.1,.3,.5,.8) for lag in (1,2,3,5)}
    seeds=[]
    for cell in p['cells']:
        assert len(cell['allocation'])==50 and len(cell['jobs'])==12
        seeds.append(cell['bootstrap']['seed'])
        for replica in cell['allocation']:
            seeds.append(replica['data']['seed']);seeds.extend(v['seed'] for v in replica['surrogates'].values())
            for value in replica['jobs'].values():
                seeds.append(value['seed'])
                for c in value['components']:seeds.extend((c['training_seed'],c['initialization'],c['shuffle']))
    assert len(set(seeds))==len(seeds) and not set(seeds)&protocol.prior_seeds()
    assert protocol.load_definition()==protocol.definition()
    assert not Path(p['output_root']).exists()


@pytest.fixture
def tiny(monkeypatch):
    d=deepcopy(protocol.definition())
    d.update(couplings=[],delays=[1],lengths=[60],replicas=2,candidates=[1,2],family_size=4,
             surrogate_draws=3,bootstrap_draws=3,burn_in=10)
    monkeypatch.setattr(protocol,'definition',lambda:deepcopy(d))
    return protocol.preview()


def test_preparation_hashes_portability_and_no_overwrite(tiny,tmp_path):
    assert tiny==protocol.preview()
    path=tmp_path/'prep';protocol.materialize(path)
    assert protocol.validate_bundle(path)==tiny
    assert len(list(path.iterdir()))==2
    with pytest.raises(FileExistsError):protocol.materialize(path)
    manifest=path/'protocol.json';data=json.loads(manifest.read_text());data['protocol_hash']='bad';manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):protocol.validate_bundle(path)
    for bad in ('/home/private','C:/private','../escape','bad\\path'):
        with pytest.raises(ValueError):protocol.preview(bad)


def install_mock_fits(monkeypatch):
    calls=[]
    class TE:
        def __init__(self,config,*,embedding,source_id,target_id,source_alphabet,target_alphabet,gap,purge):
            self.config,self.spec,self.source_id,self.target_id,self.gap,self.purge=config,embedding,source_id,target_id,gap,purge
        def fit(self,source,target,*,dataset_id,source_times,target_times):
            embedded=build_temporal_embedding(source,target,spec=self.spec,dataset_id=dataset_id,
                                               source_id=self.source_id,target_id=self.target_id,
                                               source_times=source_times,target_times=target_times)
            self._parts=make_chronological_split(embedded,gap=self.gap,purge=self.purge)
            order=self.config.component_order;ys=self.spec.past_order(self.source_id,source=True);zs=self.spec.past_order(self.target_id,source=False)
            alphabets={v:np.array([0,1]) for v in order+ys+zs}
            plans=NeuralConditionalMutualInformationEstimator(self.config,y_order=ys,z_order=zs,alphabets=alphabets)._plans()
            components=[]
            from src.utils.reproducibility import derive_term_seed
            for i,(name,context,config) in enumerate(plans):
                value=(.125,.375)[i]
                tid=json.dumps(['njee-conditional-term-v1',order[0],list(context)],separators=(',',':'))
                args={f.name:None for f in fields(TermDiagnostic)}
                args.update(target=order[0],conditioning=context,contribution_nats=value,
                            initialization_seed=derive_term_seed(config.training_seed,tid,purpose='initialization'),
                            shuffle_seed=derive_term_seed(config.training_seed,tid,purpose='shuffle'),
                            best_validation_ce_nats=value,method='neural_test_ce',position=1,
                            environment=MappingProxyType({'fixture':True}))
                entropy=ConditionalNeuralEntropyReport(value,order,context,(TermDiagnostic(**args),),
                                                       'strict',self._parts.train.dataset_id,self._parts.train.split_id)
                components.append(InformationComponent(name,config,context,entropy,0.,0.))
            self.cmi=NeuralInformationReport(-.25,'cmi',order,ys,zs,self._parts.train.dataset_id,self._parts.train.split_id,
                                            self.config,'fixture',MappingProxyType({}),tuple(components))
            self._model=SimpleNamespace(_components=[(None,None,None,None,SimpleNamespace(_results=()))]*2,
                                        estimate=lambda batch:SimpleNamespace(information_nats=-.25))
            calls.append(self)
            return self
        def estimate(self):
            return NeuralTransferEntropyReport(-.25,(self.source_id,self.target_id),self.spec,
                                              self._parts.test.variable_ids,self._parts.provenance,self._parts.dropped_embedding_rows,
                                              60,self._parts.algorithm,(.6,.2,.2),self.gap,self.purge,self.cmi)
    monkeypatch.setattr(runner,'NeuralTransferEntropyEstimator',TE)
    from src.simulation.coupled_binary import CoupledBinaryModel
    monkeypatch.setattr(CoupledBinaryModel,'generate',lambda self,n,**kw:(np.arange(n)%2,1-np.arange(n)%2,{'fixture':True}))
    monkeypatch.setattr(runner,'environment',lambda:{'fixture':True})
    return calls


def refs(cell):
    return {j['job_id']:{'transfer_entropy_nats':0.,'status':'fixture-exact-zero',
                          'entropy_truncation_error_bound_nats':0.} for j in cell['jobs']}


def test_alignment_all_jobs_directions_and_mock_worker(tiny,tmp_path,monkeypatch):
    calls=install_mock_fits(monkeypatch);cell=tiny['cells'][0];replica=cell['allocation'][0]
    rows,manifest=runner.run_process_replica(cell,replica,tmp_path,refs(cell))
    assert len(calls)==len(cell['jobs'])==8
    supports=[tuple(p.target_timestamps for p in c._parts.provenance) for c in calls]
    assert all(s==supports[0] for s in supports)
    for c in calls:
        for left,right in zip(c._parts.provenance,c._parts.provenance[1:]):
            assert min(min(row) for row in right.observation_timestamps)>left.target_timestamps[-1]
    assert all(r['te_nats']==-.25 and r['status']=='complete' for r in rows)
    assert all(r['p_value']==1 for r in rows if r['embedding']=='canonical')
    assert manifest['status']=='complete' and (tmp_path/'process.npz').exists()
    assert (tmp_path/'X_to_Y-surrogates.json').exists()


def setup(tiny,tmp_path,monkeypatch):
    prep=tmp_path/'prep';protocol.materialize(prep)
    monkeypatch.setattr(runner,'_git_provenance',lambda:{'dirty':False,'commit':'fixture'})
    monkeypatch.setattr(runner,'environment',lambda:{'fixture':True})
    monkeypatch.setattr(runner,'reference_te',lambda *a,**k:{'transfer_entropy_nats':0.,'status':'fixture-exact-zero','entropy_truncation_error_bound_nats':0.})
    return prep,tmp_path/'output'


def test_controller_mock_execution_outputs_and_read_only_reporting(tiny,tmp_path,monkeypatch):
    prep,root=setup(tiny,tmp_path,monkeypatch);calls=install_mock_fits(monkeypatch)
    runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    assert len(calls)==16
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result=runner.inspect_results(root)
    assert result['manifest']['status']=='complete'
    assert result['cells'][0]['summary']['accuracy']['false_link']['n_evaluated']==2
    assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert all((root/name).exists() for name in ('replica_results.csv','summary.csv','replica_metrics.csv','direction_lag_accuracy.csv'))
    with pytest.raises(FileExistsError):runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    assert len(calls)==16
    artifact=next(root.rglob('diagnostics.json'));artifact.write_text('{}')
    with pytest.raises(ValueError,match='hash'):runner.inspect_results(root)


def test_failed_fits_no_replacement_and_complete_ledger(tiny,tmp_path,monkeypatch):
    prep,root=setup(tiny,tmp_path,monkeypatch);install_mock_fits(monkeypatch)
    attempted=[]
    def fail(*a,**k):attempted.append(k['job']['job_id']);raise RuntimeError('fit failed')
    monkeypatch.setattr(runner,'_aligned_fit',fail)
    runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    assert len(attempted)==16
    summary=json.loads((root/tiny['cells'][0]['cell_id']/'summary.json').read_text())
    assert all(s['n_failed']==2 for s in summary['summaries'])
    assert summary['accuracy']['false_link']['n_evaluated']==0
    assert summary['process_accounting']['statuses']['complete_with_failures']==2


def test_interruption_preserves_estimates_and_pending_replicas(tiny,tmp_path,monkeypatch):
    prep,root=setup(tiny,tmp_path,monkeypatch);install_mock_fits(monkeypatch)
    monkeypatch.setattr(runner,'frozen_path_test',lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt('stop inference')))
    with pytest.raises(KeyboardInterrupt):runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    cell=tiny['cells'][0];rows=runner.read_cell(root/cell['cell_id'],cell)
    assert sum(r['status']=='complete' for r in rows)==8
    assert sum(r['status']=='not_attempted' for r in rows)==8
    assert json.loads((root/'manifest.json').read_text())['status']=='interrupted'


def test_guards_dry_run_and_cancel_before_training(tiny,tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(protocol,'load_definition',lambda:protocol.definition())
    monkeypatch.setattr(runner,'run_process_replica',lambda *a,**k:pytest.fail('Unauthorized worker invocation'))
    runner.main(['--dry-run'])
    assert 'conditional_fits' in capsys.readouterr().out
    with pytest.raises(PermissionError):runner.execute('missing')
    with pytest.raises(SystemExit):runner.main(['--dry-run','--authorize-execution'])
    prep,root=setup(tiny,tmp_path,monkeypatch)
    cancel=tmp_path/'cancel';cancel.touch()
    with pytest.raises(KeyboardInterrupt):
        runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root,cancel_file=cancel)
    assert not list(root.glob('T*'))
    assert all(e['status']=='not_attempted' for e in json.loads((root/'execution_ledger.json').read_text()))


def test_dirty_checkout_blocked(tiny,tmp_path,monkeypatch):
    prep,root=setup(tiny,tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'_git_provenance',lambda:{'dirty':True})
    with pytest.raises(ValueError):runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    assert not root.exists()


def test_generation_failure_records_all_jobs_without_retry(tiny,tmp_path,monkeypatch):
    prep,root=setup(tiny,tmp_path,monkeypatch);install_mock_fits(monkeypatch)
    from src.simulation.coupled_binary import CoupledBinaryModel
    calls=[]
    def fail(*a,**k):calls.append(k['seed']);raise RuntimeError('generation failed')
    monkeypatch.setattr(CoupledBinaryModel,'generate',fail)
    runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    assert len(calls)==len(set(calls))==2
    summary=json.loads((root/tiny['cells'][0]['cell_id']/'summary.json').read_text())
    assert summary['process_accounting']['statuses']['failed']==2
    assert all(s['n_failed']==2 for s in summary['summaries'])


def test_inference_failure_and_export_interruption_are_not_complete(tiny,tmp_path,monkeypatch):
    prep,root=setup(tiny,tmp_path,monkeypatch);install_mock_fits(monkeypatch)
    monkeypatch.setattr(runner,'frozen_path_test',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('inference failed')))
    runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=root)
    result=runner.inspect_results(root)
    assert result['manifest']['status']=='complete_with_failures'
    assert result['cells'][0]['summary']['accuracy']['false_link']['n_evaluated']==0
    assert all(s['n_complete']==2 for s in result['cells'][0]['summary']['summaries'])
    monkeypatch.setattr(runner,'export_tables',lambda *a:(_ for _ in ()).throw(KeyboardInterrupt('export stopped')))
    another=tmp_path/'second-output'
    with pytest.raises(KeyboardInterrupt):runner.execute(prep,authorized=True,allow_local_windows=True,output_dir=another)
    assert json.loads((another/'manifest.json').read_text())['status']=='interrupted'


def test_deterministic_nonce_probing_and_configuration_tamper(tiny,tmp_path,monkeypatch):
    first=tiny['cells'][0]['bootstrap']['seed']
    monkeypatch.setattr(protocol,'prior_seeds',lambda:frozenset({first}))
    resolved=protocol.preview()
    assert resolved['cells'][0]['bootstrap']['seed']!=first
    assert resolved['cells'][0]['bootstrap']['nonce']>=1
    import yaml
    config=protocol.definition();config['neural']['max_epochs']=1
    path=tmp_path/'changed.yaml';path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):protocol.load_definition(path)
