"""Execution-control tests use fake workers/clocks, never neural campaigns."""
import json
import os
from pathlib import Path
import sys
import threading

import pytest

from experiments import final_validation as control
from experiments.final_protocol import materialize, preview, ORDER, REPLICAS


class Clock:
    def __init__(self,steps=None):self.now=0.;self.steps=list(steps or [])
    def __call__(self):return self.now
    def sleep(self,seconds):self.now+=self.steps.pop(0) if self.steps else seconds


def raw(output,complete=False):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,rid in enumerate(REPLICAS):
        for _ in range(5):
            rows.append({'replica_id':rid,'status':'complete' if complete else 'running' if i==0 else 'not_attempted'})
    control._write(output/'raw_results.json',rows)


class Worker:
    def __init__(self,command,workdir,mode,clock):
        self.command=command;self.workdir=workdir;self.mode=mode;self.clock=clock
        self.cancel=Path(command[command.index('--cancel-file')+1])
        self.outcome=Path(command[command.index('--worker-status')+1])
        config=Path(command[command.index('--worker')+1])
        self.cid=config.name.removesuffix('-campaign.yaml')
        self.output=workdir/'science'/self.cid
        self.returncode=None;self.killed=False
        if mode!='complete':raw(self.output)

    def poll(self):
        if self.returncode is not None:return self.returncode
        if self.mode=='complete':
            raw(self.output,True)
            control._write(self.output/'campaign_manifest.json',{'status':'complete'})
            control._write(self.outcome,{'status':'complete'})
            self.returncode=0
        elif self.mode=='cooperative' and self.cancel.exists():
            control._write(self.outcome,{'status':'interrupted'})
            self.returncode=130
        elif self.mode=='external':self.returncode=-9
        elif self.mode=='exception':
            control._write(self.outcome,{'status':'failed','failure':{'type':'RuntimeError'}})
            self.returncode=1
        return self.returncode

    def kill(self):self.killed=True;self.returncode=-9
    def wait(self,timeout):return self.returncode


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    prep=tmp_path/'prep';materialize(prep,'science')
    monkeypatch.setattr(control,'_git',lambda:'reviewed-checkpoint')
    monkeypatch.setattr(control,'_environment',lambda:{'python':'fixture','cpu':'fixture'})
    return prep


def run(prepared,tmp_path,clock,mode='complete',**kwargs):
    workers=[]
    def launch(command,log,workdir):
        assert command[0]==sys.executable or kwargs.get('interpreter')==command[0]
        workers.append(Worker(command,workdir,mode,clock));return workers[-1]
    result=control.execute(prepared,workdir=tmp_path,authorized=True,allow_local_windows=True,
                           clock=clock,sleep=clock.sleep,launcher=launch,**kwargs)
    return result,workers


def test_fixed_order_full_accounting_and_no_overwrite(prepared,tmp_path):
    result,workers=run(prepared,tmp_path,Clock())
    assert result['status']=='complete' and len(workers)==27
    assert [r['cell'] for r in result['campaigns']]==list(ORDER)
    assert sum(r['replica_counts']['completed'] for r in result['campaigns'])==810
    assert all(r['termination'] is None for r in result['campaigns'])
    before=(tmp_path/'science/control/ledger.json').read_bytes()
    with pytest.raises(FileExistsError):run(prepared,tmp_path,Clock())
    assert (tmp_path/'science/control/ledger.json').read_bytes()==before


def test_no_authorization_no_execution(prepared,tmp_path):
    with pytest.raises(ValueError,match='authorization'):control.execute(prepared,workdir=tmp_path)
    assert not (tmp_path/'science').exists()


def test_windows_requires_explicit_local_exception(prepared,tmp_path):
    if os.name!='nt':pytest.skip('Native Windows authorization gate')
    with pytest.raises(ValueError,match='Windows'):control.execute(prepared,workdir=tmp_path,authorized=True)
    assert not (tmp_path/'science').exists()


def test_do_not_launch_without_shutdown_and_minimum_window(prepared,tmp_path,monkeypatch):
    clock=Clock()
    def git():clock.now=17900;return 'reviewed'
    monkeypatch.setattr(control,'_git',git)
    result,workers=run(prepared,tmp_path,clock)
    assert not workers and result['status']=='budget_stopped'
    assert sum(e['replica_counts']['unattempted'] for e in result['campaigns'])==810


def test_cooperative_deadline_interruption(prepared,tmp_path):
    result,workers=run(prepared,tmp_path,Clock([17940,1]),'cooperative')
    assert len(workers)==1 and not workers[0].killed
    assert result['status']=='budget_stopped'
    first=result['campaigns'][0]
    assert first['status']=='interrupted' and first['termination']=='cooperative'
    assert first['replica_counts']=={'completed':0,'failed':0,'interrupted':1,'unattempted':29}
    assert sum(e['replica_counts']['unattempted'] for e in result['campaigns'])==809


def test_force_nonresponsive_worker_at_total_deadline(prepared,tmp_path):
    result,workers=run(prepared,tmp_path,Clock([17940,60]),'unresponsive')
    assert len(workers)==1 and workers[0].killed
    assert result['elapsed_seconds']==18000
    assert result['campaigns'][0]['termination']=='forced_deadline'
    assert result['campaigns'][0]['status']=='interrupted'


@pytest.mark.parametrize('mode,termination',[('external','external_or_unreported_termination'),('exception','worker_exception')])
def test_failure_stops_without_replacements(prepared,tmp_path,mode,termination):
    result,workers=run(prepared,tmp_path,Clock(),mode)
    assert len(workers)==1 and result['status']=='paused_for_review'
    assert result['campaigns'][0]['termination']==termination
    assert all(e['status']=='not_attempted' for e in result['campaigns'][1:])


def test_controller_keyboard_interrupt_preserves_ledger(prepared,tmp_path):
    clock=Clock();workers=[];first=True
    def launch(command,log,workdir):
        workers.append(Worker(command,workdir,'cooperative',clock));return workers[-1]
    def sleep(seconds):
        nonlocal first
        if first:first=False;raise KeyboardInterrupt()
        clock.now+=seconds
    with pytest.raises(KeyboardInterrupt):
        control.execute(prepared,workdir=tmp_path,authorized=True,allow_local_windows=True,
                        clock=clock,sleep=sleep,launcher=launch)
    ledger=json.loads((tmp_path/'science/control/ledger.json').read_text())
    assert ledger['status']=='interrupted' and not workers[0].killed
    assert ledger['campaigns'][0]['termination']=='cooperative_controller_shutdown'


def test_configurable_control_paths(prepared,tmp_path):
    result,_=run(prepared,tmp_path,Clock(),interpreter=sys.executable,
                 log_dir='execution-logs/run1',ledger_path='ledgers/run1.json')
    assert result['status']=='complete'
    assert (tmp_path/'ledgers/run1.json').exists() and (tmp_path/'execution-logs/run1').is_dir()


def test_control_paths_cannot_invade_scientific_directories(prepared,tmp_path):
    cid=preview('science')['cells'][0]['campaign_id']
    with pytest.raises(ValueError,match='control paths'):
        run(prepared,tmp_path,Clock(),log_dir=f'science/{cid}/logs')
    assert not (tmp_path/'science').exists()


def test_dry_run_default_never_launches_or_writes(tmp_path,monkeypatch,capsys):
    monkeypatch.chdir(tmp_path)
    def forbidden(*a,**k):pytest.fail('Execution during dry run')
    monkeypatch.setattr(control,'execute',forbidden)
    monkeypatch.setattr(control,'worker',forbidden)
    assert control.main([])==0
    summary=json.loads(capsys.readouterr().out)
    assert summary['mode']=='dry-run' and summary['expected']['replicas']==810
    assert list(tmp_path.iterdir())==[]


def test_worker_interruption_and_environment_with_mock_campaign(tmp_path,monkeypatch):
    from experiments import neural_campaign
    monkeypatch.setattr(control,'_environment',lambda:{'cpu':'mock','torch_num_threads':4})
    def interrupted(*a,**k):raise KeyboardInterrupt()
    monkeypatch.setattr(neural_campaign,'run_campaign',interrupted)
    path=tmp_path/'worker.json'
    assert control.worker(tmp_path/'unused.yaml',tmp_path/'cancel',path)==130
    result=json.loads(path.read_text())
    assert result['status']=='interrupted' and result['environment']['torch_num_threads']==4
    with pytest.raises(FileExistsError):control.worker(tmp_path/'unused.yaml',tmp_path/'cancel',path)


def test_spawn_uses_portable_environment_and_cwd(tmp_path,monkeypatch):
    captured={}
    def popen(command,**kwargs):captured.update(command=command,**kwargs);return 'process'
    monkeypatch.setattr(control.subprocess,'Popen',popen)
    assert control._spawn([sys.executable,'-m','module'],None,tmp_path)=='process'
    assert captured['cwd']==tmp_path
    assert captured['env']['PYTHONPATH'].split(os.pathsep)[0]==str(control.PROJECT_ROOT)


def test_cancel_file_monitor_is_platform_independent(tmp_path,monkeypatch):
    from experiments import neural_campaign
    interrupted=threading.Event()
    monkeypatch.setattr(control,'_environment',lambda:{'cpu':'fixture'})
    monkeypatch.setattr(control._thread,'interrupt_main',interrupted.set)
    cancel=tmp_path/'cancel'
    def fake_campaign(*a,**k):
        cancel.touch()
        assert interrupted.wait(5), 'Cancellation monitor never requested main-thread interruption'
        raise KeyboardInterrupt()
    monkeypatch.setattr(neural_campaign,'run_campaign',fake_campaign)
    assert control.worker(tmp_path/'unused.yaml',cancel,tmp_path/'worker.json')==130
    assert interrupted.is_set()


def test_private_worker_cli_does_not_bypass_authorization(tmp_path):
    with pytest.raises(SystemExit):control.main(['--worker',str(tmp_path/'unused.yaml')])


def test_control_file_atomic_replace_retries_transient_windows_lock(tmp_path,monkeypatch):
    path=tmp_path/'ledger.json'
    path.write_text('{"old":true}')
    original=Path.replace;attempts=[];sleeps=[]
    def replace(source,target):
        attempts.append(1)
        if len(attempts)<3:
            assert json.loads(path.read_text())=={'old':True}
            raise PermissionError('Temporary sharing violation')
        return original(source,target)
    monkeypatch.setattr(Path,'replace',replace)
    monkeypatch.setattr(control.time,'sleep',sleeps.append)
    control._write(path,{'new':True})
    assert json.loads(path.read_text())=={'new':True}
    assert len(attempts)==3 and sleeps==[.01,.02]


def test_control_file_persistent_lock_fails_without_destroying_old_ledger(tmp_path,monkeypatch):
    path=tmp_path/'ledger.json';path.write_text('{"old":true}')
    def locked(*a,**k):raise PermissionError('Persistent lock')
    sleeps=[]
    monkeypatch.setattr(Path,'replace',locked)
    monkeypatch.setattr(control.time,'sleep',sleeps.append)
    with pytest.raises(PermissionError):control._write(path,{'new':True})
    assert json.loads(path.read_text())=={'old':True} and len(sleeps)==5


def test_hard_watchdog_does_not_depend_on_controller_io_or_poll_loop(monkeypatch):
    callbacks=[]
    class Timer:
        def __init__(self,seconds,callback):assert seconds==60;callbacks.append(callback)
        def start(self):pass
    class Process:
        killed=False
        def poll(self):return None
        def kill(self):self.killed=True
    monkeypatch.setattr(control.threading,'Timer',Timer)
    process=Process()
    _,forced,errors=control._watchdog(process,60)
    assert not forced.is_set()
    callbacks[0]() # Main controller need not execute another polling step.
    assert process.killed and forced.is_set() and not errors


def test_recheck_deadline_after_prelaunch_ledger_io(prepared,tmp_path,monkeypatch):
    clock=Clock();original=control._write
    def write(path,value):
        original(path,value)
        if path.name=='ledger.json' and any(e['status']=='running' for e in value['campaigns']):
            clock.now=17900 # Simulate slow prelaunch persistence.
    monkeypatch.setattr(control,'_write',write)
    result,workers=run(prepared,tmp_path,clock)
    assert not workers and result['status']=='budget_stopped'
    assert result['campaigns'][0]['status']=='not_attempted'
    assert sum(e['replica_counts']['unattempted'] for e in result['campaigns'])==810


def test_already_cancelled_worker_never_starts_campaign(tmp_path,monkeypatch):
    from experiments import neural_campaign
    monkeypatch.setattr(control,'_environment',lambda:{'cpu':'fixture'})
    monkeypatch.setattr(neural_campaign,'run_campaign',lambda *a,**k:pytest.fail('Work after cancellation'))
    cancel=tmp_path/'cancel';cancel.touch()
    assert control.worker(tmp_path/'unused.yaml',cancel,tmp_path/'worker.json')==130
