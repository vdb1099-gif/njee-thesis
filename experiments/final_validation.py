import argparse
import _thread
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

from experiments.final_protocol import (BUDGET, DEFAULT_OUTPUT, MIN_CAMPAIGN_WINDOW,
                                        ORDER, REPLICAS, SHUTDOWN_RESERVE,
                                        materialize, preview, relative_path, validate_bundle)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    """Replace only an owned execution-control file; retain last complete JSON."""
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    # Windows scanners/readers can briefly deny atomic replacement. Retry only
    # this owned control file, with bounded delay; never replace scientific data.
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.01 * 2**attempt)


def _environment():
    import numpy as np
    import torch
    return {'python': platform.python_version(), 'interpreter': sys.executable,
            'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(),
            'cpu': platform.processor() or 'unavailable', 'machine': platform.machine(), 'cpu_count': os.cpu_count(),
            'torch_num_threads': torch.get_num_threads(), 'torch_num_interop_threads': torch.get_num_interop_threads(),
            'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            'device': 'cpu', 'reproducibility_mode': 'strict',
            'limitation': 'Strict mode does not guarantee bitwise identity across hardware, OS or versions.'}


def _git():
    commit = subprocess.run(['git','rev-parse','HEAD'], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(['git','status','--porcelain=v1','--untracked-files=all'], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    if status:
        raise ValueError(f'Execution requires a clean reviewed checkout: {status}')
    return commit


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _account(entry, output, termination=None):
    """Never rewrite worker results; infer interrupted attempts from durable state."""
    by_replica = {rid: [] for rid in REPLICAS}
    raw = output/'raw_results.json'
    if raw.exists():
        for row in _read(raw):
            if row['replica_id'] not in by_replica:
                raise ValueError('Unexpected replica identity in campaign evidence.')
            by_replica[row['replica_id']].append(row['status'])
    counts = {'completed': 0, 'failed': 0, 'interrupted': 0, 'unattempted': 0}
    states = {}
    for rid, statuses in by_replica.items():
        run_exists = (output/'replicas'/rid).exists()
        if len(statuses) == 5 and all(s == 'complete' for s in statuses):
            state = 'completed'
        elif 'failed' in statuses:
            state = 'failed'
        elif ('interrupted' in statuses or 'running' in statuses or run_exists):
            state = 'interrupted'
        else:
            state = 'unattempted'
        counts[state] += 1
        states[rid] = state
    entry.update(replica_counts=counts, replica_states=states)
    if termination is not None:
        entry['termination'] = termination
    return entry


def _spawn(command, log, workdir):
    # No console-control dependency: Windows and Linux both use cancel-file IPC.
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
    return subprocess.Popen(command, cwd=workdir, stdout=log, stderr=subprocess.STDOUT, env=environment)


def _watchdog(process, seconds):
    """Kill independently of controller polling or ledger I/O at the hard limit."""
    forced = threading.Event()
    errors = []
    def stop():
        if process.poll() is None:
            forced.set()
            try:
                process.kill()
            except OSError as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
    timer = threading.Timer(max(0, seconds), stop)
    timer.daemon = True
    timer.start()
    return timer, forced, errors


def execute(preparation_dir, *, workdir=PROJECT_ROOT, interpreter=None, log_dir=None, ledger_path=None,
            authorized=False, allow_local_windows=False, clock=time.monotonic,
            sleep=time.sleep, launcher=_spawn):
    """One run, one global deadline, no retries/resume; clock/launcher injectable for tests.

    Stop launching with <=120s left. Request cooperative cancellation at 17940s;
    at 18000s kill a nonresponsive worker. Persistence/reaping can have OS latency:
    the hard limit governs scientific worker lifetime, not guaranteed disk flush.
    """
    if not authorized:
        raise ValueError('Final evaluation requires separate explicit execution authorization.')
    if os.name == 'nt' and not allow_local_windows:
        raise ValueError('Prefer Linux server; local Windows needs explicit exception authorization.')
    start = clock()
    stop, hard = start+BUDGET-SHUTDOWN_RESERVE, start+BUDGET
    plan = validate_bundle(preparation_dir)
    commit = _git()
    workdir = Path(workdir).resolve()
    preparation_dir = Path(preparation_dir).resolve()
    output = workdir/plan['output_root']
    logs = workdir/relative_path(log_dir or f'{plan["output_root"]}/control/logs')
    ledger_file = workdir/relative_path(ledger_path or f'{plan["output_root"]}/control/ledger.json')
    # Check every owned destination before any mutation. Reserve output exclusively.
    if output.exists() or logs.exists() or ledger_file.exists():
        raise FileExistsError('Scientific output, log directory or ledger already exists.')
    for cell in plan['cells']:
        campaign_dir = output/cell['campaign_id']
        if any(p == campaign_dir or campaign_dir in p.parents for p in (logs, ledger_file)):
            raise ValueError('Execution-control paths must stay outside scientific campaign directories.')
    output.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    ledger_file.parent.mkdir(parents=True, exist_ok=True)
    ledger = {'schema':'njee-final-execution-v1','status':'running','started_at_utc':_now(),
              'controller_pid':os.getpid(),
              'git_commit':commit,'protocol_hash':plan['protocol_hash'],'environment':_environment(),
              'budget_seconds':BUDGET,'shutdown_reserve_seconds':SHUTDOWN_RESERVE,
              'minimum_campaign_window_seconds':MIN_CAMPAIGN_WINDOW,
              'output_root':plan['output_root'],'preparation_dir':str(preparation_dir),
              'interpreter':str(interpreter or sys.executable), 'campaigns':[]}
    by_cell = {e['cell']:e for e in plan['cells']}
    for key in ORDER:
        entry={'cell':key,'campaign_id':by_cell[key]['campaign_id'],'status':'not_attempted','termination':None}
        _account(entry, output/entry['campaign_id'])
        ledger['campaigns'].append(entry)
    _write(ledger_file, ledger)
    active = None
    process = None
    cancel = None
    watchdog = None
    forced = None
    try:
        for entry in ledger['campaigns']:
            if stop-clock() <= MIN_CAMPAIGN_WINDOW:
                ledger['status']='budget_stopped'
                break
            active=entry
            cid=entry['campaign_id']
            cancel=logs/f'{cid}.cancel'
            outcome=logs/f'{cid}.worker.json'
            command=[str(interpreter or sys.executable),'-m','experiments.final_validation','--worker',
                     str(preparation_dir/f'{cid}-campaign.yaml'),'--cancel-file',str(cancel),
                     '--worker-status',str(outcome),'--authorize-execution']
            entry.update(status='running',started_at_utc=_now(),command=command)
            _write(ledger_file,ledger)
            with (logs/f'{cid}.log').open('x',encoding='utf-8') as log:
                if stop-clock() <= MIN_CAMPAIGN_WINDOW:
                    entry.update(status='not_attempted', reason='insufficient_prelaunch_time')
                    entry.pop('started_at_utc', None)
                    ledger['status']='budget_stopped'
                    active=None
                    break
                process=launcher(command,log,workdir)
                watchdog,forced,kill_errors=_watchdog(process,hard-clock())
                entry['worker_pid']=getattr(process,'pid',None)
                requested=False
                while process.poll() is None:
                    _account(entry,output/cid)
                    _write(ledger_file,ledger)
                    if clock() >= stop and not requested:
                        cancel.touch(exist_ok=False)
                        entry['termination']='cooperative_requested'
                        requested=True
                    if clock() >= hard:
                        # Persist intention before the OS operation; stale campaign files stay untouched.
                        entry.update(status='interrupted',termination='forced_deadline')
                        _write(ledger_file,ledger)
                        forced.set()
                        process.kill()
                        process.wait(timeout=5)
                        break
                    sleep(.25)
                entry['returncode']=process.poll()
                watchdog.cancel()
                if forced.is_set():
                    entry.update(status='interrupted',termination='forced_deadline',kill_errors=kill_errors)
            worker=_read(outcome) if outcome.exists() else None
            manifest=output/cid/'campaign_manifest.json'
            status=_read(manifest).get('status') if manifest.exists() else None
            if entry['termination']=='forced_deadline':
                entry['status']='interrupted'
            elif worker and worker['status']=='interrupted':
                entry.update(status='interrupted',termination='cooperative')
            elif worker and worker['status']=='complete' and status=='complete' and entry['returncode']==0:
                entry['status']='complete'
            elif worker and worker['status']=='complete' and status=='complete_with_failures':
                entry['status']='failed'
            elif worker and worker['status']=='failed':
                entry.update(status='failed',termination='worker_exception')
            else:
                entry.update(status='interrupted',termination='external_or_unreported_termination')
            entry['worker_outcome']=worker
            entry['finished_at_utc']=_now()
            _account(entry,output/cid)
            _write(ledger_file,ledger)
            process=None
            active=None
            if requested or entry['status'] != 'complete':
                ledger['status']='budget_stopped' if requested else 'paused_for_review'
                break
        else:
            ledger['status']='complete'
    except (Exception,KeyboardInterrupt) as exc:
        ledger.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                      failure={'type':type(exc).__name__,'message':str(exc)})
        if process is not None and process.poll() is None:
            if not cancel.exists():cancel.touch()
            until=min(hard,clock()+SHUTDOWN_RESERVE)
            while process.poll() is None and clock()<until:sleep(.25)
            if process.poll() is None:
                if active:active['termination']='forced_controller_shutdown'
                _write(ledger_file,ledger)
                process.kill()
                process.wait(timeout=5)
            elif active:
                active['termination']='forced_deadline' if forced is not None and forced.is_set() else 'cooperative_controller_shutdown'
        if active:
            active['status']='interrupted' if process is not None or isinstance(exc,KeyboardInterrupt) else 'failed'
            _account(active,output/active['campaign_id'])
        raise
    finally:
        if watchdog is not None:watchdog.cancel()
        ledger.update(finished_at_utc=_now(),elapsed_seconds=clock()-start)
        _write(ledger_file,ledger)
    return ledger


def worker(config_path,cancel_file,status_file):
    """Run the existing campaign; monitor cancellation on both supported OSes."""
    from experiments.neural_campaign import run_campaign
    done=threading.Event()
    def monitor():
        while not done.wait(.1):
            if cancel_file.exists():
                _thread.interrupt_main()
                return
    if status_file.exists():raise FileExistsError(status_file)
    thread=threading.Thread(target=monitor,daemon=True)
    status={'status':'running','started_at_utc':_now(),'environment':_environment()}
    _write(status_file,status)
    code=0
    try:
        if cancel_file.exists():
            raise KeyboardInterrupt()
        thread.start()
        run_campaign(config_path,invocation=sys.argv)
        status['status']='complete'
    except KeyboardInterrupt:
        status['status']='interrupted';code=130
    except Exception as exc:
        status.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc)});code=1
    finally:
        done.set()
        if thread.ident is not None:
            thread.join(timeout=1)
        status['finished_at_utc']=_now()
        _write(status_file,status)
    return code


def main(argv=None):
    parser=argparse.ArgumentParser(description='Preview, prepare, execute or report frozen NJEE validation; execution requires authorization.')
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--dry-run',action='store_true')
    action.add_argument('--prepare',type=Path)
    action.add_argument('--execute',type=Path)
    action.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    action.add_argument('--report',type=Path,help='Read-only confirmatory report from existing final artifacts')
    parser.add_argument('--output-root',default=DEFAULT_OUTPUT)
    parser.add_argument('--workdir',type=Path,default=PROJECT_ROOT)
    parser.add_argument('--interpreter',help='Default: active sys.executable')
    parser.add_argument('--log-dir')
    parser.add_argument('--ledger-path')
    parser.add_argument('--authorize-execution',action='store_true')
    parser.add_argument('--allow-local-windows',action='store_true')
    parser.add_argument('--cancel-file',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--worker-status',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        if not args.authorize_execution:parser.error('Worker requires explicit execution authorization.')
        if args.cancel_file is None or args.worker_status is None:parser.error('Worker control paths required.')
        return worker(args.worker,args.cancel_file,args.worker_status)
    if args.report:
        from src.metrics.confirmatory import paired_intervals
        records=[]
        for e in preview()['cells']:
            if e['cell'] in ('F01','F02','F17','F18'):
                path=args.report/e['campaign_id']/'raw_results.json'
                if path.exists():records.extend(_read(path))
        print(json.dumps(paired_intervals(records),indent=2,allow_nan=False))
    elif args.execute:
        result=execute(args.execute,workdir=args.workdir,interpreter=args.interpreter,
                       log_dir=args.log_dir,ledger_path=args.ledger_path,
                       authorized=args.authorize_execution,allow_local_windows=args.allow_local_windows)
        print(json.dumps(result,indent=2))
    else:
        plan=materialize(args.prepare,args.output_root) if args.prepare else preview(args.output_root)
        print(json.dumps({'mode':'preparation' if args.prepare else 'dry-run','protocol_hash':plan['protocol_hash'],
                          'expected':plan['expected'],'execution_order':plan['execution_order'],
                          'historical_seed_check':plan['historical_seed_check'],
                          'cells':[{k:e[k] for k in ('cell','campaign_id','master_seed','data','baseline_hash','campaign_hash')}
                                   for e in plan['cells']]},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
