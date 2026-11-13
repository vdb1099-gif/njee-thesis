import argparse
from dataclasses import fields,is_dataclass
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from experiments import transfer_protocol as protocol
from experiments.information_validation import (write_json as base_write_json,write_csv as base_write_csv,
                                                environment,_git_provenance)
from src.estimators.transfer_entropy import NeuralTransferEntropyEstimator
from src.metrics.transfer import frozen_path_test,summarize_transfer
from src.simulation.coupled_binary import CoupledBinaryModel,reference_te,coupling_bound


def portable(value):
    if is_dataclass(value):return {f.name:portable(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,Mapping):return {k:portable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [portable(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def write_json(path,value):base_write_json(path,portable(value))
def write_csv(path,value):base_write_csv(path,portable(value))


def process_model(cell):
    return CoupledBinaryModel(cell['coupling'],cell['delay'],cell['noise_x'],cell['noise_y'])


def empty_rows(cell,replica):
    return [{**job,'cell_id':cell['cell_id'],'replica_id':replica['replica_id'],
             'configuration_hash':cell['configuration_hash'],'data_seed':replica['data']['seed'],
             'training_seed':replica['jobs'][job['job_id']]['seed'],'status':'not_attempted',
             'te_nats':None,'reference_nats':None,'signed_error_nats':None,
             'validation_score_nats':None,'runtime_seconds':None,'failure':None,
             'test_status':'not_attempted' if job['embedding']=='canonical' else 'not_applicable',
             'p_value':None,'test_timestamps':None} for job in cell['jobs']]


def _aligned_fit(x,y,*,job,seed,identity):
    """Use the UNCHANGED TE public fit API with deterministic prefix offsets.

    Dropping raw prefix M-max(lags) yields N-M embedded targets for every job.
    Common gap M dominates every job's purge span. Global timestamps preserve
    original positions despite different raw origins. Thus retained TEST times
    match exactly across lag, embedding and direction comparisons.
    """
    spec=protocol.spec_for(job);d=protocol.definition()
    offset=d['common_span']-max(spec.source_lags+spec.target_lags)
    if offset<0 or d['gap']<d['common_span'] or not d['purge']:
        raise ValueError('Frozen alignment span/gap/purge contract violated.')
    source,target=(x,y) if job['direction']=='X_to_Y' else (y,x)
    source_id,target_id=('X','Y') if job['direction']=='X_to_Y' else ('Y','X')
    estimator=NeuralTransferEntropyEstimator(protocol.config_for(job,seed),embedding=spec,
                                            source_id=source_id,target_id=target_id,
                                            source_alphabet=np.array([0,1]),target_alphabet=np.array([0,1]),
                                            gap=d['gap'],purge=True)
    timestamps=np.arange(offset,len(x),dtype=np.int64)
    estimator.fit(source[offset:],target[offset:],dataset_id=identity,
                  source_times=timestamps,target_times=timestamps)
    return estimator,offset


def _audit_seeds(report,expected):
    actual=[]
    for component in report.cmi_report.components:
        if len(component.entropy.terms)!=1:raise ValueError('Unexpected TE chain structure.')
        term=component.entropy.terms[0]
        actual.append({'name':component.name,'training_seed':component.config.training_seed,
                       'target':term.target,'conditioning':list(term.conditioning),
                       'initialization':term.initialization_seed,'shuffle':term.shuffle_seed})
    if actual!=expected:raise ValueError('Executed TE seeds differ from frozen allocation.')


def run_process_replica(cell,replica,directory,references,*,cancelled=lambda:False):
    """One scientific worker; tests replace generator/fits with exact fixtures."""
    d=protocol.definition();rows=empty_rows(cell,replica);process=process_model(cell)
    manifest={'status':'running','cell_id':cell['cell_id'],'replica':replica,'environment':environment()}
    write_json(directory/'manifest.json',manifest)
    write_json(directory/'raw_results.json',rows)
    total_start=perf_counter();interruption=None
    try:
        x,y,generated=process.generate(cell['n'],seed=replica['data']['seed'],burn_in=d['burn_in'],initial=d['initial'])
        data_hash=hashlib.sha256(np.column_stack((x,y)).tobytes()).hexdigest()
        identity=f"{cell['cell_id']}:{replica['replica_id']}:{data_hash}"
        np.savez_compressed(directory/'process.npz',X=x,Y=y,time=np.arange(len(x)))
        manifest.update(dataset_id=identity,dataset_sha256=data_hash,process=generated,
                        initialization_tv_bound_after_burn_in=coupling_bound(process,d['burn_in'],d['common_span']))
        write_json(directory/'manifest.json',manifest)
        models={};reports={};support=None
        for row,job in zip(rows,cell['jobs']):
            if cancelled():raise KeyboardInterrupt('Cancellation before TE job.')
            start=perf_counter();row['status']='running';write_json(directory/'raw_results.json',rows)
            jd=directory/job['job_id'];jd.mkdir(exist_ok=False)
            try:
                estimator,offset=_aligned_fit(x,y,job=job,seed=row['training_seed'],identity=identity)
                if cancelled():raise KeyboardInterrupt('Cancellation after TE fit.')
                report=estimator.estimate()
                _audit_seeds(report,replica['jobs'][job['job_id']]['components'])
                timestamps=report.partitions[2].target_timestamps
                all_support=tuple(p.target_timestamps for p in report.partitions)
                if support is not None and all_support!=support:raise ValueError('Unaligned chronological populations.')
                support=all_support
                first,second=(c.entropy for c in report.cmi_report.components)
                score=first.terms[0].best_validation_ce_nats-second.terms[0].best_validation_ce_nats
                value=report.transfer_entropy_nats;truth=references[job['job_id']]
                if not np.isfinite([value,score]).all():raise ValueError('Nonfinite TE/validation score.')
                if value!=first.entropy_nats-second.entropy_nats:raise ValueError('TE reconstruction mismatch.')
                write_json(jd/'diagnostics.json',report)
                write_json(jd/'reference.json',truth)
                for ci,entry in enumerate(estimator._model._components):
                    for ti,term in enumerate(entry[4]._results):
                        np.savez_compressed(jd/f'checkpoint-c{ci}-t{ti}.npz',
                                            **{k:v.detach().cpu().numpy() for k,v in term.model.state_dict().items()})
                row.update(status='complete',te_nats=value,reference_nats=truth['transfer_entropy_nats'],
                           reference_status=truth['status'],reference_error_bound_nats=truth['entropy_truncation_error_bound_nats'],
                           signed_error_nats=value-truth['transfer_entropy_nats'],validation_score_nats=score,
                           dataset_id=report.cmi_report.dataset_id,split_id=report.cmi_report.split_id,
                           raw_origin_offset=offset,test_timestamps=list(map(int,timestamps)))
                if job['embedding']=='canonical':
                    models[job['job_id']]=estimator;reports[job['job_id']]=report
            except (Exception,KeyboardInterrupt) as exc:
                row.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                           failure={'type':type(exc).__name__,'message':str(exc)})
                write_json(jd/'failure.json',row['failure'])
                if isinstance(exc,KeyboardInterrupt):raise
            finally:
                row['runtime_seconds']=perf_counter()-start
                write_json(directory/'raw_results.json',rows)
        for direction in ('X_to_Y','Y_to_X'):
            selected=[r for r in rows if r['direction']==direction and r['embedding']=='canonical']
            if not all(r['status']=='complete' for r in selected):
                for row in selected:row['test_status']='incomplete_fits'
                continue
            if cancelled():raise KeyboardInterrupt('Cancellation before path inference.')
            start=perf_counter()
            try:
                source,target=(x,y) if direction=='X_to_Y' else (y,x)
                inference=frozen_path_test([models[r['job_id']] for r in selected],[reports[r['job_id']] for r in selected],
                                           source,target,process=process,direction=direction,
                                           seed=replica['surrogates'][direction]['seed'],draws=d['surrogate_draws'],cancelled=cancelled)
                if len(inference['p_values'])!=len(selected) or not all(type(p) in (int,float) and np.isfinite(p) and 0<=p<=1 for p in inference['p_values']):
                    raise ValueError('Invalid/incomplete surrogate p-value family.')
                write_json(directory/f'{direction}-surrogates.json',inference)
                for row,p in zip(selected,inference['p_values']):row.update(test_status='complete',p_value=p)
            except (Exception,KeyboardInterrupt) as exc:
                failure={'type':type(exc).__name__,'message':str(exc)}
                for row in selected:row.update(test_status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',test_failure=failure)
                write_json(directory/f'{direction}-surrogates-failure.json',failure)
                if isinstance(exc,KeyboardInterrupt):raise
            finally:
                for row in selected:row['surrogate_family_runtime_seconds']=perf_counter()-start
                write_json(directory/'raw_results.json',rows)
    except (Exception,KeyboardInterrupt) as exc:
        manifest['failure']={'type':type(exc).__name__,'message':str(exc)}
        manifest['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        if isinstance(exc,KeyboardInterrupt):interruption=exc
        # Generation failures mark unattempted jobs failed; interrupted pending jobs stay pending.
        if not isinstance(exc,KeyboardInterrupt):
            for row in rows:
                if row['status']=='not_attempted':row.update(status='failed',failure=manifest['failure'])
    else:
        manifest['status']='complete' if all(r['status']=='complete' and r['test_status'] in ('complete','not_applicable') for r in rows) else 'complete_with_failures'
    finally:
        manifest['runtime_seconds']=perf_counter()-total_start
        write_json(directory/'raw_results.json',rows)
        manifest['artifact_sha256']={p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in directory.rglob('*') if p.is_file() and p!=directory/'manifest.json'}
        write_json(directory/'manifest.json',manifest)
    if interruption is not None:raise interruption
    return rows,manifest


def read_cell(directory,cell):
    expected={(r['replica_id'],r['job_id']):r for replica in cell['allocation'] for r in empty_rows(cell,replica)}
    for replica in cell['allocation']:
        path=directory/replica['replica_id']/'raw_results.json'
        if not path.exists():continue
        saved=json.loads(path.read_text(encoding='utf-8'))
        ids={(r['replica_id'],r['job_id']) for r in saved}
        if len(saved)!=len(cell['jobs']) or ids!={(replica['replica_id'],j['job_id']) for j in cell['jobs']}:
            raise ValueError('Replica job ledger identity mismatch.')
        for row in saved:
            baseline=expected[row['replica_id'],row['job_id']]
            for key in ('cell_id','configuration_hash','data_seed','training_seed','direction','lag','embedding','source_lags','target_lags','horizon'):
                if row[key]!=baseline[key]:raise ValueError('Frozen row configuration/seed differs.')
            expected[row['replica_id'],row['job_id']]=row
    return list(expected.values())


def cell_summary(cell,rows):
    d=protocol.definition()
    return summarize_transfer(rows,[r['replica_id'] for r in cell['allocation']],cell['jobs'],
                              true_delay=cell['delay'],coupling=cell['coupling'],candidates=d['candidates'],
                              draws=d['bootstrap_draws'],seed=cell['bootstrap']['seed'])


def process_accounting(directory,cell):
    ledger=[]
    for replica in cell['allocation']:
        path=directory/replica['replica_id']/'manifest.json'
        manifest=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        ledger.append({'replica_id':replica['replica_id'],'status':manifest.get('status','not_attempted'),
                       'runtime_seconds':manifest.get('runtime_seconds'),'failure':manifest.get('failure')})
    return {'n_planned':len(ledger),'statuses':{s:sum(r['status']==s for r in ledger)
             for s in ('complete','complete_with_failures','failed','interrupted','running','not_attempted')},'ledger':ledger}


def report_cell(directory,cell):
    rows=read_cell(directory,cell);summary=cell_summary(cell,rows)
    summary['process_accounting']=process_accounting(directory,cell)
    write_json(directory/'raw_results.json',rows);write_csv(directory/'raw_results.csv',rows)
    write_json(directory/'summary.json',summary);write_csv(directory/'summary.csv',summary['summaries'])
    write_csv(directory/'direction_lag_accuracy.csv',[{'metric':k,**v} for k,v in summary['accuracy'].items()])
    write_csv(directory/'replica_metrics.csv',summary['replica_metrics'])
    terms=[];components=[]
    for replica in cell['allocation']:
        for job in cell['jobs']:
            path=directory/replica['replica_id']/job['job_id']/'diagnostics.json'
            if not path.exists():continue
            report=json.loads(path.read_text(encoding='utf-8'))
            for c in report['cmi_report']['components']:
                common={'cell_id':cell['cell_id'],'replica_id':replica['replica_id'],'job_id':job['job_id'],'component':c['name']}
                components.append({**common,'entropy_nats':c['entropy']['entropy_nats'],'config':c['config']})
                terms.extend({**common,**t} for t in c['entropy']['terms'])
    write_csv(directory/'components.csv',components);write_csv(directory/'term_diagnostics.csv',terms)
    return rows,summary


def export_tables(root,plan):
    all_rows=[];summaries=[];accuracy=[];replicas=[]
    for cell in plan['cells']:
        directory=root/cell['cell_id'];all_rows.extend(read_cell(directory,cell))
        path=directory/'summary.json'
        if not path.exists():
            summaries.extend({'cell_id':cell['cell_id'],**j,'status':'summary_unavailable'} for j in cell['jobs'])
            continue
        summary=json.loads(path.read_text(encoding='utf-8'))
        common={'cell_id':cell['cell_id'],'n':cell['n'],'coupling':cell['coupling'],'delay':cell['delay']}
        summaries.extend({**common,**s} for s in summary['summaries'])
        accuracy.extend({**common,'metric':k,**v} for k,v in summary['accuracy'].items())
        replicas.extend({**common,**r} for r in summary['replica_metrics'])
    write_csv(root/'replica_results.csv',all_rows);write_csv(root/'summary.csv',summaries)
    write_csv(root/'direction_lag_accuracy.csv',accuracy);write_csv(root/'replica_metrics.csv',replicas)


def execute(preparation,*,authorized=False,allow_local_windows=False,output_dir=None,log_dir=None,cancel_file=None):
    if not authorized:raise PermissionError('Separate scientific execution authorization required.')
    if os.name=='nt' and not allow_local_windows:raise PermissionError('Separate local Windows exception required.')
    plan=protocol.validate_bundle(preparation);git=_git_provenance()
    if git['dirty'] is not False:raise ValueError('A clean identifiable Git checkpoint is required.')
    root=Path(plan['output_root'] if output_dir is None else output_dir).resolve();prep=Path(preparation).resolve()
    logs=root/'logs' if log_dir is None else Path(log_dir).resolve()
    if root==prep or root in prep.parents or prep in root.parents:raise ValueError('Preparation/output overlap.')
    if root.exists() or logs.exists():raise FileExistsError('Existing output/log path; no overwrite/resume.')
    if logs==root or logs in root.parents or logs==prep or logs in prep.parents or prep in logs.parents:
        raise ValueError('Unsafe log/preparation/output overlap.')
    env=environment();root.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    write_json(root/'protocol.json',plan)
    manifest={'schema':protocol.SCHEMA,'status':'running','protocol_hash':plan['protocol_hash'],'git':git,
              'environment':env,'working_directory':str(Path.cwd()),'preparation':str(prep),'output_directory':str(root),
              'log_directory':str(logs),'started_utc':datetime.now(timezone.utc).isoformat()}
    ledger=[{'cell_id':cid,'status':'not_attempted'} for cid in plan['execution_order']]
    write_json(root/'manifest.json',manifest);write_json(root/'execution_ledger.json',ledger)
    by_id={c['cell_id']:c for c in plan['cells']};cancelled=lambda:cancel_file is not None and Path(cancel_file).exists()
    start=perf_counter()
    try:
        for entry in ledger:
            if cancelled():raise KeyboardInterrupt('Cancellation before cell.')
            cell=by_id[entry['cell_id']];directory=root/cell['cell_id'];directory.mkdir(exist_ok=False)
            entry['status']='running';write_json(root/'execution_ledger.json',ledger)
            write_json(directory/'configuration.json',cell)
            process=process_model(cell)
            refs={j['job_id']:reference_te(process,protocol.spec_for(j),direction=j['direction'],
                                           tolerance=protocol.definition()['reference']['tv_tolerance']) for j in cell['jobs']}
            write_json(directory/'references.json',refs)
            cell_start=perf_counter()
            try:
                for replica in cell['allocation']:
                    if cancelled():raise KeyboardInterrupt('Cancellation before process replica.')
                    rd=directory/replica['replica_id'];rd.mkdir(exist_ok=False)
                    try:
                        _,result=run_process_replica(cell,replica,rd,refs,cancelled=cancelled)
                    finally:
                        with (logs/(cell['cell_id']+'.jsonl')).open('a',encoding='utf-8') as f:
                            f.write(json.dumps({'replica_id':replica['replica_id'],'event':'worker_returned_or_interrupted'})+'\n')
            except BaseException:
                entry['status']='interrupted_or_failed';raise
            else:
                states=[json.loads((directory/r['replica_id']/'manifest.json').read_text(encoding='utf-8'))['status'] for r in cell['allocation']]
                entry['status']='complete' if all(s=='complete' for s in states) else 'complete_with_failures'
            finally:
                entry['runtime_seconds']=perf_counter()-cell_start
                report_cell(directory,cell);write_json(root/'execution_ledger.json',ledger)
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                        failure={'type':type(exc).__name__,'message':str(exc)})
        raise
    else:
        manifest['status']='complete' if all(e['status']=='complete' for e in ledger) else 'complete_with_failures'
    finally:
        try:export_tables(root,plan)
        except (Exception,KeyboardInterrupt) as exc:
            manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'reporting_failed',
                            reporting_failure={'type':type(exc).__name__,'message':str(exc)})
            raise
        finally:
            manifest.update(runtime_seconds=perf_counter()-start,finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(root/'manifest.json',manifest);write_json(root/'execution_ledger.json',ledger)
    return root


def inspect_results(root):
    root=Path(root);plan=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    if plan!=protocol.preview(plan['output_root']):raise ValueError('Result protocol hash/metadata mismatch.')
    output=[]
    for cell in plan['cells']:
        directory=root/cell['cell_id'];rows=read_cell(directory,cell)
        for replica in cell['allocation']:
            rd=directory/replica['replica_id'];mp=rd/'manifest.json'
            if not mp.exists():continue
            manifest=json.loads(mp.read_text(encoding='utf-8'))
            if manifest['replica']!=replica:raise ValueError('Replica seed provenance differs.')
            for name,sha in manifest.get('artifact_sha256',{}).items():
                path=rd/protocol.relative_path(name)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Replica artifact hash mismatch.')
        summary=cell_summary(cell,rows);summary['process_accounting']=process_accounting(directory,cell)
        output.append({'cell_id':cell['cell_id'],'summary':summary})
    return {'protocol_hash':plan['protocol_hash'],'cells':output,
            'manifest':json.loads((root/'manifest.json').read_text(encoding='utf-8')) if (root/'manifest.json').exists() else None,
            'execution_ledger':json.loads((root/'execution_ledger.json').read_text(encoding='utf-8')) if (root/'execution_ledger.json').exists() else None}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Preview, prepare, execute or report the frozen transfer entropy study; execution requires authorization.');group=parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true');group.add_argument('--prepare',type=Path)
    group.add_argument('--execute',type=Path);group.add_argument('--report',type=Path)
    parser.add_argument('--authorize-execution',action='store_true');parser.add_argument('--allow-local-windows',action='store_true')
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--log-dir',type=Path);parser.add_argument('--cancel-file',type=Path)
    args=parser.parse_args(argv)
    if args.execute is None and (args.authorize_execution or args.allow_local_windows or args.log_dir or args.cancel_file):
        parser.error('Execution controls require --execute.')
    protocol.load_definition()
    if args.execute:
        print(execute(args.execute,authorized=args.authorize_execution,allow_local_windows=args.allow_local_windows,
                      output_dir=args.output_dir,log_dir=args.log_dir,cancel_file=args.cancel_file))
    elif args.prepare:print(json.dumps(protocol.materialize(args.prepare,output_root=args.output_dir)['expected']))
    elif args.report:
        if args.output_dir:parser.error('--report is read-only.')
        print(json.dumps(inspect_results(args.report),allow_nan=False))
    else:
        plan=protocol.preview(args.output_dir)
        print(json.dumps({k:plan[k] for k in ('protocol_hash','expected','execution_order','seed_audit','output_root')},indent=2))


if __name__=='__main__':main()
