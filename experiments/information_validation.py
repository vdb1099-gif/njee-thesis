import argparse
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping

import numpy as np
import torch

from experiments import information_protocol as protocol
from experiments.neural_baseline import _git_provenance
from src.metrics.information import classical_information, heldout_permutation_test, summarize_information
from src.simulation.information import InformationModel
from src.utils.categorical import CategoricalDataset
from src.utils.reproducibility import capture_environment
from src.utils.split import make_iid_split, save_split


VARIANTS = ('neural', 'plug_in_full', 'miller_madow_full', 'plug_in_train', 'miller_madow_train')


def json_value(value):
    if is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    """Only replace files inside a newly exclusively-owned run, never another run."""
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    names = sorted({k for r in rows for k in r})
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(json_value(v), allow_nan=False) if isinstance(v, (dict,list,tuple))
                             else v for k,v in row.items()})


def environment():
    return {**capture_environment(mode='strict', device='cpu'),
            'python_executable': sys.executable, 'torch_num_interop_threads': torch.get_num_interop_threads(),
            'thread_environment': {k: os.environ.get(k) for k in
                                   ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            'note': 'Caller backend flags; scoped active training flags are in term diagnostics. '
                    'No cross-environment bitwise guarantee. Peak memory not measured.'}


def _empty_rows(cell):
    truth = cell['truth']
    return [{'campaign_id':cell['cell_id'], 'scenario_id':cell['configuration_hash'],
             'replica_id':r['replica_id'], 'variant':v, 'status':'not_attempted',
             'information_nats':None, 'entropy_nats':None, 'ground_truth_nats':truth['information_nats'],
             'signed_error_nats':None, 'first_entropy_nats':None, 'second_entropy_nats':None,
             'first_truth_nats':truth['first_entropy_nats'], 'second_truth_nats':truth['second_entropy_nats'],
             'is_null':truth['is_null'], 'runtime_seconds':None, 'fit_runtime_seconds':None,
             'inference_runtime_seconds':None, 'failure':None,
             'test_status':'not_attempted' if v=='neural' and truth['is_null'] else 'not_applicable',
             'p_value':None, 'reject':None, 'seeds':r,
             'independence_runtime_seconds':None, 'replica_wall_runtime_seconds':None,
             'sample_protocol':('train_plugin_plus_test_ce' if cell['model']['family']=='bsc' else 'test_conditional_ce')
                               if v=='neural' else v.rsplit('_',1)[1]}
            for r in cell['allocation'] for v in VARIANTS]


def run_replica(cell, replica, directory, *, cancelled=lambda: False):
    """Actual scientific worker; call only through separately authorized execution.

    A unit test may replace the neural fits with controlled fixtures. Never called
    by preview/preparation/reporting. Saves generated data and exact split arrays.
    """
    model = InformationModel(**cell['model'])
    truth = model.truths()
    start = perf_counter()
    values = model.generate(cell['n'], seed=replica['data_seed'])
    content_hash = hashlib.sha256(values.tobytes(order='C')).hexdigest()
    identity = f"{cell['cell_id']}:{replica['replica_id']}:{content_hash}"
    split = make_iid_split(cell['n'], seed=replica['split_seed'])
    split_id = f"{identity}:{split.algorithm}:{split.seed}"
    dataset = CategoricalDataset(values, identity, model.variable_ids)
    train, validation, test = (dataset.batch(split, role, split_id=split_id)
                               for role in ('train','validation','test'))
    np.savez_compressed(directory/'dataset.npz', values=values)
    save_split(split, directory/'split.npz')
    metadata = {'dataset_id':identity, 'split_id':split_id, 'dataset_sha256':content_hash,
                'variable_ids':model.variable_ids, 'seeds':replica, 'truth':truth,
                'sample_counts':{'full':len(values), 'train':len(train.values),
                                 'validation':len(validation.values), 'test':len(test.values)},
                'environment':environment(), 'status':'fitting'}
    write_json(directory/'manifest.json', metadata)
    config = protocol.neural_config(replica['split_seed'], replica['training_seed'])
    estimator = protocol.make_estimator(model, config)
    if cancelled():
        raise KeyboardInterrupt('Cancellation requested before fit.')
    fit_start = perf_counter()
    estimator.fit(train, validation)
    fit_time = perf_counter()-fit_start
    if cancelled():
        raise KeyboardInterrupt('Cancellation requested after fit.')
    eval_start = perf_counter()
    report = estimator.estimate(test)
    eval_time = perf_counter()-eval_start
    a,b = (c.entropy.entropy_nats for c in report.components)
    if report.information_nats != a-b or not np.isfinite([a,b,report.information_nats]).all():
        raise RuntimeError('Neural information reconstruction failed.')
    # Audit that the public estimator executed the frozen effective seed plan.
    actual = [{'name':c.name, 'training_seed':c.config.training_seed,
               'terms':[{'target':t.target, 'conditioning':list(t.conditioning),
                         'initialization':t.initialization_seed, 'shuffle':t.shuffle_seed}
                        for t in c.entropy.terms if t.method != 'train_plugin']}
              for c in report.components]
    if actual != replica['components']:
        raise RuntimeError('Executed component seeds differ from frozen allocation.')
    write_json(directory/'neural_diagnostics.json', report)
    # Portable, non-pickle checkpoint arrays; architecture/config is in diagnostics.
    for component_index, entry in enumerate(estimator._components):
        for term_index, term in enumerate(entry[4]._results):
            np.savez_compressed(directory/f'checkpoint-c{component_index}-t{term_index}.npz',
                                **{k:v.detach().cpu().numpy() for k,v in term.model.state_dict().items()})
    results = {'neural': {'information_nats':report.information_nats,
                          'first_entropy_nats':a, 'second_entropy_nats':b,
                          'fit_runtime_seconds':fit_time, 'inference_runtime_seconds':eval_time,
                          'runtime_seconds':fit_time+eval_time}}
    for variant in VARIANTS[1:]:
        method, scope = variant.rsplit('_',1)
        tick = perf_counter()
        result = classical_information(values if scope=='full' else train.values, model.variable_ids,
                                        conditional=model.family!='bsc', method=method)
        elapsed = perf_counter()-tick
        results[variant] = {**result, 'runtime_seconds':elapsed,
                            'fit_runtime_seconds':0., 'inference_runtime_seconds':elapsed}
    # Save estimates BEFORE inference; inference failure must not erase them.
    write_json(directory/'estimates.json', results)
    inference = None
    inference_start = perf_counter()
    if truth['is_null']:
        try:
            inference = heldout_permutation_test(
                estimator, test, conditional=model.family!='bsc', seed=replica['permutation_seed'],
                draws=protocol.definition()['permutation_draws'], alpha=protocol.definition()['test_alpha'],
                observed=report.information_nats, cancelled=cancelled)
            write_json(directory/'independence_test.json', inference)
        except (Exception, KeyboardInterrupt) as exc:
            inference = {'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                         'failure':{'type':type(exc).__name__, 'message':str(exc)}}
            write_json(directory/'independence_test.json', inference)
    metadata.update(status='complete', runtime_seconds=perf_counter()-start,
                    test_status=inference.get('status','complete') if inference is not None else 'not_applicable',
                    independence_runtime_seconds=perf_counter()-inference_start if truth['is_null'] else 0.,
                    artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in directory.iterdir() if p.is_file() and p.name!='manifest.json'})
    write_json(directory/'manifest.json', metadata)
    return results, inference, metadata


def execute(preparation, *, authorized=False, allow_local_windows=False, output_dir=None,
            cancel_file=None, log_dir=None):
    if not authorized:
        raise PermissionError('Scientific execution requires explicit authorization.')
    if os.name == 'nt' and not allow_local_windows:
        raise PermissionError('Separate local Windows execution exception required.')
    execution_start = perf_counter()
    plan = protocol.validate_bundle(preparation)
    git = _git_provenance()
    if git['dirty'] is not False:
        raise ValueError('Execution requires a clean, identifiable Git checkpoint.')
    root = Path(plan['output_root'] if output_dir is None else output_dir).resolve()
    prep = Path(preparation).resolve()
    if root == prep or root in prep.parents or prep in root.parents:
        raise ValueError('Output and preparation paths must be disjoint.')
    logs = root/'logs' if log_dir is None else Path(log_dir).resolve()
    if logs == root or logs in root.parents or logs.exists() or root.exists():
        raise FileExistsError('Output/log directories must not already exist.')
    if logs == prep or prep in logs.parents or logs in prep.parents:
        raise ValueError('Log directory overlaps preparation bundle.')
    caller_environment = environment()
    root.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    write_json(root/'protocol.json', plan)
    manifest = {'schema':protocol.SCHEMA, 'protocol_hash':plan['protocol_hash'], 'git':git,
                'environment':caller_environment, 'status':'running', 'working_directory':str(Path.cwd()),
                'output_directory':str(root), 'log_directory':str(logs),
                'preparation_directory':str(prep),
                'started_utc':datetime.now(timezone.utc).isoformat(),
                'policy':'sequential; no automatic retry/resume; preserve all failed/unattempted replicas'}
    write_json(root/'manifest.json', manifest)
    ledger = [{'cell_id':cid, 'status':'not_attempted'} for cid in plan['execution_order']]
    write_json(root/'execution_ledger.json', ledger)
    by_id = {c['cell_id']:c for c in plan['cells']}
    cancelled = lambda: cancel_file is not None and Path(cancel_file).exists()
    interruption = None
    try:
        for entry in ledger:
            if cancelled():
                raise KeyboardInterrupt('Cancellation requested before cell.')
            cell = by_id[entry['cell_id']]
            cell_start = perf_counter()
            directory = root/cell['cell_id']
            directory.mkdir(exist_ok=False)
            write_json(directory/'configuration.json', cell)
            rows = _empty_rows(cell)
            write_json(directory/'raw_results.json', rows)
            entry['status'] = 'running'
            write_json(root/'execution_ledger.json', ledger)
            for replica in cell['allocation']:
                if cancelled():
                    interruption = KeyboardInterrupt('Cancellation requested before replica.')
                    break
                selected = [r for r in rows if r['replica_id']==replica['replica_id']]
                rd = directory/replica['replica_id']
                rd.mkdir(exist_ok=False)
                for r in selected:
                    r['status'] = 'running'
                write_json(directory/'raw_results.json', rows)
                replica_start = perf_counter()
                try:
                    results, inference, metadata = run_replica(cell, replica, rd, cancelled=cancelled)
                    # Validate every variant before publishing any successful ledger row.
                    if set(results) != set(VARIANTS):
                        raise ValueError('Missing or unexpected estimator variants.')
                    for result in results.values():
                        numbers = [result[k] for k in ('information_nats','first_entropy_nats','second_entropy_nats')]
                        if not np.isfinite(numbers).all() or numbers[0] != numbers[1]-numbers[2]:
                            raise ValueError('Invalid result arithmetic.')
                    for r in selected:
                        result = results[r['variant']]
                        r.update({k:v for k,v in result.items() if k!='information_nats'})
                        estimate = result['information_nats']
                        r.update(status='complete', information_nats=estimate, entropy_nats=estimate,
                                 signed_error_nats=estimate-cell['truth']['information_nats'],
                                 dataset_id=metadata['dataset_id'], split_id=metadata['split_id'],
                                 sample_counts=metadata['sample_counts'],
                                 replica_wall_runtime_seconds=metadata['runtime_seconds'],
                                 independence_runtime_seconds=metadata['independence_runtime_seconds']
                                                              if r['variant']=='neural' else 0.)
                        if r['variant']=='neural' and inference is not None:
                            r['test_status'] = inference.get('status','complete')
                            r['p_value'],r['reject'] = inference.get('p_value'),inference.get('reject')
                            r['test_failure'] = inference.get('failure')
                            if inference.get('status') == 'interrupted':
                                interruption = KeyboardInterrupt('Interrupted during inference; estimates preserved.')
                except (Exception, KeyboardInterrupt) as exc:
                    status = 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
                    for r in selected:
                        r.update(status=status, failure={'type':type(exc).__name__, 'message':str(exc)},
                                 replica_wall_runtime_seconds=perf_counter()-replica_start)
                    write_json(rd/'failure.json', selected[0]['failure'])
                    mp = rd/'manifest.json'
                    if mp.exists():
                        failed = json.loads(mp.read_text(encoding='utf-8'))
                        failed.update(status=status, failure=selected[0]['failure'])
                        write_json(mp, failed)
                    if isinstance(exc,KeyboardInterrupt):
                        interruption = exc
                write_json(directory/'raw_results.json', rows)
                with (logs/f"{cell['cell_id']}.jsonl").open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'replica_id':replica['replica_id'], 'status':selected[0]['status']})+'\n')
                if interruption is not None:
                    break
            entry['status'] = ('interrupted' if interruption is not None else 'complete'
                               if all(r['status']=='complete' and r['test_status'] not in ('failed','interrupted')
                                      for r in rows) else 'complete_with_failures')
            entry['runtime_seconds'] = perf_counter()-cell_start
            entry['failed_replicas'] = sum(r['status'] in ('failed','interrupted') for r in rows if r['variant']=='neural')
            entry['failed_independence_tests'] = sum(r['test_status'] in ('failed','interrupted') for r in rows)
            write_json(directory/'ledger.json', [{'replica_id':r['replica_id'], 'status':r['status'],
                                                  'failure':r['failure']} for r in rows if r['variant']=='neural'])
            _report_cell(directory, cell, rows)
            write_json(root/'execution_ledger.json', ledger)
            if interruption is not None:
                raise interruption
    except BaseException as exc:
        manifest['status'] = 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        manifest['failure'] = {'type':type(exc).__name__, 'message':str(exc)}
        raise
    else:
        manifest['status'] = 'complete' if all(e['status']=='complete' for e in ledger) else 'complete_with_failures'
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        try:
            export_tables(root, plan)
        except (Exception, KeyboardInterrupt) as exc:
            manifest['status'] = 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'reporting_failed'
            manifest['reporting_failure'] = {'type':type(exc).__name__, 'message':str(exc)}
            raise
        finally:
            manifest['runtime_seconds'] = perf_counter()-execution_start
            write_json(root/'execution_ledger.json', ledger)
            write_json(root/'manifest.json', manifest)
    return root


def validate_rows(cell, rows):
    expected = {(r['replica_id'],r['variant']):r for r in _empty_rows(cell)}
    if len(rows)!=len(expected) or {(r['replica_id'],r['variant']) for r in rows}!=set(expected):
        raise ValueError('Raw ledger identity mismatch.')
    by_replica = {}
    for row in rows:
        baseline = expected[row['replica_id'],row['variant']]
        for name in ('campaign_id','scenario_id','seeds','ground_truth_nats','first_truth_nats',
                     'second_truth_nats','is_null','sample_protocol'):
            if row[name] != baseline[name]:
                raise ValueError('Raw record differs from frozen cell/seed/truth contract.')
        if row['status']=='complete':
            identity = (row['dataset_id'],row['split_id'],row['sample_counts'])
            if row['replica_id'] in by_replica and by_replica[row['replica_id']] != identity:
                raise ValueError('Estimator variants do not share partitions/identities.')
            by_replica[row['replica_id']] = identity


def _report_cell(directory, cell, rows):
    validate_rows(cell, rows)
    summary = summarize_information(rows, [r['replica_id'] for r in cell['allocation']],
                                    bootstrap_draws=protocol.definition()['bootstrap_draws'],
                                    bootstrap_seed=cell['allocation'][0]['bootstrap_seed'])
    write_json(directory/'summary.json', summary)
    write_csv(directory/'raw_results.csv', rows)
    write_csv(directory/'summary.csv', summary['summaries'])
    write_csv(directory/'paired_comparisons.csv', summary['paired_comparisons'])
    components, terms = [], []
    for replica in cell['allocation']:
        path = directory/replica['replica_id']/'neural_diagnostics.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding='utf-8'))
        for index, component in enumerate(report['components']):
            common = {'cell_id':cell['cell_id'], 'replica_id':replica['replica_id'], 'component':component['name']}
            components.append({**common, 'entropy_nats':component['entropy']['entropy_nats'],
                               'truth_nats':cell['truth']['first_entropy_nats'] if index==0
                                            else cell['truth']['second_entropy_nats'],
                               'config':component['config']})
            terms.extend({**common, **t} for t in component['entropy']['terms'])
    write_csv(directory/'components.csv', components)
    write_csv(directory/'term_diagnostics.csv', terms)
    return summary


def inspect_results(root):
    """Read-only full-matrix report; missing cells remain explicit, never dropped."""
    root = Path(root)
    plan = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    if plan != protocol.preview(plan['output_root']):
        raise ValueError('Result protocol integrity check failed.')
    cells = []
    for cell in plan['cells']:
        path = root/cell['cell_id']/'raw_results.json'
        rows = json.loads(path.read_text(encoding='utf-8')) if path.exists() else _empty_rows(cell)
        validate_rows(cell, rows)
        for replica in cell['allocation']:
            mp = root/cell['cell_id']/replica['replica_id']/'manifest.json'
            if mp.exists():
                manifest = json.loads(mp.read_text(encoding='utf-8'))
                if (manifest['seeds'] != replica or manifest['truth'] != cell['truth']
                        or manifest['variable_ids'] != list(InformationModel(**cell['model']).variable_ids)):
                    raise ValueError('Replica manifest provenance mismatch.')
                for row in rows:
                    if row['replica_id']==replica['replica_id'] and row['status']=='complete':
                        if any(row[k]!=manifest[k] for k in ('dataset_id','split_id','sample_counts')):
                            raise ValueError('Raw/manifest identity mismatch.')
                for name, sha in manifest.get('artifact_sha256',{}).items():
                    if (any(s in name for s in ('/','\\',':')) or Path(name).name!=name
                            or hashlib.sha256((mp.parent/name).read_bytes()).hexdigest()!=sha):
                        raise ValueError('Replica artifact hash mismatch.')
        summary = summarize_information(rows, [r['replica_id'] for r in cell['allocation']],
                                        bootstrap_draws=protocol.definition()['bootstrap_draws'],
                                        bootstrap_seed=cell['allocation'][0]['bootstrap_seed'])
        cells.append({'cell_id':cell['cell_id'], 'summary':summary})
    return {'protocol_hash':plan['protocol_hash'], 'cells':cells,
            'manifest':json.loads((root/'manifest.json').read_text(encoding='utf-8')) if (root/'manifest.json').exists() else None,
            'execution_ledger':json.loads((root/'execution_ledger.json').read_text(encoding='utf-8'))
                               if (root/'execution_ledger.json').exists() else None,
            'inference':'Pointwise exploratory; no final superiority decisions or cross-cell pooling.'}


def export_tables(root, plan):
    """Automatic tables inside the owned run, including pending-cell placeholders.

    No additional fitting or inference, no recomputation of completed bootstrap
    summaries. These portable CSVs are the source tables for thesis figures.
    """
    raw, summaries, pairs, fpr = [], [], [], []
    for cell in plan['cells']:
        directory = root/cell['cell_id']
        path = directory/'raw_results.json'
        raw.extend(json.loads(path.read_text(encoding='utf-8')) if path.exists() else _empty_rows(cell))
        sp = directory/'summary.json'
        if sp.exists():
            summary = json.loads(sp.read_text(encoding='utf-8'))
            summaries.extend({'cell_id':cell['cell_id'], 'n':cell['n'], **s} for s in summary['summaries'])
            pairs.extend({'cell_id':cell['cell_id'], **p} for p in summary['paired_comparisons'])
            fpr.append({'cell_id':cell['cell_id'], **summary['false_positive_analysis']})
        else:
            summaries.extend({'cell_id':cell['cell_id'], 'n':cell['n'], 'variant':v,
                              'n_planned':cell['replicas'], 'status':'summary_unavailable'} for v in VARIANTS)
            if cell['truth']['is_null']:
                fpr.append({'cell_id':cell['cell_id'], 'n_planned_null':cell['replicas'],
                            'n_tested':0, 'n_untested':cell['replicas'], 'rate':None,
                            'status':'summary_unavailable'})
    write_csv(root/'replica_results.csv', raw)
    write_csv(root/'summary.csv', summaries)
    write_csv(root/'paired_comparisons.csv', pairs)
    write_csv(root/'false_positive_rates.csv', fpr)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Preview, prepare, execute or report the frozen MI/CMI study; execution requires authorization.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--prepare', type=Path)
    group.add_argument('--execute', type=Path)
    group.add_argument('--report', type=Path)
    parser.add_argument('--authorize-execution', action='store_true')
    parser.add_argument('--allow-local-windows', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--cancel-file', type=Path)
    args = parser.parse_args(argv)
    if args.execute is None and (args.authorize_execution or args.allow_local_windows or args.log_dir or args.cancel_file):
        parser.error('Execution controls require --execute.')
    protocol.load_definition()
    if args.execute is not None:
        print(execute(args.execute,authorized=args.authorize_execution,
                      allow_local_windows=args.allow_local_windows,output_dir=args.output_dir,
                      cancel_file=args.cancel_file,log_dir=args.log_dir))
    elif args.prepare is not None:
        print(json.dumps(protocol.materialize(args.prepare,output_root=args.output_dir)['expected'],indent=2))
    elif args.report is not None:
        if args.output_dir:
            parser.error('--report is read-only; --output-dir is unsupported.')
        print(json.dumps(inspect_results(args.report),indent=2,allow_nan=False))
    else:
        plan = protocol.preview(args.output_dir)
        print(json.dumps({k:plan[k] for k in ('protocol_hash','expected','execution_order','output_root','seed_audit')},indent=2))


if __name__ == '__main__':
    main()
