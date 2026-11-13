"""Process-replica summaries, validation-only lag selection and conditional tests."""
from dataclasses import replace
import math

import numpy as np

from src.metrics.aggregation import _metrics, _intervals, bootstrap_seed_for
from src.metrics.information import binomial_interval
from src.utils.reproducibility import validate_seed
from src.utils.temporal_surrogates import conditional_source_path


def family_decisions(p_values,*,alpha=.05,family_size=8):
    """Fixed Bonferroni family, NEVER shrink on failure. Incomplete => no decisions."""
    if len(p_values)!=family_size or not 0<alpha<1:
        raise ValueError('Unexpected hypothesis family.')
    finite=[p is not None and type(p) in (int,float) and math.isfinite(p) and 0<=p<=1 for p in p_values]
    if any(p is not None and not good for p,good in zip(p_values,finite)):
        raise ValueError('Invalid completed p-value.')
    if not all(finite):
        return {'complete':False,'adjusted_p_values':[None]*family_size,'reject':[None]*family_size,
                'any_reject':None,'family_size':family_size,'alpha':alpha}
    adjusted=[min(1.,p*family_size) for p in p_values]
    rejects=[p<=alpha for p in adjusted]
    return {'complete':True,'adjusted_p_values':adjusted,'reject':rejects,
            'any_reject':any(rejects),'family_size':family_size,'alpha':alpha}


def frozen_path_test(models,reports,source,target,*,process,direction,seed,draws=999,cancelled=lambda:False):
    """One direction's canonical-lag statistics, frozen fits and common target times.

    Models/reports are in fixed candidate order. Generate whole conditional paths,
    jointly evaluating candidates; neither candidates nor surrogates are replicas.
    The adapter changes only source-past TEST columns in the existing CMI batch.
    """
    validate_seed(seed)
    if type(draws) is not int or draws<1 or not models or len(models)!=len(reports):
        raise ValueError('Invalid surrogate test inputs.')
    times=reports[0].partitions[2].target_timestamps
    cutoff=int(reports[0].partitions[1].target_timestamps[-1])
    observed=[r.transfer_entropy_nats for r in reports]
    if not all(math.isfinite(t) for t in observed):raise ValueError('Invalid observed statistic.')
    for report in reports:
        if report.partitions[2].target_timestamps!=times or report.embedding.target_lags!=(1,):
            raise ValueError('Primary tests require aligned canonical own-history (1,).')
        if report.partitions[1].target_timestamps[-1]!=cutoff:
            raise ValueError('Training/validation prefixes differ.')
    rng=np.random.default_rng(seed);statistics=[]
    for _ in range(draws):
        if cancelled():raise KeyboardInterrupt('Cancellation during conditional path inference.')
        surrogate=conditional_source_path(source,target,model=process,direction=direction,cutoff=cutoff,rng=rng)
        values=[]
        for model,report in zip(models,reports):
            batch=model._parts.test
            table=batch.values.copy()
            for j in range(len(report.embedding.source_lags)):
                positions=np.array([row[1+j] for row in report.partitions[2].observation_timestamps],dtype=np.int64)
                if positions.min()<=cutoff:raise ValueError('Surrogate context crosses fitted prefix.')
                table[:,1+j]=surrogate[positions]
            value=model._model.estimate(replace(batch,values=table)).information_nats
            if not math.isfinite(value):raise RuntimeError('Nonfinite surrogate statistic.')
            values.append(value)
        statistics.append(values)
    exceed=np.sum(np.asarray(statistics)>=observed,axis=0)
    return {'method':'oracle-conditional-path-frozen-statistic-v1','direction':direction,'seed':seed,
            'cutoff':cutoff,'draws':draws,'observed_nats':observed,
            'p_values':[(1+int(k))/(draws+1) for k in exceed],
            'surrogate_statistics_nats':statistics,'refits':0,
            'assumption':'specified path kernel and structural null; not a distribution-free TE test'}


def select_lag(rows,candidates):
    """Validation scores only; ties follow the declared candidate order."""
    by_lag={r['lag']:r for r in rows if r['embedding']=='canonical'}
    if set(by_lag)!=set(candidates) or any(by_lag[k]['status']!='complete' for k in candidates):
        return {'status':'incomplete','selected_lag':None,'test_te_nats':None}
    if not all(math.isfinite(by_lag[k]['validation_score_nats']) for k in candidates):
        raise ValueError('Nonfinite validation selection score.')
    lag=max(candidates,key=lambda k:by_lag[k]['validation_score_nats'])
    return {'status':'complete','selected_lag':lag,'test_te_nats':by_lag[lag]['te_nats'],
            'selection':'max frozen-checkpoint validation CE difference; declared-order ties'}


def summarize_transfer(rows,replicas,jobs,*,true_delay,coupling,candidates,draws=10000,seed=271818):
    expected={(r,j['job_id']) for r in replicas for j in jobs}
    if len(rows)!=len(expected) or {(r['replica_id'],r['job_id']) for r in rows}!=expected:
        raise ValueError('Missing/duplicate/unplanned process-job records.')
    if any(r['status'] not in ('complete','failed','interrupted','running','not_attempted') for r in rows):
        raise ValueError('Unknown job status.')
    schema={j['job_id']:j for j in jobs}
    if any(any(r[k]!=v for k,v in schema[r['job_id']].items()) for r in rows):
        raise ValueError('Record job metadata differs from the declared design.')
    summaries=[]
    for job in jobs:
        selected=[r for r in rows if r['job_id']==job['job_id']]
        valid=[r for r in selected if r['status']=='complete']
        for row in valid:
            if (not all(math.isfinite(row[k]) for k in ('te_nats','reference_nats','signed_error_nats'))
                    or row['signed_error_nats']!=row['te_nats']-row['reference_nats']):
                raise ValueError('Invalid completed TE/error record.')
        estimates=np.array([r['te_nats'] for r in valid]);truths=np.array([r['reference_nats'] for r in valid])
        bootstrap=bootstrap_seed_for(seed,job['job_id'])
        summaries.append({**job,'n_planned':len(replicas),'n_complete':len(valid),
                          'n_failed':sum(r['status'] in ('failed','interrupted','running') for r in selected),
                          'n_not_attempted':sum(r['status']=='not_attempted' for r in selected),
                          'metrics':_metrics(estimates,truths) if valid else None,
                          'reference_statuses':sorted({r.get('reference_status','fixture') for r in valid}),
                          'max_reference_truncation_bound_nats':max((r.get('reference_error_bound_nats',0.) for r in valid),default=None),
                          'bootstrap':_intervals(estimates,truths,draws=draws,seed=bootstrap),
                          'negative_frequency':float(np.mean(estimates<0)) if valid else None,
                          'mean_runtime_seconds':float(np.mean([r['runtime_seconds'] for r in valid])) if valid else None})
    by_replica=[]
    for replica in replicas:
        own=[r for r in rows if r['replica_id']==replica]
        selection={d:select_lag([r for r in own if r['direction']==d],candidates) for d in ('X_to_Y','Y_to_X')}
        oracle={r['direction']:r for r in own if r['embedding']=='canonical' and r['lag']==true_delay}
        delta=None
        if len(oracle)==2 and all(r['status']=='complete' for r in oracle.values()):
            if oracle['X_to_Y']['test_timestamps']!=oracle['Y_to_X']['test_timestamps']:
                raise ValueError('Direction comparison has non-comparable test support.')
            delta=oracle['X_to_Y']['te_nats']-oracle['Y_to_X']['te_nats']
        primary=[r for r in own if r['embedding']=='canonical']
        decisions=family_decisions([r.get('p_value') if r.get('test_status')=='complete' else None for r in primary],
                                   family_size=2*len(candidates))
        runtimes={}
        for direction in ('X_to_Y','Y_to_X'):
            directional=[r for r in own if r['direction']==direction]
            times=[r['runtime_seconds'] for r in directional if r.get('runtime_seconds') is not None]
            families=[r['surrogate_family_runtime_seconds'] for r in directional if r.get('surrogate_family_runtime_seconds') is not None]
            runtimes[direction]={'observed_job_seconds':math.fsum(times) if times else None,
                                 'n_job_timings':len(times),'surrogate_family_seconds':max(families) if families else None,
                                 'note':'one shared surrogate family per direction; do not sum its repeated row field'}
        by_replica.append({'replica_id':replica,'selection':selection,'oracle_direction_difference_nats':delta,
                           'direction_runtime':runtimes,
                           'direction_correct':delta>0 if coupling>0 and delta is not None else None,
                           'lag_correct':selection['X_to_Y']['selected_lag']==true_delay
                           if coupling>0 and selection['X_to_Y']['status']=='complete' else None,
                           'family':decisions,'false_link':decisions['any_reject'] if coupling==0 else None})
    accuracy={}
    for field in ('direction_correct','lag_correct','false_link'):
        decisions=[r[field] for r in by_replica if r[field] is not None]
        count=sum(decisions);n=len(decisions)
        eligible=(coupling==0)==(field=='false_link')
        accuracy[field]={'n_eligible':len(replicas) if eligible else 0,
                         'n_evaluated':n,'n_missing_or_inapplicable':len(replicas)-n,
                         'count':count,'rate':count/n if n else None,'interval':binomial_interval(count,n),
                         'planned_bounds':[count/len(replicas),(count+len(replicas)-n)/len(replicas)]
                                          if eligible else None}
    deltas=np.array([r['oracle_direction_difference_nats'] for r in by_replica if r['oracle_direction_difference_nats'] is not None])
    return {'summaries':summaries,'replica_metrics':by_replica,'accuracy':accuracy,
            'direction_difference':{'n_pairs':len(deltas),'mean_nats':float(np.mean(deltas)) if len(deltas) else None,
                                    'bootstrap':_intervals(deltas,np.zeros(len(deltas)),draws=draws,
                                                           seed=bootstrap_seed_for(seed,'direction-difference'))['mean_estimate_nats']},
            'inference':'pointwise exploratory replica intervals; fixed eight-test Bonferroni within each process replica',
            'unit':'independent process replica; no within-series IID bootstrap'}
