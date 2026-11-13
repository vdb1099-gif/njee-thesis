from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from experiments.final_protocol import digest,relative_path
from experiments.information_protocol import preview as information_preview, _prior_seeds as information_prior
from src.estimators.neural_information import NeuralConditionalMutualInformationEstimator
from src.utils.neural_config import validate_neural_config
from src.utils.reproducibility import derive_term_seed
from src.utils.temporal import TemporalEmbeddingSpec


SCHEMA='njee-experiment3-v1'
CONFIG_PATH=Path(__file__).resolve().parents[1]/'configs'/'transfer_validation.yaml'


def definition():
    return {'schema':SCHEMA,'couplings':[.1,.3,.5,.8],'delays':[1,2,3,5],
            'lengths':[1000,5000,10000],'replicas':50,'null_delay':1,
            'noise_x':.1,'noise_y':.1,'burn_in':1000,'initial':'random',
            'candidates':[1,2,3,5],'common_span':6,'gap':6,'purge':True,
            'surrogate_draws':999,'alpha':.05,'family_size':8,
            'master_seed_start':180000,'bootstrap_master':271818,'bootstrap_draws':10000,
            'output_root':'results/experiment3_final',
            'reference':{'method':'pair-chain-power-coupling-v1','tv_tolerance':1e-13,'state_memory_cap':8,
                         'floating_point_bound':'truncation bound excludes roundoff'},
            'inference':{'null':'forward no-coupling / reverse no-feedback under known path kernels',
                         'surrogate':'oracle-conditional-path-frozen-statistic-v1',
                         'multiple_comparisons':'Bonferroni eight canonical lag-direction tests per replica',
                         'lag_selection':'validation CE difference; ascending candidate ties; separate test estimation',
                         'accuracy':'oracle true-delay signed test-direction difference; forward validation-lag accuracy',
                         'intervals':'pointwise exploratory independent-process-replica percentile bootstrap',
                         'failure':'no retry/replacement; incomplete family has no decision'},
            'neural':{'split_seed':0,'split_fractions':[.6,.2,.2],'training_seed':0,
                      'component_order':['placeholder'],'alphabet_policy':'declared','marginal_method':'train_plugin',
                      'hidden_sizes':[16],'activation':'relu','optimizer':'adam','learning_rate':.01,
                      'batch_size':64,'max_epochs':20,'patience':5,'min_delta':.0001,'device':'cpu',
                      'reproducibility_mode':'strict','num_workers':0}}


def load_definition(path=CONFIG_PATH):
    config=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if config!=definition():raise ValueError('Experiment 3 definition differs from frozen source policy.')
    return config


def jobs(delay):
    specs=[('canonical',lag,(lag,),(1,)) for lag in definition()['candidates']]
    specs.extend([('insufficient',delay,(delay,),(2,)),('excessive',delay,(delay,delay+1),(1,2))])
    return [{'job_id':f'{direction}-{kind}-lag{lag}','direction':direction,'embedding':kind,'lag':lag,
             'source_lags':list(source),'target_lags':list(target),'horizon':1}
            for direction in ('X_to_Y','Y_to_X') for kind,lag,source,target in specs]


def spec_for(job):
    return TemporalEmbeddingSpec(tuple(job['source_lags']),tuple(job['target_lags']),job['horizon'])


def config_for(job,seed):
    config=deepcopy(definition()['neural'])
    target='Y' if job['direction']=='X_to_Y' else 'X'
    config.update(training_seed=seed,component_order=list(spec_for(job).target_order(target)))
    return validate_neural_config(config)


def effective_streams(job,seed):
    spec=spec_for(job);source,target=('X','Y') if job['direction']=='X_to_Y' else ('Y','X')
    config=config_for(job,seed)
    alphabets={v:np.array([0,1]) for v in config.component_order+spec.past_order(source,source=True)+spec.past_order(target,source=False)}
    estimator=NeuralConditionalMutualInformationEstimator(config,y_order=spec.past_order(source,source=True),
                                                         z_order=spec.past_order(target,source=False),alphabets=alphabets)
    streams=[];components=[]
    for name,context,component in estimator._plans():
        streams.append(component.training_seed)
        tid=json.dumps(['njee-conditional-term-v1',component.component_order[0],list(context)],separators=(',',':'))
        seeds={p:derive_term_seed(component.training_seed,tid,purpose=p) for p in ('initialization','shuffle')}
        streams.extend(seeds.values())
        components.append({'name':name,'training_seed':component.training_seed,'target':component.component_order[0],
                           'conditioning':list(context),**seeds})
    return streams,components


@lru_cache(maxsize=1)
def prior_seeds():
    seeds=set(information_prior())
    for cell in information_preview()['cells']:
        seeds.add(cell['master_seed'])
        for replica in cell['allocation']:
            seeds.update(replica[k] for k in ('data_seed','split_seed','training_seed','permutation_seed','bootstrap_seed'))
            for component in replica['components']:
                seeds.add(component['training_seed'])
                for term in component['terms']:
                    seeds.update(term[p] for p in ('initialization','shuffle'))
    seeds.update((271816,271818))
    return frozenset(seeds)


def cells():
    d=definition();settings=[(0.,d['null_delay'])]+[(c,lag) for c in d['couplings'] for lag in d['delays']]
    result=[]
    for c,lag in settings:
        for n in d['lengths']:
            i=len(result)
            cell={'cell_id':f'T{i+1:03d}-c{round(c*100):03d}-delay{lag}-n{n:05d}',
                  'coupling':c,'delay':lag,'n':n,'replicas':d['replicas'],'master_seed':d['master_seed_start']+i,
                  'noise_x':d['noise_x'],'noise_y':d['noise_y'],'jobs':jobs(lag)}
            cell['configuration_hash']=digest(cell)
            result.append(cell)
    return result


def preview(output_root=None):
    d=definition();matrix=cells();reserved=set(prior_seeds());count=len(reserved)
    reserved.update(c['master_seed'] for c in matrix)
    for cell in matrix:
        allocation=[]
        def allocate(replica,role,job=None):
            for nonce in range(10000):
                master=d['bootstrap_master'] if role=='bootstrap' else cell['master_seed']
                payload=json.dumps([SCHEMA,master,cell['cell_id'],replica,role,nonce],separators=(',',':')).encode()
                seed=int.from_bytes(hashlib.sha256(payload).digest()[:4],'big')
                try:streams,components=effective_streams(job,seed) if job else ([],[])
                except ValueError:continue
                group=[seed,*streams]
                if len(set(group))!=len(group) or reserved.intersection(group):continue
                reserved.update(group)
                return {'seed':seed,'nonce':nonce,'components':components} if job else {'seed':seed,'nonce':nonce}
            raise ValueError('Seed namespace allocation exhausted.')
        cell['bootstrap']=allocate('cell','bootstrap')
        for i in range(d['replicas']):
            rid=f'replica-{i:06d}'
            allocation.append({'replica_id':rid,'data':allocate(rid,'data'),
                               'surrogates':{v:allocate(rid,'surrogate-'+v) for v in ('X_to_Y','Y_to_X')},
                               'jobs':{j['job_id']:allocate(rid,'training-'+j['job_id'],j) for j in cell['jobs']}})
        cell['allocation']=allocation
    order=[c['cell_id'] for c in sorted(matrix,key=lambda c:(c['coupling']!=0,c['n'],c['cell_id']))]
    scientific={'definition':{k:v for k,v in d.items() if k!='output_root'},'cells':matrix,'execution_order':order,
                'seed_policy':'sha256-uint32-global-nonce-collision-rejection-v1'}
    replicas=sum(c['replicas'] for c in matrix);fits=sum(c['replicas']*len(c['jobs']) for c in matrix)
    return {'schema':SCHEMA,'output_root':relative_path(d['output_root'] if output_root is None else output_root),
            **scientific,'protocol_hash':digest(scientific),
            'expected':{'cells':len(matrix),'process_replicas':replicas,'te_fits':fits,'conditional_fits':2*fits,
                        'primary_tests':replicas*d['family_size'],'surrogate_te_statistics':replicas*d['family_size']*d['surrogate_draws']},
            'seed_audit':{'prior_effective_values_reserved':count,'accepted_collisions':0}}


def materialize(directory,*,output_root=None):
    directory=Path(directory);plan=preview(output_root)
    root=Path(plan['output_root']).resolve();prep=directory.resolve()
    if root==prep or root in prep.parents or prep in root.parents:raise ValueError('Disjoint preparation/output paths required.')
    directory.mkdir(parents=True,exist_ok=False)
    for cell in plan['cells']:
        with (directory/(cell['cell_id']+'.yaml')).open('x',encoding='utf-8') as f:yaml.safe_dump(cell,f,sort_keys=True)
    with (directory/'protocol.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2,allow_nan=False)
    return plan


def validate_bundle(directory):
    directory=Path(directory);saved=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    expected=preview(saved['output_root'])
    if expected!=saved:raise ValueError('Frozen Experiment 3 manifest/hash mismatch.')
    if {p.name for p in directory.iterdir()}!={'protocol.json',*[c['cell_id']+'.yaml' for c in expected['cells']]}:
        raise ValueError('Unexpected/missing preparation files.')
    for cell in expected['cells']:
        if yaml.safe_load((directory/(cell['cell_id']+'.yaml')).read_text(encoding='utf-8'))!=cell:
            raise ValueError('Cell YAML differs from frozen protocol.')
    return expected
