"""Roadmap coupled temporal process and finite-history stationary TE references."""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.sparse import csr_matrix

from src.utils.reproducibility import validate_seed
from src.utils.temporal import TemporalEmbeddingSpec


@dataclass(frozen=True)
class CoupledBinaryModel:
    coupling: float
    delay: int
    noise_x: float = .1
    noise_y: float = .1

    def __post_init__(self):
        if type(self.delay) is not int or self.delay<1:
            raise ValueError('delay must be a positive Python int.')
        for name, lower in (('coupling',0.),('noise_x',0.),('noise_y',0.)):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError('Process probabilities must be finite numbers.')
            if name=='coupling':
                if not 0<=value<=1:raise ValueError('coupling must lie in [0,1].')
            elif not lower<value<=.5:
                raise ValueError('Noise must lie in (0,.5] for the ergodic reference contract.')

    def generate(self,n,*,seed,burn_in=1000,initial='random'):
        """X history at times 1-delay..0, Y_0; transition t>=1 uses X_{t-delay}.

        Random initialization draws independent fair bits, not a stationary joint
        history. Burn-in discards times 0..burn_in-1; returned position j is
        physical time burn_in+j. All switches/noises are independently sampled.
        """
        if type(n) is not int or n<1 or type(burn_in) is not int or burn_in<0:
            raise ValueError('Require positive n and nonnegative integer burn_in.')
        if initial not in ('random','zeros'):
            raise ValueError('Initial policy must be random or zeros.')
        validate_seed(seed)
        rng=np.random.default_rng(seed)
        history=rng.binomial(1,.5,self.delay) if initial=='random' else np.zeros(self.delay,dtype=int)
        y0=int(rng.binomial(1,.5)) if initial=='random' else 0
        steps=burn_in+n-1
        ex=rng.binomial(1,self.noise_x,steps)
        ey=rng.binomial(1,self.noise_y,steps)
        switches=rng.binomial(1,self.coupling,steps)
        x=np.empty(self.delay+steps,dtype=np.int64);x[:self.delay]=history
        y=np.empty(steps+1,dtype=np.int64);y[0]=y0
        for t in range(1,steps+1):
            x[self.delay-1+t]=x[self.delay-2+t]^ex[t-1]
            parent=x[t-1] if switches[t-1] else y[t-1]
            y[t]=parent^ey[t-1]
        a=x[self.delay-1+burn_in:self.delay-1+burn_in+n].copy()
        b=y[burn_in:burn_in+n].copy()
        a.flags.writeable=b.flags.writeable=False
        return a,b,{'seed':seed,'burn_in':burn_in,'initial_policy':initial,
                    'initial_x_history':tuple(map(int,history)),'initial_y':y0,
                    'physical_time_start':burn_in,'physical_time_stop':burn_in+n-1,
                    'process':'roadmap-coupled-binary-v1','rng':'PCG64'}


def transition_matrix(model,memory):
    """State stores M pairs oldest to newest; pair digit is 2*X+Y."""
    if type(memory) is not int or not model.delay<=memory<=8:
        raise ValueError('Reference requires delay<=memory<=8 (explicit state-space cap).')
    states=np.arange(4**memory,dtype=np.int64)
    xprev=(states>>1)&1;yprev=states&1
    delayed=(states>>(2*(model.delay-1)+1))&1
    px=model.noise_x+(1-2*model.noise_x)*xprev
    py=model.noise_y+(1-2*model.noise_y)*(model.coupling*delayed+(1-model.coupling)*yprev)
    rows=[];cols=[];prob=[]
    for x in (0,1):
        for y in (0,1):
            rows.extend(states);cols.extend(((states<<2)&(4**memory-1))|(2*x+y))
            prob.extend((px if x else 1-px)*(py if y else 1-py))
    return csr_matrix((prob,(rows,cols)),shape=(len(states),len(states)))


def coupling_bound(model,steps,memory):
    """Union bound on TV distance of M-pair histories from stationarity.

    Maximal conditional Bernoulli coupling gives u_t=a*u_{t-1} for X,
    v_t<=b*((1-c)*v_{t-1}+c*u_{t-delay}) for Y. Initial disagreement bounds
    are 1. Sum the last M bounds. This does NOT include floating-point error.
    """
    u=[1.];v=[1.]
    a=abs(1-2*model.noise_x);b=abs(1-2*model.noise_y)
    for t in range(1,steps+1):
        u.append(a*u[-1])
        v.append(b*((1-model.coupling)*v[-1]+model.coupling*u[max(0,t-model.delay)]))
    return min(1.,sum(u[max(0,steps-j)]+v[max(0,steps-j)] for j in range(memory)))


@lru_cache(maxsize=32)
def stationary_reference(model,memory,tolerance=1e-13,max_iterations=10000):
    if type(tolerance) not in (int,float) or not 0<tolerance<1 or type(max_iterations) is not int or max_iterations<1:
        raise ValueError('Invalid stationary solver controls.')
    matrix=transition_matrix(model,memory)
    distribution=np.full(matrix.shape[0],1/matrix.shape[0])
    u=[1.];v=[1.];bound=1.
    for t in range(1,max_iterations+1):
        distribution=matrix.T@distribution
        distribution/=distribution.sum()
        u.append(abs(1-2*model.noise_x)*u[-1])
        v.append(abs(1-2*model.noise_y)*((1-model.coupling)*v[-1]+model.coupling*u[max(0,t-model.delay)]))
        bound=min(1.,sum(u[max(0,t-j)]+v[max(0,t-j)] for j in range(memory)))
        if bound<=tolerance:break
    else:
        raise RuntimeError('Stationary reference did not reach its declared coupling bound.')
    residual=float(np.abs(matrix.T@distribution-distribution).sum())
    if not np.isfinite(distribution).all() or np.min(distribution)<0 or residual>1e-10:
        raise RuntimeError('Invalid numerical stationary reference.')
    distribution.flags.writeable=False
    return distribution,{'status':'numerical_stationary_enumeration','version':'pair-chain-power-coupling-v1',
                         'memory':memory,'n_states':len(distribution),'iterations':t,
                         'tv_truncation_bound':bound,'stationarity_l1_residual':residual,
                         'floating_point':'float64; analytic truncation bound excludes roundoff'}


def reference_te(model,spec,*,direction='X_to_Y',tolerance=1e-13):
    """Population finite-embedding TE, NOT full-history TE or a neural estimate."""
    if not isinstance(spec,TemporalEmbeddingSpec) or spec.horizon!=1 or direction not in ('X_to_Y','Y_to_X'):
        raise ValueError('Reference supports explicit directions and horizon=1 only.')
    m=max(model.delay,*spec.source_lags,*spec.target_lags)
    pi,metadata=stationary_reference(model,m,tolerance)
    states=np.arange(len(pi));history=np.zeros(len(pi),dtype=np.int64)
    forward=direction=='X_to_Y'
    for lag in spec.source_lags:
        history=(history<<1)|((states>>(2*(lag-1)+(1 if forward else 0)))&1)
    for lag in spec.target_lags:
        history=(history<<1)|((states>>(2*(lag-1)+(0 if forward else 1)))&1)
    if forward:
        delayed=(states>>(2*(model.delay-1)+1))&1
        p=model.noise_y+(1-2*model.noise_y)*(model.coupling*delayed+(1-model.coupling)*(states&1))
    else:
        p=model.noise_x+(1-2*model.noise_x)*((states>>1)&1)
    bits=len(spec.source_lags)+len(spec.target_lags)
    joint=np.bincount(history,weights=pi*(1-p),minlength=2**(bits+1))
    joint+=np.bincount((1<<bits)|history,weights=pi*p,minlength=len(joint))
    indices=np.arange(len(joint));k=len(spec.target_lags);mask=(1<<k)-1
    def entropy(codes):
        masses=np.bincount(codes,weights=joint)
        return -math.fsum(float(q)*math.log(float(q)) for q in masses if q>0)
    a=entropy(((indices>>bits)<<k)|(indices&mask))-entropy(indices&mask)
    b=entropy(indices)-entropy(indices&((1<<bits)-1))
    value=a-b
    analytic_zero=model.coupling==0 or (not forward and 1 in spec.target_lags) or model.noise_y==.5
    delta=metadata['tv_truncation_bound']
    continuity=4*(delta*math.log(max(2,len(joint))-1)
                  -delta*math.log(delta) -(1-delta)*math.log1p(-delta)) if delta else 0.
    if analytic_zero:
        if abs(value)>continuity+1e-12:raise RuntimeError('Enumeration contradicts analytic zero.')
        b=a;value=0.
    return {**metadata,'status':'analytic_zero' if analytic_zero else metadata['status'],
            'transfer_entropy_nats':value,'first_entropy_nats':a,'second_entropy_nats':b,
            'direction':direction,'source_lags':spec.source_lags,'target_lags':spec.target_lags,
            'horizon':1,'entropy_truncation_error_bound_nats':continuity,
            'reference_scope':'stationary population under exactly the stated finite histories'}
