"""Position-based causal embeddings and purged chronological categorical batches.

No temporal resampling, discretization, RNG or statistical independence claim.
Timestamps must already align exactly; lags count observations, not clock units.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from src.utils.categorical import CategoricalBatch, CategoricalDataset
from src.utils.split import _allocation


ROLES = ('train', 'validation', 'test')
ALGORITHM = 'chronological-leading-gap-conservative-purge-v1'


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive Python int.')


def _process(value):
    if type(value) is not str or not value.strip():
        raise ValueError('Process identities must be nonempty strings.')


def _readonly(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class TemporalEmbeddingSpec:
    """At reference r, predict target[p=r+h] using process[r+1-lag].

    h=1 gives the usual process[p-lag]. At h>1 the latest permitted
    observation is p-h; horizons do not admit intermediate future context.
    Supplied lag order is preserved exactly. Both nonempty blocks are required.
    """
    source_lags: tuple[int, ...] = (1,)
    target_lags: tuple[int, ...] = (1,)
    horizon: int = 1

    def __post_init__(self):
        _positive(self.horizon, 'horizon')
        for name in ('source_lags', 'target_lags'):
            lags = getattr(self, name)
            if not isinstance(lags, tuple) or not lags:
                raise ValueError('Lag sets must be nonempty tuples.')
            for lag in lags:
                _positive(lag, 'lag')
            if len(set(lags)) != len(lags):
                raise ValueError('Lag sets must not contain duplicates.')

    def target_order(self, target_id):
        _process(target_id)
        return (json.dumps(['temporal-v1',target_id,'future',self.horizon],separators=(',',':')),)

    def past_order(self, process_id, *, source):
        _process(process_id)
        return tuple(json.dumps(['temporal-v1',process_id,'past',lag,self.horizon],separators=(',',':'))
                     for lag in (self.source_lags if source else self.target_lags))


@dataclass(frozen=True, eq=False)
class TemporalEmbedding:
    dataset: CategoricalDataset
    spec: TemporalEmbeddingSpec
    source_id: str
    target_id: str
    timestamps: np.ndarray
    target_positions: np.ndarray
    observation_positions: np.ndarray
    # observation_positions columns exactly match dataset.variable_ids.

    def __post_init__(self):
        if not isinstance(self.dataset,CategoricalDataset) or not isinstance(self.spec,TemporalEmbeddingSpec):
            raise TypeError('Invalid temporal dataset/specification.')
        _process(self.source_id); _process(self.target_id)
        if self.source_id==self.target_id:
            raise ValueError('Temporal process identities must be distinct.')
        _times(self.timestamps,len(self.timestamps))
        spec=self.spec
        predicted=np.arange(max(spec.source_lags+spec.target_lags)+spec.horizon-1,len(self.timestamps))
        order=spec.target_order(self.target_id)+spec.past_order(self.source_id,source=True)+spec.past_order(self.target_id,source=False)
        expected=np.column_stack([predicted,*[predicted-spec.horizon+1-lag
                                               for lag in spec.source_lags+spec.target_lags]])
        if (not len(predicted) or self.dataset.variable_ids!=order or len(self.dataset.values)!=len(predicted)
                or not isinstance(self.target_positions,np.ndarray) or self.target_positions.dtype.kind not in 'iu'
                or not isinstance(self.observation_positions,np.ndarray) or self.observation_positions.dtype.kind not in 'iu'
                or not np.array_equal(predicted,self.target_positions)
                or not np.array_equal(expected,self.observation_positions)):
            raise ValueError('Temporal alignment/provenance does not match lag/horizon contract.')
        for name in ('timestamps','target_positions','observation_positions'):
            object.__setattr__(self,name,_readonly(getattr(self,name)))


def _times(times, n):
    if (np.ma.isMaskedArray(times) or not isinstance(times,np.ndarray) or times.ndim!=1
            or len(times)!=n or times.dtype.kind not in 'iufM'):
        raise ValueError('Timestamps must be aligned 1D numeric or datetime64 arrays.')
    if times.dtype.kind=='M':
        if np.isnat(times).any():
            raise ValueError('NaT timestamps are invalid.')
    elif not np.isfinite(times).all():
        raise ValueError('Timestamps must be finite.')
    if np.any(times[1:]<=times[:-1]):
        raise ValueError('Timestamps must be strictly increasing without duplicates.')
    return times


def build_temporal_embedding(source, target, *, spec=TemporalEmbeddingSpec(),
                             dataset_id, source_id='X', target_id='Y',
                             source_times=None, target_times=None):
    if not isinstance(spec,TemporalEmbeddingSpec):
        raise TypeError('spec must be TemporalEmbeddingSpec.')
    _process(source_id); _process(target_id)
    if source_id==target_id:
        raise ValueError('Source and target identities must be distinct.')
    for series in (source,target):
        if not isinstance(series,np.ndarray) or np.ma.isMaskedArray(series) or series.ndim!=1:
            raise ValueError('Source/target must be unmasked univariate ndarrays.')
        # Reuse numeric discrete-label validation, without learning any categories.
        CategoricalDataset(series,dataset_id,('series',))
    if len(source)!=len(target):
        raise ValueError('Series lengths must match; no automatic alignment/truncation.')
    n = len(source)
    common=np.result_type(source.dtype,target.dtype)
    # NumPy may promote mixed signed/unsigned labels to an inexact float dtype.
    for series in (source,target):
        if any(int(a)!=int(b) for a,b in zip(series,series.astype(common))):
            raise ValueError('Combining series would lose categorical label identities.')
    if (source_times is None)!=(target_times is None):
        raise ValueError('Supply both timestamp arrays or neither.')
    if source_times is None:
        times=np.arange(n,dtype=np.int64)
    else:
        times=_times(source_times,n)
        _times(target_times,n)
        if source_times.dtype!=target_times.dtype or not np.array_equal(source_times,target_times):
            raise ValueError('Source and target timestamps must match exactly, including dtype.')
    first=max(spec.source_lags+spec.target_lags)+spec.horizon-1
    if first>=n:
        raise ValueError('Insufficient sequence length for lag/horizon specification.')
    predicted=np.arange(first,n,dtype=np.int64)
    source_positions=[predicted-spec.horizon+1-lag for lag in spec.source_lags]
    target_positions=[predicted-spec.horizon+1-lag for lag in spec.target_lags]
    order=spec.target_order(target_id)+spec.past_order(source_id,source=True)+spec.past_order(target_id,source=False)
    positions=np.column_stack([predicted,*source_positions,*target_positions])
    values=np.column_stack([target[predicted],*[source[p] for p in source_positions],
                            *[target[p] for p in target_positions]])
    return TemporalEmbedding(CategoricalDataset(values,dataset_id,order),spec,source_id,target_id,
                             _readonly(times),_readonly(predicted),_readonly(positions))


@dataclass(frozen=True)
class TemporalPartitionProvenance:
    role: str
    compact_row_ids: tuple[int, ...]
    embedding_rows: tuple[int, ...]
    target_positions: tuple[int, ...]
    target_timestamps: tuple
    observation_positions: tuple[tuple[int, ...], ...]
    observation_timestamps: tuple[tuple, ...]


@dataclass(frozen=True)
class ChronologicalPartitions:
    """Compact categorical rows cover retained data; provenance retains raw times."""
    train: CategoricalBatch
    validation: CategoricalBatch
    test: CategoricalBatch
    provenance: tuple[TemporalPartitionProvenance, ...]
    dropped_embedding_rows: tuple[int, ...]
    fractions: tuple[float, float, float]
    gap: int
    purge: bool
    algorithm: str = ALGORITHM


def make_chronological_split(embedding, *, fractions=(.6,.2,.2), gap=0, purge=True):
    """Nominal largest-remainder blocks, then remove leading rows from val/test.

    Drop `gap` embedded rows at each nominal boundary (not elapsed clock time).
    With purge=True also drop until min(context positions)>previous retained
    block's final prediction position. Thus conservative closed raw-time window
    hulls cannot overlap, even for sparse lags. Never move a row to another block
    or replenish a dropped row. Without purge, causal histories can cross the
    boundary and overlap earlier target times; all prediction times stay disjoint.
    """
    if not isinstance(embedding,TemporalEmbedding):
        raise TypeError('embedding must be TemporalEmbedding.')
    if type(gap) is not int or gap<0 or type(purge) is not bool:
        raise ValueError('gap must be a nonnegative Python int; purge must be bool.')
    n=len(embedding.target_positions)
    counts=_allocation(n,fractions,0)  # Validation/rounding only; no RNG or IID split.
    edges=np.cumsum((0,*counts))
    blocks=[]
    for i in range(3):
        rows=np.arange(edges[i],edges[i+1],dtype=np.int64)
        if i:
            rows=rows[gap:]
            if purge:
                end=embedding.target_positions[blocks[-1][-1]]
                rows=rows[embedding.observation_positions[rows,1:].min(axis=1)>end]
        if not len(rows):
            raise ValueError('Insufficient data: gap/purge produces an empty temporal partition.')
        blocks.append(rows)
    retained=np.concatenate(blocks)
    policy={'version':ALGORITHM,'source':embedding.source_id,'target':embedding.target_id,
            'source_lags':embedding.spec.source_lags,'target_lags':embedding.spec.target_lags,
            'horizon':embedding.spec.horizon,'fractions':fractions,'gap':gap,'purge':purge,
            'retained':retained.tolist(), 'timestamps_dtype':embedding.timestamps.dtype.str,
            'timestamps_sha256':hashlib.sha256(embedding.timestamps.tobytes()).hexdigest()}
    digest=hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dataset_id=f'{embedding.dataset.dataset_id}:temporal:{digest}'
    split_id=f'{ALGORITHM}:{digest}'
    batches=[];provenance=[];offset=0
    for role,rows in zip(ROLES,blocks):
        compact=np.arange(offset,offset+len(rows),dtype=np.int64);offset+=len(rows)
        batches.append(CategoricalBatch(embedding.dataset.values[rows],dataset_id,embedding.dataset.variable_ids,
                                        compact,role,split_id,len(retained)))
        positions=embedding.observation_positions[rows]
        targets=embedding.target_positions[rows]
        provenance.append(TemporalPartitionProvenance(
            role,tuple(map(int,compact)),tuple(map(int,rows)),tuple(map(int,targets)),
            tuple(embedding.timestamps[targets]),tuple(tuple(map(int,p)) for p in positions),
            tuple(tuple(embedding.timestamps[p]) for p in positions)))
    dropped=tuple(map(int,np.setdiff1d(np.arange(n),retained)))
    return ChronologicalPartitions(*batches,tuple(provenance),dropped,fractions,gap,purge)
