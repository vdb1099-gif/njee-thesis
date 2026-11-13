"""Exact time alignment, chronology and compact-row compatibility."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from src.utils.temporal import TemporalEmbeddingSpec, build_temporal_embedding, make_chronological_split


def embedding(n=40,spec=TemporalEmbeddingSpec()):
    return build_temporal_embedding(np.arange(n),100+np.arange(n),spec=spec,dataset_id='data')


def test_exact_order_alignment_and_horizon():
    spec=TemporalEmbeddingSpec((3,1),(2,1),2)
    e=embedding(9,spec)
    assert len(e.dataset.values)==9-3-2+1==5
    np.testing.assert_array_equal(e.target_positions,np.arange(4,9))
    np.testing.assert_array_equal(e.observation_positions[0],[4,0,2,1,2])
    np.testing.assert_array_equal(e.dataset.values[0],[104,0,2,101,102])
    assert e.dataset.variable_ids==(spec.target_order('Y')+spec.past_order('X',source=True)+spec.past_order('Y',source=False))
    assert np.all(e.observation_positions[:,1:]<=e.target_positions[:,None]-2)
    with pytest.raises(FrozenInstanceError):spec.horizon=3
    assert not e.dataset.values.flags.writeable and not e.observation_positions.flags.writeable


def test_irregular_and_datetime_timestamps_are_position_lags():
    source=np.arange(6);target=source+10
    times=np.array([2,4,8,16,32,64])
    e=build_temporal_embedding(source,target,dataset_id='d',source_times=times,target_times=times)
    np.testing.assert_array_equal(e.timestamps[e.observation_positions[0]],[4,2,2])
    dates=np.arange('2020-01-01','2020-01-07',dtype='datetime64[D]')
    e=build_temporal_embedding(source,target,dataset_id='d',source_times=dates,target_times=dates)
    assert e.timestamps[e.target_positions[0]]==dates[1]
    np.testing.assert_array_equal(source,np.arange(6))


@pytest.mark.parametrize('source,target,horizon',[((),(1,),1),([1],(1,),1),((0,),(1,),1),
                                              ((True,),(1,),1),((1,1),(1,),1),((1,),(1,),0),
                                              ((1,),(1,),True),((1,),(1,),1.5)])
def test_invalid_lags_and_horizon(source,target,horizon):
    with pytest.raises(ValueError):TemporalEmbeddingSpec(source,target,horizon)


@pytest.mark.parametrize('change',['length','shape','nan','fractional','masked','object','empty','short'])
def test_invalid_series(change):
    s=np.arange(5);t=np.arange(5)
    if change=='length':t=t[:-1]
    if change=='shape':s=s[:,None]
    if change=='nan':s=np.array([0,1,2,3,np.nan])
    if change=='fractional':s=np.array([0,1,2,3,4.5])
    if change=='masked':s=np.ma.array(s)
    if change=='object':s=s.astype(object)
    if change=='empty':s=t=np.array([],dtype=int)
    if change=='short':s=t=np.array([0])
    with pytest.raises((ValueError,TypeError)):build_temporal_embedding(s,t,dataset_id='d')


@pytest.mark.parametrize('a,b',[(np.array([0,1,2,3,4]),None),
                              (np.array([0,1,2,3,4]),np.array([0,1,2,3,5])),
                              (np.array([0,1,1,3,4]),np.array([0,1,1,3,4])),
                              (np.array([4,3,2,1,0]),np.array([4,3,2,1,0])),
                              (np.array([0,1,2,3,np.nan]),np.array([0,1,2,3,np.nan]))])
def test_inconsistent_timestamps(a,b):
    with pytest.raises(ValueError):build_temporal_embedding(np.arange(5),np.arange(5),dataset_id='d',source_times=a,target_times=b)


def test_category_identity_not_lost_by_numpy_promotion():
    a=np.array([2**63+1]*5,dtype=np.uint64);b=np.array([-1]*5,dtype=np.int64)
    with pytest.raises(ValueError,match='identities'):build_temporal_embedding(a,b,dataset_id='d')
    a=np.array([2**63+1]*5,dtype=np.uint64)
    e=build_temporal_embedding(a,a,dataset_id='d')
    assert int(e.dataset.values[0,0])==2**63+1


def test_chronological_no_purge_causal_context_can_cross_boundary():
    e=embedding(11)
    parts=make_chronological_split(e,purge=False)
    assert [len(getattr(parts,r).values) for r in ('train','validation','test')]==[6,2,2]
    assert [p.target_positions for p in parts.provenance]==[tuple(range(1,7)),(7,8),(9,10)]
    assert parts.provenance[1].observation_positions[0]==(7,6,6)
    assert parts.dropped_embedding_rows==()
    for left,right in zip(parts.provenance,parts.provenance[1:]):
        assert left.target_positions[-1]<right.target_positions[0]


@pytest.mark.parametrize('gap,purge',[(0,True),(2,True),(4,False)])
def test_purge_gap_metadata_and_exact_compact_coverage(gap,purge):
    e=embedding(41,TemporalEmbeddingSpec((3,1),(2,),2))
    p=make_chronological_split(e,gap=gap,purge=purge)
    batches=[getattr(p,r) for r in ('train','validation','test')]
    combined=np.concatenate([b.row_indices for b in batches])
    np.testing.assert_array_equal(combined,np.arange(len(combined)))
    assert all(b.n_samples==len(combined) for b in batches)
    assert len(set(b.dataset_id for b in batches))==len(set(b.split_id for b in batches))==1
    kept=[row for part in p.provenance for row in part.embedding_rows]
    assert sorted(kept+list(p.dropped_embedding_rows))==list(range(len(e.dataset.values)))
    for b,prov in zip(batches,p.provenance):
        np.testing.assert_array_equal(b.values,e.dataset.values[list(prov.embedding_rows)])
        assert prov.target_timestamps==prov.target_positions  # default timestamps are positions
    if purge:
        for a,b in zip(p.provenance,p.provenance[1:]):
            assert min(min(row) for row in b.observation_positions)>a.target_positions[-1]
    assert make_chronological_split(e,gap=gap,purge=purge).train.split_id==p.train.split_id
    with pytest.raises(FrozenInstanceError):p.gap=3


def test_gap_purge_are_leading_removal_without_reallocation():
    e=embedding(21) # m=20 -> nominal 12/4/4
    p=make_chronological_split(e,gap=2,purge=True)
    assert [len(getattr(p,r).values) for r in ('train','validation','test')]==[12,2,2]
    assert p.dropped_embedding_rows==(12,13,16,17)
    assert p.provenance[1].target_positions==(15,16)
    assert p.provenance[2].target_positions==(19,20)
    assert p.train.split_id!=make_chronological_split(e,gap=2,purge=False).train.split_id


@pytest.mark.parametrize('kwargs',[{'gap':True},{'gap':-1},{'gap':1.5},{'purge':1},
                                    {'fractions':(.5,.5,0.)},{'gap':100}])
def test_invalid_split_settings(kwargs):
    with pytest.raises(ValueError):make_chronological_split(embedding(),**kwargs)


def test_purge_empty_and_provenance_tampering():
    with pytest.raises(ValueError,match='empty'):make_chronological_split(embedding(5))
    e=embedding()
    bad=e.observation_positions.copy();bad[0,1]=1 # contemporaneous source
    with pytest.raises(ValueError,match='alignment'):replace(e,observation_positions=bad)
    with pytest.raises(ValueError):replace(e,target_positions=e.target_positions[::-1])
