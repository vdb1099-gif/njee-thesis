"""Directional neural TE by composition with the existing CMI estimator."""
from dataclasses import dataclass
import math

import numpy as np

from src.estimators.neural_information import NeuralConditionalMutualInformationEstimator, NeuralInformationReport
from src.utils.neural_config import NeuralConfig
from src.utils.temporal import (TemporalEmbeddingSpec, TemporalPartitionProvenance,
                                build_temporal_embedding, make_chronological_split)


@dataclass(frozen=True)
class NeuralTransferEntropyReport:
    transfer_entropy_nats: float
    direction: tuple[str, str]
    embedding: TemporalEmbeddingSpec
    variable_ids: tuple[str, ...]
    partitions: tuple[TemporalPartitionProvenance, ...]
    dropped_embedding_rows: tuple[int, ...]
    raw_sequence_length: int
    split_algorithm: str
    split_fractions: tuple[float, float, float]
    gap: int
    purge: bool
    cmi_report: NeuralInformationReport


class NeuralTransferEntropyEstimator:
    """Fit chronological TRAIN/VALIDATION; estimate() evaluates stored TEST only.

    config.component_order MUST equal embedding.target_order(target_id).
    CMI X is the future target, Y is source past, Z is target past. Consequently
    existing CMI feature order in the second component is source-past then
    target-past; this is H(Y_future|X_past,Y_past), equal to the TE identity.
    Ordered lags are never silently sorted or symmetrized. split_seed is unused
    compatibility metadata: temporal partitioning has no RNG. Distinct physical
    process IDs ensure explicit reversal also separates neural seed namespaces.
    Only declared alphabets are supported, supplied before held-out inspection.
    Chronological/purged partitions are not statistically independent by default.
    Observational TE does not establish structural causality. No IID confidence
    intervals or IID permutation tests are supplied for dependent temporal rows.
    """
    def __init__(self, config: NeuralConfig, *, embedding=TemporalEmbeddingSpec(),
                 source_id='X', target_id='Y', source_alphabet, target_alphabet,
                 gap=0, purge=True):
        if not isinstance(config,NeuralConfig) or not isinstance(embedding,TemporalEmbeddingSpec):
            raise TypeError('Validated NeuralConfig and TemporalEmbeddingSpec required.')
        target_order=embedding.target_order(target_id)
        embedding.past_order(source_id,source=True)
        if source_id==target_id:
            raise ValueError('Source and target identities must be distinct.')
        if config.component_order!=target_order or config.alphabet_policy!='declared':
            raise ValueError('Require declared policy and component_order=embedding.target_order(target_id).')
        if type(gap) is not int or gap<0 or type(purge) is not bool:
            raise ValueError('Invalid gap or purge policy.')
        # Existing encoder performs complete support/type/duplicate validation.
        if any(not isinstance(a,np.ndarray) for a in (source_alphabet,target_alphabet)):
            raise ValueError('Declared alphabets must be NumPy arrays.')
        self._config,self._embedding=config,embedding
        self._source_id,self._target_id=source_id,target_id
        self._source_alphabet,self._target_alphabet=source_alphabet.copy(),target_alphabet.copy()
        self._gap,self._purge=gap,purge
        self._fitted=False

    @property
    def config(self):
        return self._config

    def fit(self, source, target, *, dataset_id, source_times=None, target_times=None):
        if self._fitted:
            raise RuntimeError('Estimator is already fitted.')
        embedded=build_temporal_embedding(source,target,spec=self._embedding,dataset_id=dataset_id,
                                         source_id=self._source_id,target_id=self._target_id,
                                         source_times=source_times,target_times=target_times)
        parts=make_chronological_split(embedded,fractions=self.config.split_fractions,
                                       gap=self._gap,purge=self._purge)
        source_order=self._embedding.past_order(self._source_id,source=True)
        target_past=self._embedding.past_order(self._target_id,source=False)
        alphabets={v:self._source_alphabet for v in source_order}
        alphabets.update({v:self._target_alphabet for v in self.config.component_order+target_past})
        model=NeuralConditionalMutualInformationEstimator(
            self.config,y_order=source_order,z_order=target_past,alphabets=alphabets)
        model.fit(parts.train,parts.validation)
        # Nothing is published if either CMI component fails.
        self._model,self._parts=model,parts
        self._variable_ids=embedded.dataset.variable_ids
        self._raw_length=len(embedded.timestamps)
        self._fitted=True
        return self

    def estimate(self):
        if not self._fitted:
            raise RuntimeError('Estimator must be fitted before estimate.')
        report=self._model.estimate(self._parts.test)
        if not math.isfinite(report.information_nats):
            raise RuntimeError('Nonfinite transfer entropy.')
        return NeuralTransferEntropyReport(
            report.information_nats,(self._source_id,self._target_id),self._embedding,
            self._variable_ids,self._parts.provenance,self._parts.dropped_embedding_rows,
            self._raw_length,self._parts.algorithm,self._parts.fractions,self._parts.gap,
            self._parts.purge,report)
