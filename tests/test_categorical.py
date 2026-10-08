"""Categorical identity, exact-label encoding and adversarial leakage tests."""

from dataclasses import replace

import numpy as np
import pytest

from src.utils.categorical import (
    CategoricalBatch, CategoricalDataset, CategoricalEncoder, validate_variable_order,
)
from src.utils.split import make_iid_split


def partitions(values, ids=None):
    if ids is None:
        ids = tuple(range(1 if values.ndim == 1 else values.shape[1]))
    dataset = CategoricalDataset(values, "dataset", ids)
    split = make_iid_split(len(values), seed=42)
    return dataset, split, {role: dataset.batch(split, role, split_id="split")
                            for role in ("train", "validation", "test")}


def test_partition_metadata_and_non_mutation():
    source = np.arange(40).reshape(10, 4)[:, ::2]
    before = source.copy()
    source.flags.writeable = False
    dataset, split, batches = partitions(source, ("X", "Y"))
    for role, batch in batches.items():
        np.testing.assert_array_equal(batch.row_indices, getattr(split, role))
        np.testing.assert_array_equal(batch.values, source[batch.row_indices])
        assert batch.dataset_id == "dataset" and batch.split_id == "split"
        assert batch.role == role and batch.variable_ids == ("X", "Y")
        assert batch.n_samples == 10
        assert not batch.values.flags.writeable and not batch.row_indices.flags.writeable
    encoder = CategoricalEncoder.fit(batches["train"])
    encoder.target_indices(batches["train"], "X")
    encoder.conditioning_features(batches["train"], ("Y",), target="X")
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(dataset.values, before)


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int64, np.uint64, np.float64])
def test_one_dimensional_dtype_and_round_trip(dtype):
    source = np.tile([0, 1], 5).astype(dtype)
    dataset, _, batches = partitions(source)
    assert dataset.values.shape == (10, 1) and dataset.values.dtype == source.dtype
    encoder = CategoricalEncoder.fit(batches["train"], policy="declared",
                                    alphabets={0: np.array([1, 0], dtype=dtype)})
    for batch in batches.values():
        indices = encoder.target_indices(batch, 0)
        assert indices.shape == (len(batch.values),) and indices.dtype == np.int64
        decoded = encoder.inverse_target(indices, 0)
        np.testing.assert_array_equal(decoded, batch.values[:, 0])
        assert decoded.dtype == dtype
        assert encoder.conditioning_features(batch, (), target=0).shape == (len(indices), 0)


@pytest.mark.parametrize("source", [np.array([-19, 4, 42] * 4),
    np.array([-(2**63), 2**53, 2**53 + 1, 2**63 - 1] * 3, dtype=np.int64),
    np.array([0, 2**64 - 2, 2**64 - 1] * 4, dtype=np.uint64),
    np.array([-4., 0., 7.] * 4)])
def test_exact_labels(source):
    _, _, batches = partitions(source)
    encoder = CategoricalEncoder.fit(batches["train"], policy="declared",
                                    alphabets={0: np.unique(source)})
    labels = encoder.categories[0]
    assert labels.dtype == source.dtype
    assert len(encoder.mappings[0]) == len(np.unique(source))
    for batch in batches.values():
        np.testing.assert_array_equal(encoder.inverse_target(encoder.target_indices(batch, 0), 0),
                                      batch.values[:, 0])


@pytest.mark.parametrize("values", [np.array([]), np.empty((0, 2)), np.empty((2, 0)),
    np.array(1), np.zeros((2, 2, 2)), np.array([0.5]), np.array([np.nan]),
    np.array([np.inf]), np.array([None]), np.array(["a"]), np.array([1j]),
    np.array([0, 1], dtype=object), np.ma.array([0, 1], mask=False), [0, 1]])
def test_invalid_values(values):
    with pytest.raises((ValueError, TypeError)):
        CategoricalDataset(values, "dataset", (0,))


@pytest.mark.parametrize("ids", [(0,), (0, 0), (True, 1), ("", "Y"), [0, 1], (None, 1)])
def test_invalid_column_identities(ids):
    with pytest.raises(ValueError):
        CategoricalDataset(np.zeros((5, 2), dtype=int), "dataset", ids)


@pytest.mark.parametrize("identity", ["", "  ", None, 42])
def test_invalid_dataset_identity(identity):
    with pytest.raises(ValueError):
        CategoricalDataset(np.zeros(5, dtype=int), identity, (0,))


@pytest.mark.parametrize("change", [
    {"split_id": ""}, {"role": "other"}, {"n_samples": True}, {"n_samples": 0},
    {"row_indices": np.array([0, 0, 2])}, {"row_indices": np.array([-1, 1, 2])},
    {"row_indices": np.array([0, 1, 5])}, {"row_indices": np.array([0, 1])},
    {"row_indices": np.array([0., 1., 2.])}, {"row_indices": np.array([[0, 1, 2]])},
    {"row_indices": np.ma.array([0, 1, 2], mask=False)},
])
def test_invalid_batch_metadata(change):
    batch = CategoricalBatch(np.zeros(3, dtype=int), "dataset", (0,),
                             np.arange(3), "train", "split", 5)
    with pytest.raises(ValueError):
        replace(batch, **change)


def test_split_count_role_and_integrity():
    dataset, split, _ = partitions(np.zeros(10, dtype=int))
    with pytest.raises(ValueError, match="count"):
        dataset.batch(make_iid_split(5, seed=0), "train", split_id="s")
    with pytest.raises(ValueError, match="role"):
        dataset.batch(split, "unknown", split_id="s")
    with pytest.raises(TypeError):
        dataset.batch(None, "train", split_id="s")
    split.train.flags.writeable = True
    split.train[0] = split.test[0]
    with pytest.raises(ValueError):
        dataset.batch(split, "train", split_id="s")


def test_copies_own_values_and_rows():
    source = np.zeros(5, dtype=int)
    dataset, _, batches = partitions(source)
    source[:] = 8
    assert not np.any(dataset.values)
    values, rows = np.zeros(3, dtype=int), np.arange(3)
    batch = CategoricalBatch(values, "dataset", (0,), rows, "train", "split", 5)
    values[:] = 9
    rows[:] = 4
    assert not np.any(batch.values)
    np.testing.assert_array_equal(batch.row_indices, np.arange(3))


def test_declared_binary_constant_train_retains_both_classes():
    source = np.zeros(10, dtype=int)
    split = make_iid_split(10, seed=42)
    source[split.validation] = 1
    source[split.test] = 1
    _, _, batches = partitions(source)
    encoder = CategoricalEncoder.fit(batches["train"], policy="declared",
                                    alphabets={0: np.array([0, 1])})
    assert encoder.cardinalities[0] == 2
    np.testing.assert_array_equal(encoder.target_indices(batches["test"], 0), [1, 1])
    observed = CategoricalEncoder.fit(batches["train"])
    assert observed.cardinalities[0] == 1
    for role in ("validation", "test"):
        with pytest.raises(ValueError, match=f"category 1.*variable 0.*{role}"):
            observed.target_indices(batches[role], 0)
    assert observed.cardinalities[0] == 1


@pytest.mark.parametrize("role", ["validation", "test"])
def test_fit_rejects_nontraining_roles(role):
    _, _, batches = partitions(np.zeros(10, dtype=int))
    with pytest.raises(ValueError, match="train role"):
        CategoricalEncoder.fit(batches[role])


@pytest.mark.parametrize("changed_role", ["validation", "test"])
def test_holdout_changes_never_change_mapping(changed_role, monkeypatch):
    source = np.zeros((10, 2), dtype=np.int64)
    split = make_iid_split(10, seed=42)
    source[split.train, 0] = [-7, 20, -7, 20, -7, 20]
    _, _, original = partitions(source, ("target", "context"))
    first = CategoricalEncoder.fit(original["train"])
    source[getattr(split, changed_role)] = 2**53 + 1
    _, _, changed = partitions(source, ("target", "context"))
    # Fitting may inspect unique values only in columns of the training batch.
    original_unique = np.unique
    inspected = []

    def checked_unique(values, *args, **kwargs):
        inspected.append(values.copy())
        assert not np.any(values == 2**53 + 1)
        return original_unique(values, *args, **kwargs)

    monkeypatch.setattr(np, "unique", checked_unique)
    second = CategoricalEncoder.fit(changed["train"])
    assert len(inspected) == 2
    assert first.mappings == second.mappings
    assert first.cardinalities == second.cardinalities
    with pytest.raises(ValueError, match=changed_role):
        second.target_indices(changed[changed_role], "target")
    assert second.cardinalities["target"] == 2


def test_explicit_conditioning_order_excludes_target_and_metadata():
    source = np.tile([[0, -5, 7], [1, 9, 2]], (5, 1))
    _, _, batches = partitions(source, ("target", "left", "right"))
    encoder = CategoricalEncoder.fit(batches["train"], policy="declared", alphabets={
        "target": np.array([0, 1]), "left": np.array([-5, 9]), "right": np.array([2, 7]),
    })
    batch = batches["test"]
    forward = encoder.conditioning_features(batch, ("left", "right"), target="target")
    reversed_features = encoder.conditioning_features(batch, ("right", "left"), target="target")
    assert forward.shape == (len(batch.values), 4) and forward.dtype == np.float32
    np.testing.assert_array_equal(reversed_features, forward[:, [2, 3, 0, 1]])
    for i, raw in enumerate(batch.values):
        np.testing.assert_array_equal(forward[i], [raw[1] == -5, raw[1] == 9,
                                                  raw[2] == 2, raw[2] == 7])
    np.testing.assert_array_equal(forward.sum(axis=1), np.full(len(batch.values), 2))
    with pytest.raises(ValueError, match="excluded"):
        encoder.conditioning_features(batch, ("target",), target="target")
    for order in (("left", "left"), ("unknown",), ["left"], (True,)):
        with pytest.raises(ValueError):
            encoder.conditioning_features(batch, order, target="target")
    # String identities, row IDs and roles have no feature blocks.
    assert encoder.variable_ids == ("target", "left", "right")


def test_target_group_permutations_preserve_ids_and_allow_context_group():
    targets = ("X", "Z")  # A separate Y context can coexist in the dataset.
    assert validate_variable_order(("Z", "X"), targets) == ("Z", "X")
    for invalid in (("X",), ("X", "X"), ("X", "Y"), (), ["X", "Z"], (True, "Z")):
        with pytest.raises(ValueError):
            validate_variable_order(invalid, targets)


@pytest.mark.parametrize("kwargs", [{"policy": "bad"}, {"policy": "declared"},
    {"alphabets": {0: np.array([0])}}, {"max_cardinality": 0}, {"max_cardinality": True},
    {"policy": "declared", "alphabets": {}},
    {"policy": "declared", "alphabets": {0: np.array([0, 0])}},
    {"policy": "declared", "alphabets": {0: np.array([1])}},
    {"policy": "declared", "alphabets": {0: [0, 1]}},
    {"policy": "declared", "alphabets": {0: np.array([0.5])}},
    {"policy": "declared", "alphabets": {0: np.array([])}},
    {"policy": "declared", "alphabets": {0: np.ma.array([0], mask=False)}},
])
def test_invalid_alphabet_configurations(kwargs):
    _, _, batches = partitions(np.zeros(10, dtype=int))
    with pytest.raises((ValueError, TypeError)):
        CategoricalEncoder.fit(batches["train"], **kwargs)


def test_cardinality_and_memory_limits_before_dense_allocation(monkeypatch):
    _, _, batches = partitions(np.tile([[0, 0], [1, 1]], (5, 1)))
    alphabets = {0: np.arange(1000), 1: np.array([0, 1])}
    with pytest.raises(ValueError, match="max_cardinality"):
        CategoricalEncoder.fit(batches["train"], policy="declared", alphabets=alphabets,
                               max_cardinality=999)
    encoder = CategoricalEncoder.fit(batches["train"], policy="declared", alphabets=alphabets)

    def allocation_forbidden(*args, **kwargs):
        raise AssertionError("Dense allocation attempted despite memory budget")

    monkeypatch.setattr(np, "zeros", allocation_forbidden)
    with pytest.raises(ValueError, match="max_bytes"):
        encoder.conditioning_features(batches["test"], (0,), target=1, max_bytes=7999)


def test_singleton_features_and_memory_boundary():
    _, _, batches = partitions(np.zeros((10, 2), dtype=int))
    encoder = CategoricalEncoder.fit(batches["train"])
    features = encoder.conditioning_features(batches["test"], (0,), target=1, max_bytes=8)
    np.testing.assert_array_equal(features, np.ones((2, 1), dtype=np.float32))
    np.testing.assert_array_equal(encoder.target_indices(batches["test"], 1), [0, 0])
    for limit in (0, -1, True, 8.0):
        with pytest.raises(ValueError):
            encoder.conditioning_features(batches["test"], (0,), target=1, max_bytes=limit)


@pytest.mark.parametrize("change", [{"dataset_id": "other"}, {"split_id": "other"},
    {"n_samples": 11}, {"variable_ids": (1, 0)}])
def test_encoder_rejects_inconsistent_batch_identity(change):
    _, _, batches = partitions(np.zeros((10, 2), dtype=int))
    encoder = CategoricalEncoder.fit(batches["train"])
    with pytest.raises(ValueError, match="identity"):
        encoder.target_indices(replace(batches["test"], **change), 0)


def test_encoder_rejects_role_row_mismatch():
    _, _, batches = partitions(np.zeros(10, dtype=int))
    encoder = CategoricalEncoder.fit(batches["train"])
    with pytest.raises(ValueError, match="overlap"):
        encoder.target_indices(replace(batches["train"], role="test"), 0)
    with pytest.raises(ValueError, match="outside"):
        encoder.target_indices(replace(batches["test"], role="train"), 0)


@pytest.mark.parametrize("indices", [np.array([-1]), np.array([2]), np.array([0.]),
    np.array([[0]]), np.array([True]), np.ma.array([0], mask=False), [0]])
def test_invalid_inverse_indices(indices):
    _, _, batches = partitions(np.zeros(10, dtype=int))
    encoder = CategoricalEncoder.fit(batches["train"])
    with pytest.raises(ValueError):
        encoder.inverse_target(indices, 0)


def test_public_mapping_cannot_be_modified():
    _, _, batches = partitions(np.zeros(10, dtype=int))
    alphabet = np.array([0, 1])
    encoder = CategoricalEncoder.fit(batches["train"], policy="declared", alphabets={0: alphabet})
    alphabet[:] = 8
    encoder.categories[0][:] = 9
    np.testing.assert_array_equal(encoder.categories[0], [0, 1])
    with pytest.raises(TypeError):
        encoder.mappings[0][2] = 2
    with pytest.raises(TypeError):
        encoder.cardinalities[0] = 8


def test_train_observed_mapping_is_deterministic_under_row_reordering():
    _, _, batches = partitions(np.tile([-7, 40, 3], 4))
    train = batches["train"]
    reordered = replace(train, values=train.values[::-1], row_indices=train.row_indices[::-1])
    first, second = CategoricalEncoder.fit(train), CategoricalEncoder.fit(reordered)
    assert first.mappings == second.mappings == {0: {-7: 0, 3: 1, 40: 2}}
    np.testing.assert_array_equal(first.target_indices(train, 0), second.target_indices(train, 0))


def test_declared_signed_labels_do_not_merge_with_unsigned_holdout():
    _, _, batches = partitions(np.zeros(10, dtype=np.uint64))
    encoder = CategoricalEncoder.fit(batches["train"], policy="declared",
                                    alphabets={0: np.array([-1, 0], dtype=np.int64)})
    heldout = replace(batches["test"], values=np.full(2, 2**64 - 1, dtype=np.uint64))
    with pytest.raises(ValueError, match="18446744073709551615.*variable 0.*test"):
        encoder.target_indices(heldout, 0)
    assert encoder.mappings[0] == {-1: 0, 0: 1}


def test_unknown_variable_and_nonbatch_inputs():
    _, _, batches = partitions(np.zeros(10, dtype=int))
    encoder = CategoricalEncoder.fit(batches["train"])
    with pytest.raises(TypeError):
        CategoricalEncoder.fit(np.zeros(10, dtype=int))
    with pytest.raises(TypeError):
        encoder.target_indices(np.zeros(10, dtype=int), 0)
    with pytest.raises(ValueError, match="identifier"):
        encoder.target_indices(batches["test"], "missing")
    with pytest.raises(ValueError, match="identifier"):
        encoder.inverse_target(np.array([0]), "missing")
