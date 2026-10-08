"""Deterministic split invariants and adversarial persistence checks."""

from dataclasses import replace
import json

import numpy as np
import pytest

from src.utils.split import ALGORITHM, SplitIndices, load_split, make_iid_split, save_split


@pytest.mark.parametrize("n,fractions", [
    (3, (1/3, 1/3, 1/3)), (5, (0.6, 0.2, 0.2)),
    (101, (0.6, 0.2, 0.2)), (1000, (0.5, 0.3, 0.2)),
])
def test_coverage_disjointness_sorted_and_metadata(n, fractions):
    split = make_iid_split(n, fractions=fractions, seed=42)
    arrays = [split.train, split.validation, split.test]
    np.testing.assert_array_equal(np.sort(np.concatenate(arrays)), np.arange(n))
    for i, values in enumerate(arrays):
        assert values.dtype == np.int64
        assert len(values) > 0
        assert np.all(values[1:] > values[:-1])
        assert not values.flags.writeable
        for other in arrays[i + 1:]:
            assert np.intersect1d(values, other).size == 0
    assert split.n_samples == n
    assert split.fractions == fractions
    assert split.seed == 42
    assert split.algorithm == ALGORITHM


@pytest.mark.parametrize("seed", [0, 1, 42, 2**32 - 1])
def test_reproducibility(seed):
    first = make_iid_split(100, seed=seed)
    second = make_iid_split(100, seed=seed)
    for role in ("train", "validation", "test"):
        np.testing.assert_array_equal(getattr(first, role), getattr(second, role))


def test_different_seeds():
    assert not np.array_equal(make_iid_split(100, seed=1).train,
                              make_iid_split(100, seed=2).train)


def test_local_rng_isolation():
    original = np.random.get_state()
    try:
        np.random.seed(123)
        before = np.random.get_state()
        first = make_iid_split(100, seed=4)
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]
        np.random.rand(100)
        np.testing.assert_array_equal(first.train, make_iid_split(100, seed=4).train)
    finally:
        np.random.set_state(original)


@pytest.mark.parametrize("n,fractions,counts", [
    (7, (0.6, 0.2, 0.2), (4, 2, 1)),
    (8, (1/3, 1/3, 1/3), (3, 3, 2)),
    (11, (0.5, 0.3, 0.2), (6, 3, 2)),
    (10, (0.6, 0.2, 0.2), (6, 2, 2)),
    (10, (0.6, 0.2, 0.2000000000001), (6, 2, 2)),
])
def test_largest_remainder_and_ties(n, fractions, counts):
    split = make_iid_split(n, fractions=fractions, seed=0)
    assert tuple(len(getattr(split, role)) for role in ("train", "validation", "test")) == counts
    assert split.fractions == fractions


@pytest.mark.parametrize("fractions", [
    None, "0.6,0.2,0.2", [0.6, 0.2, 0.2], (0.5, 0.5),
    (0.4, 0.2, 0.2, 0.2), (True, 0.2, 0.2), ("0.6", 0.2, 0.2),
    (None, 0.2, 0.2), (complex(0.6), 0.2, 0.2),
    (float("nan"), 0.2, 0.2), (float("inf"), 0.2, 0.2),
    (0, 0.5, 0.5), (-0.1, 0.5, 0.6), (0.5, 0.2, 0.2),
    (0.6, 0.2, 0.20000000001), (1, 1, 1), (10**400, 0.2, 0.2),
])
def test_invalid_fractions(fractions):
    with pytest.raises(ValueError, match="fractions"):
        make_iid_split(100, fractions=fractions, seed=0)


@pytest.mark.parametrize("n", [0, -1, True, False, 3.0, "3", None, np.int64(3), 2**63])
def test_invalid_sample_counts(n):
    with pytest.raises(ValueError, match="n_samples"):
        make_iid_split(n, seed=0)


@pytest.mark.parametrize("seed", [-1, 2**32, True, False, 1.0, "1", None, np.int64(1)])
def test_invalid_seeds(seed):
    with pytest.raises(ValueError, match="seed"):
        make_iid_split(10, seed=seed)


@pytest.mark.parametrize("n,fractions", [(1, (0.6, 0.2, 0.2)),
    (2, (0.6, 0.2, 0.2)), (3, (0.6, 0.2, 0.2)), (100, (0.998, 0.001, 0.001))])
def test_empty_allocations(n, fractions):
    with pytest.raises(ValueError, match="empty"):
        make_iid_split(n, fractions=fractions, seed=0)


@pytest.mark.parametrize("train,validation,test", [
    ([0, 0, 2], [3], [4]), ([0, 1, 2], [2], [4]),
    ([0, 1, 5], [3], [4]), ([-1, 1, 2], [3], [4]),
    ([2, 1, 0], [3], [4]), ([0, 1], [3], [4]),
])
def test_invalid_indices(train, validation, test):
    with pytest.raises(ValueError):
        SplitIndices(*(np.array(a) for a in (train, validation, test)),
                     n_samples=5, fractions=(0.6, 0.2, 0.2), seed=0)


@pytest.mark.parametrize("values", [np.array([0., 1., 2.]), np.array([[0, 1, 2]]),
    np.array([False, True, True]), [0, 1, 2],
    np.ma.array([0, 1, 2], mask=False), np.array([0, 1, 2], dtype=object)])
def test_invalid_index_types(values):
    with pytest.raises(ValueError):
        replace(make_iid_split(5, seed=0), train=values)


def test_constructor_copies_arrays():
    train = np.array([0, 1, 2])
    split = SplitIndices(train, np.array([3]), np.array([4]), 5, (0.6, 0.2, 0.2), 0)
    train[0] = 4
    np.testing.assert_array_equal(split.train, [0, 1, 2])


def test_round_trip(tmp_path):
    path = tmp_path / "split.npz"
    original = make_iid_split(101, seed=2**32 - 1)
    save_split(original, path)
    restored = load_split(path)
    for role in ("train", "validation", "test"):
        np.testing.assert_array_equal(getattr(original, role), getattr(restored, role))
    assert restored.n_samples == original.n_samples
    assert restored.fractions == original.fractions
    assert restored.seed == original.seed
    assert restored.algorithm == original.algorithm
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"train", "validation", "test"}
        assert all(archive[k].dtype == np.int64 for k in archive.files)


@pytest.mark.parametrize("change", [
    {"seed": -1}, {"seed": True}, {"n_samples": 6}, {"fractions": [0.5, 0.2, 0.2]},
    {"fractions": "bad"}, {"algorithm": "unknown"}, {"extra": 1},
])
def test_malformed_metadata(tmp_path, change):
    path = tmp_path / "split.npz"
    save_split(make_iid_split(5, seed=0), path)
    sidecar = path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    metadata.update(change)
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="persisted split"):
        load_split(path)


@pytest.mark.parametrize("contents", ["{", "[]", "{}"])
def test_invalid_json(tmp_path, contents):
    path = tmp_path / "split.npz"
    save_split(make_iid_split(5, seed=0), path)
    path.with_suffix(".json").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        load_split(path)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "extra", "object", "truncated"])
def test_malformed_archive(tmp_path, kind):
    path = tmp_path / "split.npz"
    save_split(make_iid_split(5, seed=0), path)
    arrays = {"train": np.array([0, 1, 2]), "validation": np.array([3]), "test": np.array([4])}
    if kind == "duplicate":
        arrays["test"] = np.array([3])
    elif kind == "missing":
        del arrays["test"]
    elif kind == "extra":
        arrays["extra"] = np.array([0])
    elif kind == "object":
        arrays["test"] = np.array([4], dtype=object)
    if kind == "truncated":
        path.write_bytes(b"not an npz")
    else:
        np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_split(path)


def test_missing_files_and_path_suffix(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path / "absent.npz")
    with pytest.raises(ValueError, match="suffix"):
        save_split(make_iid_split(5, seed=0), tmp_path / "split")
    with pytest.raises(ValueError, match="suffix"):
        load_split(tmp_path / "split")


def test_save_revalidates_mutated_indices(tmp_path):
    split = make_iid_split(5, seed=0)
    split.train.flags.writeable = True
    split.train[0] = split.test[0]
    with pytest.raises(ValueError):
        save_split(split, tmp_path / "split.npz")


def test_corrupt_zip(tmp_path):
    path = tmp_path / "split.npz"
    save_split(make_iid_split(5, seed=0), path)
    path.write_bytes(b"PK\x03\x04broken zip")
    with pytest.raises(ValueError, match="persisted split"):
        load_split(path)


def test_missing_archive(tmp_path):
    path = tmp_path / "split.npz"
    path.with_suffix(".json").write_text(json.dumps({
        "n_samples": 5, "fractions": [0.6, 0.2, 0.2], "seed": 0, "algorithm": ALGORITHM,
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_split(path)


def test_allocation_counts_cannot_be_reassigned():
    with pytest.raises(ValueError, match="allocation"):
        SplitIndices(np.array([0, 1]), np.array([2, 3]), np.array([4]),
                     5, (0.6, 0.2, 0.2), 0)
