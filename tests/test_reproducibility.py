"""CPU reproducibility and simulated deterministic-backend failures."""

import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.reproducibility import (
    SEED_DERIVATION, capture_environment, derive_term_seed, make_shuffle_generator,
    reproducible_execution, validate_seed,
)
from src.utils.seed import set_seed
from src.utils.split import make_iid_split


@pytest.fixture(autouse=True)
def preserve_process_state():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    flags = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags[2:]


@pytest.mark.parametrize("seed", [0, 42, 2**32 - 1])
@pytest.mark.parametrize("mode", ["seeded", "strict"])
def test_explicit_seed_reproduces_all_cpu_rngs(seed, mode):
    results = []
    for _ in range(2):
        with reproducible_execution(seed, mode=mode):
            results.append((random.random(), np.random.rand(), torch.rand(4)))
    assert results[0][:2] == results[1][:2]
    assert torch.equal(results[0][2], results[1][2])


@pytest.mark.parametrize("seed", [-1, 2**32, True, False, 1.0, None, "1", np.int64(1)])
def test_invalid_seeds(seed):
    for operation in (validate_seed, make_shuffle_generator):
        with pytest.raises(ValueError):
            operation(seed)
    with pytest.raises(ValueError):
        derive_term_seed(seed, "X", purpose="initialization")
    with pytest.raises(ValueError):
        with reproducible_execution(seed, mode="seeded"):
            pass


def test_stable_typed_domain_separated_seed_derivation():
    payload = '["njee-term-seed-sha256-v1",202,["str","X"],"initialization"]'
    reference = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")
    assert derive_term_seed(202, "X", purpose="initialization") == reference
    seeds = [derive_term_seed(202, term, purpose=purpose)
             for term in ("X", "Y", 1, "1") for purpose in ("initialization", "shuffle")]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= s <= 2**32 - 1 for s in seeds)
    assert derive_term_seed(203, "X", purpose="initialization") != reference
    assert SEED_DERIVATION == "njee-term-seed-sha256-v1"


@pytest.mark.parametrize("term,purpose", [(True, "shuffle"), (None, "shuffle"),
    ("", "shuffle"), ("X", "test"), ("X", None)])
def test_invalid_term_derivation(term, purpose):
    with pytest.raises(ValueError):
        derive_term_seed(0, term, purpose=purpose)


def test_local_derivation_and_generator_do_not_change_global_rngs():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    generator = make_shuffle_generator(derive_term_seed(42, "X", purpose="shuffle"))
    torch.randperm(100, generator=generator)
    assert random.getstate() == before[0]
    after = np.random.get_state()
    np.testing.assert_array_equal(after[1], before[1][1])
    assert after[0] == before[1][0] and after[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


def test_deterministic_dataloader_shuffle_and_validation_order():
    dataset = TensorDataset(torch.arange(31))

    def shuffled(seed):
        loader = DataLoader(dataset, batch_size=7, shuffle=True, num_workers=0,
                            generator=make_shuffle_generator(seed), drop_last=False)
        return torch.cat([batch[0] for batch in loader])

    first = shuffled(derive_term_seed(42, "X", purpose="shuffle"))
    torch.rand(100)
    assert torch.equal(first, shuffled(derive_term_seed(42, "X", purpose="shuffle")))
    assert not torch.equal(first, shuffled(derive_term_seed(42, "Y", purpose="shuffle")))
    assert torch.equal(first.sort().values, torch.arange(31))
    validation = DataLoader(dataset, batch_size=7, shuffle=False, num_workers=0,
                            generator=make_shuffle_generator(0))
    assert torch.equal(torch.cat([batch[0] for batch in validation]), torch.arange(31))


def test_seeded_preserves_backend_flags_and_strict_restores():
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    with reproducible_execution(0, mode="seeded"):
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark
    with reproducible_execution(0, mode="strict"):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.benchmark
    assert not torch.backends.cudnn.deterministic


def test_unsupported_operation_raises_and_restores_flags():
    torch.use_deterministic_algorithms(False)
    # Model an unsupported backend operation without requiring a GPU.
    def unsupported_operation():
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("operation has no deterministic implementation")

    with pytest.raises(RuntimeError, match="no deterministic implementation"):
        with reproducible_execution(0, mode="strict"):
            unsupported_operation()
    assert not torch.are_deterministic_algorithms_enabled()


def test_actual_unsupported_cpu_operation_fails_in_strict_mode():
    # PyTorch put_ with accumulate=False has no deterministic implementation.
    with pytest.raises(RuntimeError, match="deterministic"):
        with reproducible_execution(0, mode="strict"):
            torch.zeros(3).put_(torch.tensor([0, 1]), torch.ones(2), accumulate=False)


def test_nested_context_restores_outer_strict_mode():
    with reproducible_execution(0, mode="strict"):
        with reproducible_execution(1, mode="seeded"):
            assert torch.are_deterministic_algorithms_enabled()
        assert torch.are_deterministic_algorithms_enabled()


def test_invalid_cuda_workspace_is_not_overwritten(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        with reproducible_execution(0, mode="strict", device="cuda"):
            pass
    import os
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "invalid"


@pytest.mark.parametrize("mode,device", [("bad", "cpu"), ("seeded", "auto"),
    ("strict", "mps"), ("strict", "cuda:-1")])
def test_invalid_mode_or_device(mode, device):
    with pytest.raises(ValueError):
        with reproducible_execution(0, mode=mode, device=device):
            pass


def test_unavailable_cuda_fails_without_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        with reproducible_execution(0, mode="seeded", device="cuda"):
            pass


def test_strict_cuda_requires_setup_before_initialization(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        with reproducible_execution(0, mode="strict", device="cuda"):
            pass


def test_cuda_environment_restored_on_setup_failure(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        with reproducible_execution(0, mode="strict", device="cuda"):
            pass
    import os
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_environment_metadata_is_json_serializable_and_records_active_mode():
    with reproducible_execution(42, mode="strict"):
        metadata = capture_environment(mode="strict")
        assert metadata["reproducibility_mode"] == "strict"
        assert metadata["device"] == "cpu" and metadata["cuda_device_name"] is None
        assert metadata["deterministic_algorithms"] and not metadata["deterministic_warn_only"]
        assert metadata["torch_version"] == str(torch.__version__)
        assert metadata["numpy_version"] == np.__version__
        json.dumps(metadata, allow_nan=False)
    with pytest.raises(ValueError):
        capture_environment(mode="bad")


def test_seed_and_split_nonregression():
    baseline = make_iid_split(100, seed=42)
    set_seed(42)
    expected = (random.random(), np.random.rand(), torch.rand(4))
    with reproducible_execution(42, mode="seeded"):
        assert random.random() == expected[0]
        assert np.random.rand() == expected[1]
        assert torch.equal(torch.rand(4), expected[2])
        actual = make_iid_split(100, seed=42)
    for role in ("train", "validation", "test"):
        np.testing.assert_array_equal(getattr(baseline, role), getattr(actual, role))
