"""Explicit training randomness and scoped deterministic PyTorch execution."""

from contextlib import contextmanager
import hashlib
import json
import os
import platform
from typing import Iterator

import numpy as np
import torch

from src.utils.seed import set_seed


SEED_DERIVATION = "njee-term-seed-sha256-v1"


def validate_seed(seed: int) -> None:
    """Validate the existing configured-run unsigned 32-bit seed domain."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a Python int in [0, 4294967295].")


def derive_term_seed(
    training_seed: int, term_id: str | int, *, purpose: str,
) -> int:
    """Derive a seed without global RNG access or process-randomized hash().

    SHA-256 hashes UTF-8 compact JSON [version, seed, typed term ID, purpose].
    The first four digest bytes, unsigned big-endian, form the 32-bit seed.
    String and integer term IDs are distinct; purposes are initialization/shuffle.
    Finite 32-bit seeds can collide; resolved seeds should be recorded per term.
    """
    validate_seed(training_seed)
    if type(term_id) not in (str, int) or (isinstance(term_id, str) and not term_id.strip()):
        raise ValueError("term_id must be a nonempty string or Python int.")
    if purpose not in ("initialization", "shuffle"):
        raise ValueError("purpose must be initialization or shuffle.")
    payload = [SEED_DERIVATION, training_seed,
               ["str" if type(term_id) is str else "int", term_id], purpose]
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                            .encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def make_shuffle_generator(seed: int) -> torch.Generator:
    """Return a fresh CPU generator for a training DataLoader's shuffle.

    Validation/test loaders must use shuffle=False and perform no optimization;
    these utilities do not construct loaders or enforce training-loop behavior.
    """
    validate_seed(seed)
    return torch.Generator(device="cpu").manual_seed(seed)


def validate_device(device: str) -> None:
    """Accept explicit CPU or CUDA device strings; availability is checked at use."""
    if not isinstance(device, str) or not (
        device == "cpu" or device == "cuda"
        or (device.startswith("cuda:") and device[5:].isascii()
            and device[5:].isdigit() and str(int(device[5:])) == device[5:])
    ):
        raise ValueError("device must be cpu, cuda or cuda:<nonnegative index>.")


def _available_device(device: str) -> torch.device:
    validate_device(device)
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {device!r} is unavailable; no CPU fallback.")
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device index {index} is unavailable.")
    return resolved


@contextmanager
def reproducible_execution(
    seed: int, *, mode: str, device: str = "cpu",
) -> Iterator[None]:
    """Seed via set_seed; scope strict backend settings and restore on exit.

    Seeded mode leaves deterministic/backend flags as found. Strict mode enables
    deterministic algorithms with warn_only=False and disables cuDNN benchmarking.
    Unsupported deterministic operations propagate RuntimeError unchanged.
    Strict CUDA requires a supported CUBLAS_WORKSPACE_CONFIG before initialization.
    RNG streams are intentionally reset/advanced, not restored. Backend controls
    are process-global: this context is not safe for concurrent training threads.
    No bitwise identity is promised across hardware or software environments.
    """
    validate_seed(seed)
    if mode not in ("seeded", "strict"):
        raise ValueError("mode must be seeded or strict.")
    validate_device(device)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    old_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    workspace_changed = False
    try:
        if mode == "strict":
            if device.startswith("cuda"):
                if old_workspace not in (":4096:8", ":16:8"):
                    if old_workspace is not None or torch.cuda.is_initialized():
                        raise RuntimeError("Strict CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                                           "or :16:8 before CUDA initialization.")
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                    workspace_changed = True
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        _available_device(device)
        set_seed(seed)
        yield
    finally:
        if mode == "strict":
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.deterministic = cudnn_deterministic
        if workspace_changed:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def capture_environment(*, mode: str, device: str = "cpu") -> dict:
    """Capture JSON-compatible versions, device/hardware and actual backend flags.

    Call inside reproducible_execution to record active settings. No seeds,
    secrets or arbitrary environment variables are collected.
    """
    if mode not in ("seeded", "strict"):
        raise ValueError("mode must be seeded or strict.")
    resolved = _available_device(device)
    cuda_index = None
    cuda_name = None
    if resolved.type == "cuda":
        cuda_index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        cuda_name = torch.cuda.get_device_name(cuda_index)
    return {
        "reproducibility_mode": mode, "seed_derivation": SEED_DERIVATION,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "device": str(resolved), "cuda_device_index": cuda_index,
        "cuda_device_name": cuda_name, "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_num_threads": torch.get_num_threads(),
    }
