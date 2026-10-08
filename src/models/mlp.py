"""Minimal categorical conditional classifier returning raw logits."""

import torch
from torch import nn


class ConditionalMLP(nn.Module):
    """One or two Linear/ReLU hidden layers followed by a linear logit layer.

    Input is a nonempty dense (batch, input_dim) floating tensor. Its dtype and
    device must match the model parameters; no implicit casts or reshaping occur.
    Layers start on CPU with float32 parameters, independently of the global
    default dtype. Standard Module.to()/double() conversions remain supported.

    output_cardinality=1 is valid: cross-entropy for the sole class is zero.
    An empirically constant binary target still requires two outputs when its
    encoder declares two categories. input_dim=0 is rejected: unconditional
    marginals and singleton shortcuts belong to future estimator orchestration.
    The caller supplies conditioning order and cardinality; metadata is excluded.
    """

    def __init__(
        self, input_dim: int, output_cardinality: int, *, hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        for name, value in (("input_dim", input_dim), ("output_cardinality", output_cardinality)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive Python int, got {value!r}.")
        if (not isinstance(hidden_sizes, tuple) or not 1 <= len(hidden_sizes) <= 2
                or any(type(size) is not int or size <= 0 for size in hidden_sizes)):
            raise ValueError("hidden_sizes must be a tuple of one or two positive Python ints.")
        self.input_dim = input_dim
        self.output_cardinality = output_cardinality
        self.hidden_sizes = hidden_sizes
        layers = []
        width = input_dim
        for size in hidden_sizes:
            layers.extend((nn.Linear(width, size, device="cpu", dtype=torch.float32), nn.ReLU()))
            width = size
        layers.append(nn.Linear(width, output_cardinality, device="cpu", dtype=torch.float32))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return finite (batch, output_cardinality) logits, with gradients intact.

        TypeError indicates a non-tensor or incorrect dtype. ValueError indicates
        malformed shapes, layout/device mismatch, or nonfinite inputs. Floating
        overflow/nonfinite parameters producing nonfinite logits raise RuntimeError
        rather than returning unusable cross-entropy inputs.
        """
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor.")
        if features.layout != torch.strided or features.is_nested:
            raise ValueError("features must be a dense, non-nested tensor.")
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must have nonempty shape (batch_size, input_dim).")
        if features.shape[1] != self.input_dim:
            raise ValueError(f"Feature width must equal input_dim={self.input_dim}.")
        parameter = self.network[0].weight
        if not features.is_floating_point() or features.dtype != parameter.dtype:
            raise TypeError(f"features dtype must match model floating dtype {parameter.dtype}.")
        if features.device != parameter.device:
            raise ValueError(f"features device must match model device {parameter.device}.")
        if not torch.isfinite(features).all():
            raise ValueError("features must contain only finite values.")
        logits = self.network(features)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model produced nonfinite logits; check parameter values and overflow.")
        return logits
