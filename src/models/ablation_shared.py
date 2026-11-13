import torch
from torch import nn

from src.models.mlp import ConditionalMLP


class AblationMultiHead(nn.Module):
    """Shared trunk or separate trunks, identical slots and term indicators.

    One head per non-marginal chain term. Slots identify physical variables;
    absent/future variables are zero. A one-hot term identifier is appended.
    Only trunks share parameters; heads and the plug-in marginal never share.
    All declared alphabets must have the same cardinality for this variant.
    """
    def __init__(self, d, cardinality, *, hidden_sizes=(16,), shared=True):
        super().__init__()
        if type(d) is not int or d < 2 or type(cardinality) is not int or cardinality < 2:
            raise ValueError('Require at least two variables and two classes.')
        if type(shared) is not bool:
            raise ValueError('shared must be boolean.')
        self.d, self.cardinality, self.shared = d, cardinality, shared
        self.input_dim = d * cardinality + d - 1
        # Reuse the validated layer/type validation; discard only its output layer.
        def trunk():
            network = ConditionalMLP(self.input_dim, cardinality, hidden_sizes=hidden_sizes).network
            return nn.Sequential(*list(network.children())[:-1])
        self.trunks = nn.ModuleList([trunk() for _ in range(1 if shared else d - 1)])
        self.heads = nn.ModuleList([nn.Linear(hidden_sizes[-1], cardinality,
                                            dtype=torch.float32, device='cpu') for _ in range(d - 1)])

    def forward(self, features, term):
        if type(term) is not int or not 0 <= term < self.d - 1:
            raise ValueError('Unknown term.')
        if (not isinstance(features, torch.Tensor) or features.ndim != 2 or not len(features)
                or features.shape[1] != self.input_dim):
            raise ValueError('Malformed fixed-slot input.')
        p = self.heads[term].weight
        if features.dtype != p.dtype or features.device != p.device:
            raise TypeError('Input dtype/device must match model.')
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite features.')
        result = self.heads[term](self.trunks[0 if self.shared else term](features))
        if not torch.isfinite(result).all():
            raise RuntimeError('Nonfinite logits.')
        return result


def fixed_slot_features(encoder, batch, order, position):
    """No target/future values enter features; identity is explicit and nonnumeric."""
    import numpy as np
    if type(position) is not int or not 1 <= position < len(order):
        raise ValueError('Expected a non-marginal chain position.')
    if len(set(encoder.cardinalities.values())) != 1 or set(order) != set(encoder.variable_ids):
        raise ValueError('Requires complete order and equal declared alphabets.')
    k = encoder.cardinalities[order[0]]
    features = np.zeros((len(batch.values), len(order) * k + len(order) - 1), dtype=np.float32)
    for variable in order[:position]:
        slot = encoder.variable_ids.index(variable)
        indices = encoder.target_indices(batch, variable)
        features[np.arange(len(indices)), slot * k + indices] = 1.
    features[:, len(order) * k + position - 1] = 1.
    return features
