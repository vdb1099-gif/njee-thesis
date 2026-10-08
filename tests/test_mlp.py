"""Deterministic classifier mathematics and encoder compatibility, without training."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from src.models.mlp import ConditionalMLP
from src.utils.categorical import CategoricalDataset, CategoricalEncoder
from src.utils.reproducibility import derive_term_seed, reproducible_execution
from src.utils.split import make_iid_split


@pytest.fixture(autouse=True)
def preserve_torch_rng():
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


@pytest.mark.parametrize("hidden", [(4,), (5, 3)])
def test_exact_architecture(hidden):
    model = ConditionalMLP(6, 2, hidden_sizes=hidden)
    layers = list(model.network)
    assert [type(layer) for layer in layers] == [nn.Linear, nn.ReLU] * len(hidden) + [nn.Linear]
    widths = (6, *hidden, 2)
    for layer, incoming, outgoing in zip(layers[::2], widths, widths[1:]):
        assert (layer.in_features, layer.out_features) == (incoming, outgoing)
        assert layer.bias is not None
    assert model.input_dim == 6 and model.output_cardinality == 2
    assert model.hidden_sizes == hidden


@pytest.mark.parametrize("hidden", [(4,), (5, 3)])
@pytest.mark.parametrize("batch_size", [1, 2, 17])
@pytest.mark.parametrize("cardinality", [1, 2, 7])
def test_output_shapes_and_finiteness(hidden, batch_size, cardinality):
    torch.manual_seed(11)
    model = ConditionalMLP(6, cardinality, hidden_sizes=hidden)
    inputs = torch.arange(batch_size * 6, dtype=torch.float32).reshape(batch_size, 6) / 10
    before = inputs.clone()
    logits = model(inputs)
    assert logits.shape == (batch_size, cardinality)
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    torch.testing.assert_close(inputs, before)


@pytest.mark.parametrize("field,value", [("input_dim", 0), ("input_dim", -1),
    ("input_dim", True), ("input_dim", 2.0), ("input_dim", "2"),
    ("output_cardinality", 0), ("output_cardinality", -1),
    ("output_cardinality", False), ("output_cardinality", 2.0),
    ("output_cardinality", None), ("hidden_sizes", ()), ("hidden_sizes", (1, 2, 3)),
    ("hidden_sizes", (0,)), ("hidden_sizes", (-1,)), ("hidden_sizes", (True,)),
    ("hidden_sizes", (3.0,)), ("hidden_sizes", [3]), ("hidden_sizes", None)])
def test_invalid_construction(field, value):
    kwargs = dict(input_dim=2, output_cardinality=3, hidden_sizes=(4,))
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        ConditionalMLP(**kwargs)


@pytest.mark.parametrize("shape", [(), (2,), (2, 2, 1), (0, 2), (3, 0), (3, 1), (3, 4)])
def test_invalid_input_shape(shape):
    model = ConditionalMLP(2, 3, hidden_sizes=(4,))
    with pytest.raises(ValueError):
        model(torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float64,
    torch.float16, torch.bfloat16, torch.complex64])
def test_invalid_dtype(dtype):
    model = ConditionalMLP(2, 3, hidden_sizes=(4,))
    with pytest.raises(TypeError, match="dtype"):
        model(torch.zeros((3, 2), dtype=dtype))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs(value):
    model = ConditionalMLP(2, 3, hidden_sizes=(4,))
    inputs = torch.zeros((1, 2))
    inputs[0, 1] = value
    with pytest.raises(ValueError, match="finite"):
        model(inputs)


def test_non_tensor_sparse_and_wrong_device():
    model = ConditionalMLP(2, 3, hidden_sizes=(4,))
    with pytest.raises(TypeError):
        model(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="dense"):
        model(torch.zeros((2, 2)).to_sparse())
    with pytest.raises(ValueError, match="device"):
        model(torch.empty((2, 2), device="meta"))


def test_noncontiguous_inputs_and_explicit_model_dtype_conversion():
    model = ConditionalMLP(2, 3, hidden_sizes=(4,)).double()
    inputs = torch.arange(12, dtype=torch.float64).reshape(3, 4)[:, ::2]
    assert not inputs.is_contiguous()
    assert model(inputs).dtype == torch.float64


def test_initial_float32_does_not_depend_on_default_dtype():
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = ConditionalMLP(2, 3, hidden_sizes=(4,))
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert model(torch.ones((1, 2), dtype=torch.float32)).dtype == torch.float32
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize("hidden", [(3,), (3, 2)])
def test_gradients_reach_all_layers_and_inputs(hidden):
    model = ConditionalMLP(2, 2, hidden_sizes=hidden)
    with torch.no_grad():
        for layer in model.network:
            if isinstance(layer, nn.Linear):
                layer.weight.fill_(0.2)
                layer.bias.fill_(0.1)
        model.network[-1].weight[1].fill_(0.5)
    inputs = torch.ones((2, 2), requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(inputs), torch.zeros(2, dtype=torch.int64))
    loss.backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad) > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_manual_forward_logits_and_cross_entropy_in_nats():
    model = ConditionalMLP(2, 3, hidden_sizes=(2,))
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(2))
        model.network[0].bias.zero_()
        model.network[-1].weight.copy_(torch.tensor([[1., 0.], [0., 2.], [-1., 0.]]))
        model.network[-1].bias.copy_(torch.tensor([0.5, -0.5, 1.]))
    logits = model(torch.tensor([[1., -2.], [-1., 2.]]))
    expected = [[1.5, -0.5, 0.], [0.5, 3.5, 1.]]
    torch.testing.assert_close(logits, torch.tensor(expected))
    targets = torch.tensor([0, 2], dtype=torch.int64)
    reference = sum(math.log(sum(math.exp(v) for v in row)) - row[target]
                    for row, target in zip(expected, [0, 2])) / 2
    assert nn.CrossEntropyLoss()(logits, targets).item() == pytest.approx(reference, abs=1e-6)
    assert (logits < 0).any() and (logits > 1).any()
    assert not torch.allclose(logits.sum(dim=1), torch.ones(2))


@pytest.mark.parametrize("cardinality", [1, 2])
def test_encoded_constant_targets_keep_declared_cardinality(cardinality):
    dataset = CategoricalDataset(np.tile([[-7, 0], [-7, 1]], (5, 1)), "data", ("X", "Y"))
    split = make_iid_split(10, seed=42)
    train = dataset.batch(split, "train", split_id="s")
    batch = dataset.batch(split, "test", split_id="s")
    alphabet = np.array([-7]) if cardinality == 1 else np.array([-7, 9])
    encoder = CategoricalEncoder.fit(train, policy="declared",
                                    alphabets={"X": alphabet, "Y": np.array([0, 1])})
    features = encoder.conditioning_features(batch, ("Y",), target="X")
    targets = encoder.target_indices(batch, "X")
    model = ConditionalMLP(features.shape[1], encoder.cardinalities["X"], hidden_sizes=(4,))
    logits = model(torch.from_numpy(features))
    loss = nn.CrossEntropyLoss()(logits, torch.from_numpy(targets))
    assert logits.shape == (len(batch.values), cardinality)
    assert torch.isfinite(loss)
    if cardinality == 1:
        assert loss.item() == 0.0
        loss.backward()
        assert all(torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_encoder_multiblock_order_and_model_compatibility():
    dataset = CategoricalDataset(np.tile([[0, -3, 9], [1, 5, 2]], (5, 1)),
                                 "data", ("target", "left", "right"))
    split = make_iid_split(10, seed=42)
    train = dataset.batch(split, "train", split_id="s")
    encoder = CategoricalEncoder.fit(train, policy="declared", alphabets={
        "target": np.array([0, 1]), "left": np.array([-3, 5]), "right": np.array([2, 9]),
    })
    features = encoder.conditioning_features(train, ("right", "left"), target="target")
    model = ConditionalMLP(4, 2, hidden_sizes=(5, 3))
    loss = nn.CrossEntropyLoss()(model(torch.from_numpy(features)),
                                torch.from_numpy(encoder.target_indices(train, "target")))
    assert torch.isfinite(loss)
    assert encoder.variable_ids == ("target", "left", "right")


def test_reproducible_initialization_uses_existing_utilities():
    seed = derive_term_seed(42, "X", purpose="initialization")
    parameters = []
    for _ in range(2):
        with reproducible_execution(seed, mode="strict"):
            parameters.append(ConditionalMLP(2, 3, hidden_sizes=(4,)).state_dict())
    for name in parameters[0]:
        assert torch.equal(parameters[0][name], parameters[1][name])


def test_nonfinite_logits_fail_clearly():
    model = ConditionalMLP(2, 2, hidden_sizes=(3,))
    with torch.no_grad():
        model.network[-1].bias.fill_(float("inf"))
    with pytest.raises(RuntimeError, match="nonfinite logits"):
        model(torch.ones((1, 2)))
