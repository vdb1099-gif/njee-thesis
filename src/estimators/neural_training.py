"""Train one conditional NJEE term; no test access or entropy orchestration."""

from dataclasses import dataclass
import json
import math
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import ConditionalMLP
from src.utils.categorical import CategoricalBatch, CategoricalEncoder, VariableId, validate_variable_order
from src.utils.neural_config import NeuralConfig
from src.utils.reproducibility import (
    capture_environment, derive_term_seed, make_shuffle_generator, reproducible_execution,
)


@dataclass(frozen=True)
class TermTrainingResult:
    """Completed fit with frozen eval-mode model and diagnostics in nats.

    Epochs are 1-based; singleton fits have zero epochs and best_epoch=None.
    final_training_ce_nats evaluates the restored model on TRAIN, whereas
    epoch_training_ce_nats records losses encountered during parameter updates.
    Runtime includes encoding, initialization, fit and restored-train evaluation.
    There is no resumable/refit lifecycle: each function call constructs a new model.
    """

    model: ConditionalMLP
    target: VariableId
    conditioning: tuple[VariableId, ...]
    output_cardinality: int
    term_id: str
    initialization_seed: int
    shuffle_seed: int
    epochs_trained: int
    best_epoch: int | None
    best_validation_ce_nats: float
    final_training_ce_nats: float
    runtime_seconds: float
    stop_reason: str
    warnings: tuple[str, ...]
    epoch_training_ce_nats: tuple[float, ...]
    epoch_validation_ce_nats: tuple[float, ...]
    environment: dict


def _finite_loss(loss: torch.Tensor) -> None:
    if not torch.isfinite(loss).all():
        raise RuntimeError("Nonfinite cross-entropy during conditional term training.")


def _evaluate(model: ConditionalMLP, loader: DataLoader, device: str) -> float:
    """Private TRAIN/VALIDATION CE helper, weighted by observations, without updates."""
    model.eval()
    losses, count = [], 0
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(device))
            loss = nn.functional.cross_entropy(logits, targets.to(device), reduction="none")
            _finite_loss(loss)
            losses.append(loss.double().sum().item())
            count += len(targets)
    return math.fsum(losses) / count


def train_conditional_term(
    train: CategoricalBatch, validation: CategoricalBatch, *,
    encoder: CategoricalEncoder, config: NeuralConfig,
    target: VariableId, conditioning: tuple[VariableId, ...],
    context: tuple[VariableId, ...] = (),
) -> TermTrainingResult:
    """Fit one ordered conditional term using TRAIN and VALIDATION only.

    The encoder must already be fitted. Without context, conditioning must equal
    the nonempty configured chain prefix (the original NJEE contract). With an
    explicit nonempty context Y, it must equal Y followed by the X chain prefix;
    the first X term is then supported. X and Y must be disjoint, known IDs.
    Unused dataset columns are permitted with context. Identity checks reuse
    encoder validation, and row overlap is rejected.
    A test argument, supplied model, resume checkpoint and encoder refit are absent.

    Adam uses only configured learning_rate with PyTorch's standard defaults,
    zero weight decay and no additional optimization mechanisms. TRAIN shuffles;
    VALIDATION does not. All loaders use zero workers and drop_last=False.

    Improvement is best_loss - current_loss > min_delta. The first epoch becomes
    the initial best; ties/subthreshold changes retain that snapshot and increment
    patience. Stop after patience consecutive non-improvements or max_epochs.
    Independent CPU checkpoint clones are restored before returning a frozen model.
    Numerical/deterministic failures raise, with no partial successful result.
    """
    start = perf_counter()
    if not isinstance(train, CategoricalBatch) or not isinstance(validation, CategoricalBatch):
        raise TypeError("train and validation must be CategoricalBatch objects.")
    if train.role != "train" or validation.role != "validation":
        raise ValueError("Expected train and validation roles; test is not accepted.")
    if not isinstance(config, NeuralConfig):
        raise TypeError("config must be a validated NeuralConfig.")
    if not isinstance(encoder, CategoricalEncoder):
        raise TypeError("encoder must be a fitted CategoricalEncoder.")
    if not hasattr(encoder, "_mappings"):
        raise ValueError("encoder must be fitted before training.")
    if context != ():
        validate_variable_order(context, context)
    if context:
        if set(context) & set(config.component_order):
            raise ValueError("Context and target-variable group must be disjoint.")
        if not set(context) <= set(train.variable_ids):
            raise ValueError("Unknown context variable identifier.")
        group = tuple(v for v in train.variable_ids if v in config.component_order)
        config.validate_dataset(variable_ids=group, n_samples=train.n_samples)
    else:
        config.validate_dataset(variable_ids=train.variable_ids, n_samples=train.n_samples)
    if type(target) not in (str, int) or target not in config.component_order:
        raise ValueError(f"Unknown target variable {target!r}.")
    if not isinstance(conditioning, tuple) or any(type(v) not in (str, int) for v in conditioning):
        raise ValueError("conditioning must be a tuple of valid variable identifiers.")
    if target in conditioning:
        raise ValueError("Target must be excluded from its conditioning features.")
    prefix = context + config.component_order[:config.component_order.index(target)]
    if not prefix or conditioning != prefix:
        raise ValueError("conditioning must equal context plus the nonempty configured prefix before target.")
    if encoder.policy != config.alphabet_policy:
        raise ValueError("Encoder alphabet policy differs from neural configuration.")
    if np.intersect1d(train.row_indices, validation.row_indices).size:
        raise ValueError("Training and validation observations overlap.")
    # Encoder checks dataset/split identities, original columns and fitted row roles.
    train_x = encoder.conditioning_features(train, conditioning, target=target)
    train_y = encoder.target_indices(train, target)
    validation_x = encoder.conditioning_features(validation, conditioning, target=target)
    validation_y = encoder.target_indices(validation, target)
    cardinality = encoder.cardinalities[target]
    # Structured JSON preserves integer/string IDs and conditioning order.
    term_id = json.dumps(["njee-conditional-term-v1", target, list(conditioning)],
                         ensure_ascii=False, separators=(",", ":"))
    init_seed = derive_term_seed(config.training_seed, term_id, purpose="initialization")
    shuffle_seed = derive_term_seed(config.training_seed, term_id, purpose="shuffle")
    warnings = []
    missing = np.flatnonzero(np.bincount(train_y, minlength=cardinality) == 0)
    if len(missing):
        warnings.append(f"Target classes absent from training: {missing.tolist()}; alphabet retained.")
    epoch_train, epoch_validation = [], []
    with reproducible_execution(init_seed, mode=config.reproducibility_mode, device=config.device):
        # No RNG-consuming operation occurs between seeding and model construction.
        model = ConditionalMLP(train_x.shape[1], cardinality, hidden_sizes=config.hidden_sizes)
        model.to(config.device)
        environment = capture_environment(mode=config.reproducibility_mode, device=config.device)
        if cardinality == 1:
            best_epoch, best_loss, final_train, reason = None, 0.0, 0.0, "singleton_target"
            warnings.append("Singleton target has zero cross-entropy; no optimization performed.")
        else:
            train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
            validation_data = TensorDataset(torch.from_numpy(validation_x), torch.from_numpy(validation_y))
            train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                      drop_last=False, num_workers=0,
                                      generator=make_shuffle_generator(shuffle_seed))
            # Separate generators avoid validation iteration consuming global/training RNG.
            validation_loader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=False,
                                           drop_last=False, num_workers=0,
                                           generator=make_shuffle_generator(shuffle_seed))
            train_eval_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=False,
                                           drop_last=False, num_workers=0,
                                           generator=make_shuffle_generator(shuffle_seed))
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            best_loss, best_epoch, snapshot, stale = math.inf, None, None, 0
            reason = "max_epochs"
            for epoch in range(1, config.max_epochs + 1):
                model.train()
                losses, count = [], 0
                for features, targets in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(features.to(config.device)),
                                                       targets.to(config.device), reduction="none")
                    _finite_loss(loss)
                    loss.mean().backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all()
                           for p in model.parameters()):
                        raise RuntimeError("Nonfinite gradients during conditional term training.")
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in model.parameters()):
                        raise RuntimeError("Nonfinite parameters after Adam update.")
                    losses.append(loss.detach().double().sum().item())
                    count += len(targets)
                epoch_train.append(math.fsum(losses) / count)
                validation_loss = _evaluate(model, validation_loader, config.device)
                # Also guard injected/custom evaluation failures at this boundary.
                if not math.isfinite(validation_loss):
                    raise RuntimeError("Nonfinite validation cross-entropy.")
                epoch_validation.append(validation_loss)
                if best_epoch is None or best_loss - validation_loss > config.min_delta:
                    best_loss, best_epoch, stale = validation_loss, epoch, 0
                    snapshot = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= config.patience:
                        reason = "early_stopping"
                        break
            model.load_state_dict(snapshot)
            final_train = _evaluate(model, train_eval_loader, config.device)
            if not math.isfinite(final_train):
                raise RuntimeError("Nonfinite restored-model training cross-entropy.")
            if reason == "max_epochs":
                warnings.append("Maximum epochs reached; convergence is not established.")
        model.eval()
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
    return TermTrainingResult(
        model, target, conditioning, cardinality, term_id, init_seed, shuffle_seed,
        len(epoch_train), best_epoch, best_loss, final_train, perf_counter() - start,
        reason, tuple(warnings), tuple(epoch_train), tuple(epoch_validation), environment,
    )
