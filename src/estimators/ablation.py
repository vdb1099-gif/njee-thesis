"""Experimental ablation fitting only; validated NJEE/CMI/TE remain unchanged.

This module accepts explicit TRAIN subsets, reuses train-only encoders and the
validated conditional trainer wherever its stopping semantics are appropriate.
It does not claim to extend the public validated estimator definitions.
"""
import json
import math
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.estimators.classical import PlugInEstimator
from src.estimators.neural import _test_ce
from src.estimators.neural_training import train_conditional_term
from src.models.ablation_shared import AblationMultiHead, fixed_slot_features
from src.models.mlp import ConditionalMLP
from src.utils.categorical import CategoricalEncoder
from src.utils.reproducibility import capture_environment, derive_term_seed, make_shuffle_generator, reproducible_execution


def stream_ids(order, variant):
    if variant in ('shared', 'sharing_control'):
        return ('ablation-pooled-fixed-slots-v1',)
    return tuple(json.dumps(['njee-conditional-term-v1', v, list(order[:i])],
                            ensure_ascii=False, separators=(',', ':')) for i, v in enumerate(order) if i)


def fit_arrays(model, train_x, train_y, validation_x, validation_y, *, config,
               shuffle_seed, early_stopping, pooled=False, cancelled=lambda: False):
    """Equal mean term CE for pooled models; ordinary per-term CE otherwise.

    Pooled controls have one Adam update per batch and one joint checkpoint.
    ES off uses exactly max_epochs and the LAST checkpoint, never validation
    selection. Validation still records losses without shuffling or updates.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    active_environment = capture_environment(mode=config.reproducibility_mode, device=config.device)
    generator = make_shuffle_generator(shuffle_seed)
    xs = [torch.from_numpy(x) for x in train_x]
    ys = [torch.from_numpy(y) for y in train_y]
    vx = [torch.from_numpy(x) for x in validation_x]
    vy = [torch.from_numpy(y) for y in validation_y]
    def losses(inputs, targets, indices=None):
        return [nn.functional.cross_entropy(model(x if indices is None else x[indices], i)
                                            if pooled else model(x if indices is None else x[indices]),
                                            y if indices is None else y[indices])
                for i, (x, y) in enumerate(zip(inputs, targets))]
    best, best_epoch, stale, updates = math.inf, None, 0, 0
    history, validation_history = [], []
    reason = 'max_epochs'
    for epoch in range(1, config.max_epochs + 1):
        if cancelled():
            raise KeyboardInterrupt('Ablation cancelled before epoch.')
        model.train()
        permutation = torch.randperm(len(ys[0]), generator=generator)
        totals = []
        for start in range(0, len(permutation), config.batch_size):
            if cancelled():
                raise KeyboardInterrupt('Ablation cancelled before update.')
            indices = permutation[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(losses(xs, ys, indices)).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite gradient.')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite parameter.')
            updates += 1
            totals.append((loss.item(), len(indices)))
        history.append(math.fsum(a * n for a, n in totals) / len(ys[0]))
        model.eval()
        with torch.no_grad():
            value = float(torch.stack(losses(vx, vy)).mean())
        if not math.isfinite(value):
            raise RuntimeError('Nonfinite validation loss.')
        validation_history.append(value)
        if best_epoch is None or best - value > config.min_delta:
            best, best_epoch, stale = value, epoch, 0
            snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if early_stopping and stale >= config.patience:
                reason = 'early_stopping'
                break
    if early_stopping:
        model.load_state_dict(snapshot)
    model.eval().requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return {'epochs': len(history), 'best_validation_epoch': best_epoch,
            'selected_epoch': best_epoch if early_stopping else config.max_epochs,
            'checkpoint_rule': 'validation_min_delta' if early_stopping else 'last_epoch',
            'best_validation_ce_nats': best, 'epoch_training_ce_nats': history,
            'epoch_validation_ce_nats': validation_history, 'optimizer_updates': updates,
            'term_batch_evaluations': updates * len(xs), 'stop_reason': reason,
            'environment': active_environment}


def estimate_entropy(train, validation, test, *, config, alphabets, variant='native',
                     early_stopping=True, cancelled=lambda: False):
    """One atomic result: no successful state/report is published after failure."""
    if variant not in ('native', 'shared', 'sharing_control') or type(early_stopping) is not bool:
        raise ValueError('Unknown experimental policy.')
    if config.device != 'cpu':
        raise ValueError('Frozen ablations are CPU-only.')
    if [b.role for b in (train, validation, test)] != ['train', 'validation', 'test']:
        raise ValueError('Explicit train/validation/test batches required.')
    for i, a in enumerate((train, validation, test)):
        if (a.dataset_id, a.split_id, a.variable_ids, a.n_samples) != (
                train.dataset_id, train.split_id, train.variable_ids, train.n_samples):
            raise ValueError('Partition identities differ.')
        for b in (train, validation, test)[i + 1:]:
            if np.intersect1d(a.row_indices, b.row_indices).size:
                raise ValueError('Overlapping partitions.')
    config.validate_dataset(variable_ids=train.variable_ids)
    start = perf_counter()
    encoder = CategoricalEncoder.fit(train, policy='declared', alphabets=alphabets)
    order = config.component_order
    # Do not inspect TEST labels until all fitting and checkpoint selection end.
    for v in order:
        encoder.target_indices(validation, v)
    marginal = float(PlugInEstimator().estimate(train.values[:, train.variable_ids.index(order[0])]))
    terms, checkpoints, fitted = [], {}, []
    if variant == 'native':
        for i, target in enumerate(order[1:], 1):
            if cancelled():
                raise KeyboardInterrupt('Cancellation before conditional fit.')
            if early_stopping:
                result = train_conditional_term(train, validation, encoder=encoder, config=config,
                                                target=target, conditioning=order[:i])
                model = result.model
                diagnostic = {k: getattr(result, k) for k in (
                    'initialization_seed', 'shuffle_seed', 'epochs_trained', 'best_epoch',
                    'best_validation_ce_nats', 'epoch_training_ce_nats', 'epoch_validation_ce_nats',
                    'stop_reason', 'warnings', 'environment')}
                diagnostic.update(epochs=result.epochs_trained, selected_epoch=result.best_epoch,
                                  checkpoint_rule='validation_min_delta',
                                  optimizer_updates=result.epochs_trained * math.ceil(len(train.values) / config.batch_size),
                                  term_batch_evaluations=result.epochs_trained * math.ceil(len(train.values) / config.batch_size))
            else:
                tid = stream_ids(order, variant)[i - 1]
                seeds = {p: derive_term_seed(config.training_seed, tid, purpose=p) for p in ('initialization', 'shuffle')}
                with reproducible_execution(seeds['initialization'], mode=config.reproducibility_mode, device='cpu'):
                    x = [encoder.conditioning_features(train, order[:i], target=target)]
                    vx = [encoder.conditioning_features(validation, order[:i], target=target)]
                    model = ConditionalMLP(x[0].shape[1], encoder.cardinalities[target], hidden_sizes=config.hidden_sizes)
                    diagnostic = fit_arrays(model, x, [encoder.target_indices(train, target)], vx,
                                            [encoder.target_indices(validation, target)], config=config,
                                            shuffle_seed=seeds['shuffle'], early_stopping=False, cancelled=cancelled)
                diagnostic.update(initialization_seed=seeds['initialization'], shuffle_seed=seeds['shuffle'])
            fitted.append((target, order[:i], model, diagnostic))
    else:
        tid = stream_ids(order, variant)[0]
        seeds = {p: derive_term_seed(config.training_seed, tid, purpose=p) for p in ('initialization', 'shuffle')}
        with reproducible_execution(seeds['initialization'], mode=config.reproducibility_mode, device='cpu'):
            model = AblationMultiHead(len(order), encoder.cardinalities[order[0]],
                                     hidden_sizes=config.hidden_sizes, shared=variant == 'shared')
            diagnostic = fit_arrays(model,
                [fixed_slot_features(encoder, train, order, i) for i in range(1, len(order))],
                [encoder.target_indices(train, v) for v in order[1:]],
                [fixed_slot_features(encoder, validation, order, i) for i in range(1, len(order))],
                [encoder.target_indices(validation, v) for v in order[1:]],
                config=config, shuffle_seed=seeds['shuffle'], early_stopping=early_stopping,
                pooled=True, cancelled=cancelled)
        diagnostic.update(initialization_seed=seeds['initialization'], shuffle_seed=seeds['shuffle'])
        fitted = [(v, order[:i], model, diagnostic) for i, v in enumerate(order[1:], 1)]
    fit_seconds = perf_counter() - start
    evaluation = perf_counter()
    for i, (target, conditioning, model, diagnostic) in enumerate(fitted):
        if cancelled():
            raise KeyboardInterrupt('Cancellation before held-out evaluation.')
        targets = encoder.target_indices(test, target)
        if variant == 'native':
            features = encoder.conditioning_features(test, conditioning, target=target)
            value = _test_ce(model, features, targets, cardinality=encoder.cardinalities[target], config=config)
        else:
            features = fixed_slot_features(encoder, test, order, i + 1)
            totals = []
            with torch.inference_mode():
                for s in range(0, len(targets), config.batch_size):
                    logits = model(torch.from_numpy(features[s:s + config.batch_size]), i)
                    loss = nn.functional.cross_entropy(logits, torch.from_numpy(targets[s:s + config.batch_size]), reduction='none')
                    totals.append(loss.double().sum().item())
            value = math.fsum(totals) / len(targets)
        terms.append({'target': target, 'conditioning': conditioning, 'test_ce_nats': value,
                      'cost_scope': 'individual_term' if variant == 'native' else 'joint_model_repeated_per_head',
                      'model_id': f'model-{i}' if variant == 'native' else 'model-0', **diagnostic})
        if variant == 'native' or i == 0:
            checkpoints[f'model-{i}'] = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    unique = fitted if variant == 'native' else fitted[:1]
    report = {'entropy_nats': math.fsum([marginal, *[t['test_ce_nats'] for t in terms]]),
              'marginal_train_nats': marginal, 'component_order': order, 'terms': terms,
              'variant': variant, 'early_stopping': early_stopping,
              'fit_runtime_seconds': fit_seconds, 'inference_runtime_seconds': perf_counter() - evaluation,
              'parameter_count': sum(sum(p.numel() for p in m.parameters()) for _, _, m, _ in unique),
              'optimizer_updates': sum(t['optimizer_updates'] for _, _, _, t in unique),
              'term_batch_evaluations': sum(t['term_batch_evaluations'] for _, _, _, t in unique),
              'train_rows': train.row_indices.tolist(), 'validation_rows': validation.row_indices.tolist(),
              'test_rows': test.row_indices.tolist(), 'dataset_id': train.dataset_id, 'split_id': train.split_id}
    report['effective_seeds'] = [t[p] for _, _, _, t in unique for p in ('initialization_seed', 'shuffle_seed')]
    if not math.isfinite(report['entropy_nats']):
        raise RuntimeError('Nonfinite entropy reconstruction.')
    return report, checkpoints
