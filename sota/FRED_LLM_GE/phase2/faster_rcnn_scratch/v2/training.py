"""Checkpoint-safe SGD at exact batch boundaries; V1 callers stay unchanged."""
import random
import time

import numpy as np
import torch


class Paused(Exception):
    """A safe checkpoint was written; request another allocation."""


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def train(model, data, args, *, resume_state=None, progress_callback,
          should_stop=lambda: False, checkpoint_steps=200, checkpoint_seconds=300,
          prepare=None, clock=time.monotonic):
    if prepare is None:
        from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.train_eval import prepare_batch
        prepare = prepare_batch
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    size = len(data['images'])
    if not size:
        raise ValueError('Empty training partition')
    epoch, order, cursor, steps = 0, None, 0, 0
    prior_seconds = 0.0
    last_loss = None
    if resume_state is not None:
        if resume_state.get('schema_version') != 'batch-sgd-v2':
            raise ValueError('Incompatible batch checkpoint')
        epoch, order, cursor, steps = (resume_state[k] for k in ('epoch', 'order', 'next_position', 'steps'))
        if not 0 <= epoch <= args.epochs or not 0 <= cursor <= size:
            raise ValueError('Invalid checkpoint cursor')
        if order is not None and sorted(order) != list(range(size)):
            raise ValueError('Checkpoint order differs from training partition')
        if order is None and cursor:
            raise ValueError('Cursor without sample order')
        model.load_state_dict(resume_state['state_dict'], strict=True)
        optimizer.load_state_dict(resume_state['optimizer'])
        # Optimizer tensors must follow GPU model parameters after CPU deserialization.
        for parameter, values in optimizer.state.items():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(parameter.device)
        restore_rng(resume_state['rng'])
        prior_seconds = resume_state['training_seconds']
        last_loss = resume_state.get('last_loss')
    model.train()
    started = last_saved = clock()
    last_steps = steps

    def save():
        nonlocal last_saved, last_steps
        state = {'schema_version': 'batch-sgd-v2', 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'completed_epochs': epoch, 'order': order, 'next_position': cursor,
                 'steps': steps, 'rng': capture_rng(),
                 'last_loss': last_loss,
                 'training_seconds': prior_seconds + clock() - started}
        progress_callback(state)
        last_saved, last_steps = clock(), steps

    if resume_state is None:
        save()  # A timeout before the first update still resumes this initialization.
    while epoch < args.epochs:
        if order is None:
            order = torch.randperm(size).tolist()
            cursor = 0
        if should_stop():
            save()
            raise Paused('Checkpoint saved before shutdown')
        while cursor < size:
            indices = order[cursor:cursor + args.batch_size]
            batch = prepare(model, data, indices, True)
            optimizer.zero_grad()
            loss, _ = model.parse_losses(model(**batch, mode='loss'))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Non-finite gradients')
            optimizer.step()
            last_loss = loss.item()
            cursor += len(indices)
            steps += 1
            if should_stop() or steps - last_steps >= checkpoint_steps or clock() - last_saved >= checkpoint_seconds:
                save()
            if should_stop():
                raise Paused('Checkpoint saved at batch boundary')
        epoch += 1
        order, cursor = None, 0
        print(f'epoch {epoch}/{args.epochs}: final batch loss={last_loss:.5f}', flush=True)
        save()
    return {'steps': steps, 'training_seconds': prior_seconds + clock() - started}
