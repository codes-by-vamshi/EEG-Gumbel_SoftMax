# train_utils.py
from itertools import product
import numpy as np
import torch


def grid_combos(grid):
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def exp_decay(start, end, epochs, end_epoch):
    """Geometric decay from `start`, reaching `end` at `end_epoch`, flat after."""
    t = torch.arange(0.0, epochs)
    pw = torch.clamp(t / end_epoch, 0, 1)
    return start * torch.pow(torch.tensor(end / start), pw)


def standardize(X_fit, *others):
    """Per-channel z-score using X_fit's statistics only."""
    mean = X_fit.mean(axis=(0, 2), keepdims=True)          # (1, n_ch, 1)
    std = X_fit.std(axis=(0, 2), keepdims=True) + 1e-7
    out = [np.ascontiguousarray((X - mean) / std, dtype=np.float32)
           for X in (X_fit, *others)]
    return (*out, mean, std)


def print_confusion_matrix(cm, labels=None, prefix=''):
    if labels is None:
        labels = list(range(len(cm)))
    header = ' '.join(f'{l:^7}' for l in labels)
    print(f'{prefix}    CM   {header}')
    for label, row in zip(labels, cm):
        print(f'{prefix}       {label:^3} ' + ' '.join(f'{int(v):7d}' for v in row))