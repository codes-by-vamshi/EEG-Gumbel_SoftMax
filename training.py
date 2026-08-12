# training.py
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from SelectionNet import SelectionNet
import config as cfg


def _build(N, T, K, n_classes, lr, device):
    """Model + optimizer with the selection layer on a boosted LR."""
    model = SelectionNet([N, T], K, output_dim=n_classes).to(device)
    model.set_freeze(False)
    sel_params = list(model.selection_layer.parameters())
    sel_ids = {id(p) for p in sel_params}
    net_params = [p for p in model.parameters() if id(p) not in sel_ids]
    opt = torch.optim.Adam([
        {'params': net_params, 'lr': lr},
        {'params': sel_params, 'lr': lr * cfg.FIXED['sel_lr_mult']},
    ])
    return model, opt


def _loader(X, y, shuffle=True):
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(X).unsqueeze(1),                  # (n, 1, N, T)
        torch.as_tensor(y, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=cfg.FIXED['batch_size'],
                                       shuffle=shuffle)


@torch.no_grad()
def _evaluate(model, X, y, device):
    model.eval()
    Xt = torch.as_tensor(X).unsqueeze(1).to(device)
    yt = torch.as_tensor(y, dtype=torch.long).to(device)
    out = model(Xt)
    loss = torch.nn.functional.cross_entropy(out, yt).item()
    pred = out.argmax(1)
    acc = (pred == yt).float().mean().item()
    return acc, loss, pred.cpu().numpy()

def train_one_fold(X_tr, y_tr, X_va, y_va, K, n_classes, hp,
                   temp_sched, thresh_sched, device, seed):
    """Train one SelectionNet on a fold. Returns (val_acc, mean_entropy, n_unique)."""
    torch.manual_seed(seed)
    N, T = X_tr.shape[1], X_tr.shape[2]

    model, opt = _build(N, T, K, n_classes, hp['lr'], device)
    ce = torch.nn.CrossEntropyLoss()
    tr_loader = _loader(X_tr, y_tr)

    prev_val = 1e9
    patience = 0
    best_state = None

    for epoch in range(cfg.MAX_EPOCHS):
        model.set_temperature(temp_sched[epoch].to(device))
        model.set_thresh(thresh_sched[epoch])

        # ---- train ----
        model.train()
        for data, labels in tr_loader:
            data, labels = data.to(device), labels.to(device)
            opt.zero_grad()
            sup = ce(model(data), labels)
            reg = model.regularizer(hp['lamba'], hp['weight_decay'])
            (sup + reg).backward()
            opt.step()

        # ---- validate ----
        val_acc, val_loss, _ = _evaluate(model, X_va, y_va, device)
        mean_H = model.monitor()[0].mean().item()

        # ---- early stopping: only once selection has converged ----
        if mean_H <= cfg.FIXED['entropy_lim'] and val_loss > prev_val - cfg.FIXED['stop_delta']:
            patience += 1
            if patience >= cfg.FIXED['patience']:
                break
        else:
            patience = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            prev_val = val_loss

    # restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- final fold metrics ----
    val_acc, _, _ = _evaluate(model, X_va, y_va, device)
    H, s, _ = model.monitor()
    return val_acc, H.mean().item(), int(torch.unique(s - 1).numel())

def train_final(X_tr, y_tr, X_te, y_te, K, n_classes, hp,
                temp_sched, thresh_sched, device, seed):
    """Full-length training on trainval, no early stopping. Returns a dict."""
    torch.manual_seed(seed)
    N, T = X_tr.shape[1], X_tr.shape[2]

    model, opt = _build(N, T, K, n_classes, hp['lr'], device)
    ce = torch.nn.CrossEntropyLoss()
    tr_loader = _loader(X_tr, y_tr)

    for epoch in range(cfg.MAX_EPOCHS):
        model.set_temperature(temp_sched[epoch].to(device))
        model.set_thresh(thresh_sched[epoch])
        model.train()
        for data, labels in tr_loader:
            data, labels = data.to(device), labels.to(device)
            opt.zero_grad()
            loss = ce(model(data), labels) + model.regularizer(hp['lamba'],
                                                               hp['weight_decay'])
            loss.backward()
            opt.step()

    train_acc, _, _ = _evaluate(model, X_tr, y_tr, device)
    test_acc, _, te_pred = _evaluate(model, X_te, y_te, device)

    H, sel, _ = model.monitor()
    channels = sorted(int(c) for c in torch.unique(sel - 1).cpu().numpy())
    cm = confusion_matrix(y_te, te_pred, labels=list(range(n_classes)))

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'selected_channels': channels,
        'n_unique': len(channels),
        'mean_entropy': H.mean().item(),
        'confusion_matrix': cm.tolist(),
    }