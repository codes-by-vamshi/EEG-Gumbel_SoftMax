# training.py
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from SelectionNet import SelectionNet
import config as cfg

try:
    from tqdm import trange
except Exception:  # pragma: no cover
    trange = range


def _build(N, T, K, n_classes, lr, device,
           use_subject_embedding=False, n_subjects=None, subject_embed_dim=0,
           use_subject_specific_selection=False):
    """Model + optimizer with the selection layer on a boosted LR."""
    model = SelectionNet(
        [N, T], K, output_dim=n_classes,
        use_subject_embedding=use_subject_embedding,
        n_subjects=n_subjects,
        subject_embed_dim=subject_embed_dim,
        use_subject_specific_selection=use_subject_specific_selection,
    ).to(device)
    model.set_freeze(False)
    sel_params = list(model.selection_layer.parameters())
    sel_ids = {id(p) for p in sel_params}
    net_params = [p for p in model.parameters() if id(p) not in sel_ids]
    opt = torch.optim.Adam([
        {'params': net_params, 'lr': lr},
        {'params': sel_params, 'lr': lr * cfg.FIXED['sel_lr_mult']},
    ])
    return model, opt


def _loader(X, y, subj=None, shuffle=True):
    X_t = torch.as_tensor(X).unsqueeze(1)                 # (n, 1, N, T)
    y_t = torch.as_tensor(y, dtype=torch.long)
    if subj is None:
        ds = torch.utils.data.TensorDataset(X_t, y_t)
    else:
        subj_t = torch.as_tensor(subj, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X_t, y_t, subj_t)
    return torch.utils.data.DataLoader(ds, batch_size=cfg.FIXED['batch_size'],
                                       shuffle=shuffle)

def _per_subject_acc(y_true, y_pred, subj_ids, n_subjects):
    out = np.full(int(n_subjects), np.nan, dtype=np.float32)
    for sid in range(int(n_subjects)):
        idx = np.flatnonzero(subj_ids == sid)
        if len(idx):
            out[sid] = float(np.mean(y_pred[idx] == y_true[idx]))
    return out


@torch.no_grad()
def _evaluate(model, X, y, device, subj=None):
    model.eval()
    Xt = torch.as_tensor(X).unsqueeze(1).to(device)
    yt = torch.as_tensor(y, dtype=torch.long).to(device)
    if subj is None:
        out = model(Xt)
    else:
        subjt = torch.as_tensor(subj, dtype=torch.long).to(device)
        out = model(Xt, subject_ids=subjt)
    loss = torch.nn.functional.cross_entropy(out, yt).item()
    pred = out.argmax(1)
    acc = (pred == yt).float().mean().item()
    return acc, loss, pred.cpu().numpy()

def train_one_fold(X_tr, y_tr, X_va, y_va, K, n_classes, hp,
                   temp_sched, thresh_sched, device, seed,
                   subj_tr=None, subj_va=None,
                   use_subject_embedding=False, n_subjects=None, subject_embed_dim=0,
                   use_subject_specific_selection=False,
                   subject_names_by_id=None):
    """Train one SelectionNet on a fold. Returns (val_acc, mean_entropy, n_unique)."""
    torch.manual_seed(seed)
    N, T = X_tr.shape[1], X_tr.shape[2]

    model, opt = _build(
        N, T, K, n_classes, hp['lr'], device,
        use_subject_embedding=use_subject_embedding,
        n_subjects=n_subjects,
        subject_embed_dim=subject_embed_dim,
        use_subject_specific_selection=use_subject_specific_selection,
    )
    ce = torch.nn.CrossEntropyLoss()
    tr_loader = _loader(X_tr, y_tr, subj=subj_tr)

    prev_val = 1e9
    patience = 0
    best_state = None

    epoch_iter = trange(cfg.MAX_EPOCHS, desc="epoch", leave=False)
    for epoch in epoch_iter:
        model.set_temperature(temp_sched[epoch].to(device))
        model.set_thresh(thresh_sched[epoch])

        # ---- train ----
        model.train()
        running = 0.0
        seen = 0
        for batch in tr_loader:
            if len(batch) == 2:
                data, labels = batch
                subj = None
            else:
                data, labels, subj = batch
                subj = subj.to(device)
            data, labels = data.to(device), labels.to(device)
            opt.zero_grad()
            sup = ce(model(data) if subj is None else model(data, subject_ids=subj), labels)
            reg = model.regularizer(hp['lamba'], hp['weight_decay'])
            loss = sup + reg
            loss.backward()
            opt.step()
            running += float(loss.detach().item()) * int(labels.shape[0])
            seen += int(labels.shape[0])

        # ---- validate ----
        val_acc, val_loss, va_pred = _evaluate(model, X_va, y_va, device, subj=subj_va)
        mean_H = model.monitor()[0].mean().item()
        tr_loss = running / max(seen, 1)

        postfix = {
            "tr_loss": f"{tr_loss:.3f}",
            "va_acc": f"{val_acc:.3f}",
            "va_loss": f"{val_loss:.3f}",
            "H": f"{mean_H:.3f}",
        }
        if subj_va is not None and n_subjects is not None:
            accs = _per_subject_acc(y_va, va_pred, subj_va, n_subjects)
            if subject_names_by_id is None:
                subj_str = " ".join(f"{i}:{a:.2f}" for i, a in enumerate(accs) if not np.isnan(a))
            else:
                subj_str = " ".join(
                    f"{subject_names_by_id[i]}:{float(accs[i]):.2f}"
                    for i in range(min(len(subject_names_by_id), len(accs)))
                    if not np.isnan(accs[i])
                )
            postfix["subj_va"] = subj_str
        try:
            epoch_iter.set_postfix(postfix)
        except Exception:
            pass

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
    val_acc, _, _ = _evaluate(model, X_va, y_va, device, subj=subj_va)
    H, s, _ = model.monitor()
    return val_acc, H.mean().item(), int(torch.unique(s - 1).numel())

def train_final(X_tr, y_tr, X_te, y_te, K, n_classes, hp,
                temp_sched, thresh_sched, device, seed,
                subj_tr=None, subj_te=None,
                use_subject_embedding=False, n_subjects=None, subject_embed_dim=0,
                use_subject_specific_selection=False,
                return_model=False):
    """Full-length training on trainval, no early stopping. Returns a dict (and optionally the model)."""
    torch.manual_seed(seed)
    N, T = X_tr.shape[1], X_tr.shape[2]

    model, opt = _build(
        N, T, K, n_classes, hp['lr'], device,
        use_subject_embedding=use_subject_embedding,
        n_subjects=n_subjects,
        subject_embed_dim=subject_embed_dim,
        use_subject_specific_selection=use_subject_specific_selection,
    )
    ce = torch.nn.CrossEntropyLoss()
    tr_loader = _loader(X_tr, y_tr, subj=subj_tr)

    epoch_iter = trange(cfg.MAX_EPOCHS, desc="epoch", leave=False)
    for epoch in epoch_iter:
        model.set_temperature(temp_sched[epoch].to(device))
        model.set_thresh(thresh_sched[epoch])
        model.train()
        running = 0.0
        seen = 0
        for batch in tr_loader:
            if len(batch) == 2:
                data, labels = batch
                subj = None
            else:
                data, labels, subj = batch
                subj = subj.to(device)
            data, labels = data.to(device), labels.to(device)
            opt.zero_grad()
            out = model(data) if subj is None else model(data, subject_ids=subj)
            loss = ce(out, labels) + model.regularizer(hp['lamba'], hp['weight_decay'])
            loss.backward()
            opt.step()
            running += float(loss.detach().item()) * int(labels.shape[0])
            seen += int(labels.shape[0])
        try:
            epoch_iter.set_postfix({"tr_loss": f"{(running / max(seen, 1)):.3f}"})
        except Exception:
            pass

    train_acc, _, _ = _evaluate(model, X_tr, y_tr, device, subj=subj_tr)
    test_acc, _, te_pred = _evaluate(model, X_te, y_te, device, subj=subj_te)

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
    }, model if return_model else None
