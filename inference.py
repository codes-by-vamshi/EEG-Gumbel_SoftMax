import argparse
from pathlib import Path

import numpy as np
import torch

from SelectionNet import SelectionNet


def _load_checkpoint(path, map_location):
    ckpt = torch.load(path, map_location=map_location)
    if 'model_state_dict' not in ckpt or 'model_kwargs' not in ckpt:
        raise ValueError("checkpoint missing required keys: model_state_dict/model_kwargs")
    return ckpt


@torch.no_grad()
def predict(ckpt_path, x_path, device=None, subject=None, subject_id=None,
            subject_ids_npy=None, batch_size=64):
    ckpt_path = Path(ckpt_path)
    x_path = Path(x_path)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = _load_checkpoint(str(ckpt_path), map_location='cpu')
    kwargs = ckpt['model_kwargs']

    model = SelectionNet(
        kwargs['input_dim'],
        kwargs['M'],
        output_dim=kwargs['output_dim'],
        use_subject_embedding=kwargs.get('use_subject_embedding', False),
        n_subjects=kwargs.get('n_subjects', None),
        subject_embed_dim=kwargs.get('subject_embed_dim', 0),
        use_subject_specific_selection=kwargs.get('use_subject_specific_selection', False),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    X = np.load(x_path).astype(np.float32)
    if X.ndim != 3:
        raise ValueError(f"X must be (trials, channels, times); got {X.shape}")

    mean = ckpt.get('standardize_mean', None)
    std = ckpt.get('standardize_std', None)
    if mean is not None and std is not None:
        X = np.ascontiguousarray((X - mean) / (std + 1e-12), dtype=np.float32)

    n = X.shape[0]
    uses_subject = bool(kwargs.get('use_subject_embedding', False))
    subject_to_id = ckpt.get('subject_to_id', None) or {}

    if uses_subject:
        if subject_ids_npy is not None:
            subject_ids = np.load(subject_ids_npy).astype(np.int64).ravel()
            if len(subject_ids) != n:
                raise ValueError(f"subject_ids has length {len(subject_ids)} but X has {n} trials")
        else:
            if subject_id is None:
                if subject is None:
                    raise ValueError("model requires subject id: pass --subject or --subject-id or --subject-ids-npy")
                if subject not in subject_to_id:
                    raise ValueError(f"unknown subject {subject!r}; available: {sorted(subject_to_id.keys())}")
                subject_id = int(subject_to_id[subject])
            subject_ids = np.full(n, int(subject_id), dtype=np.int64)
        subject_ids_t = torch.as_tensor(subject_ids, dtype=torch.long)
    else:
        subject_ids_t = None

    Xt = torch.as_tensor(X).unsqueeze(1)  # (n, 1, N, T)
    preds = []
    logits_all = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = Xt[start:end].to(device)
        if subject_ids_t is None:
            out = model(xb)
        else:
            sb = subject_ids_t[start:end].to(device)
            out = model(xb, subject_ids=sb)
        logits_all.append(out.cpu())
        preds.append(out.argmax(1).cpu())

    logits = torch.cat(logits_all, dim=0).numpy()
    pred_idx = torch.cat(preds, dim=0).numpy().astype(np.int64)

    classes = ckpt.get('classes', None)
    if classes is not None:
        classes = np.asarray(classes)
        pred_labels = classes[pred_idx]
    else:
        pred_labels = pred_idx

    return pred_idx, pred_labels, logits


def main():
    p = argparse.ArgumentParser(description="Inference for SelectionNet checkpoints.")
    p.add_argument("--ckpt", required=True, help="Path to .pt checkpoint saved by training.")
    p.add_argument("--x", required=True, help="Path to X.npy (trials, channels, times).")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto).")
    p.add_argument("--subject", default=None, help="Subject name (uses subject_to_id mapping in ckpt).")
    p.add_argument("--subject-id", type=int, default=None, help="Integer subject id.")
    p.add_argument("--subject-ids-npy", default=None, help="Optional .npy with per-trial subject ids.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-preds", default=None, help="Optional path to save predicted labels as .npy.")
    p.add_argument("--out-logits", default=None, help="Optional path to save logits as .npy.")
    args = p.parse_args()

    pred_idx, pred_labels, logits = predict(
        ckpt_path=args.ckpt,
        x_path=args.x,
        device=args.device,
        subject=args.subject,
        subject_id=args.subject_id,
        subject_ids_npy=args.subject_ids_npy,
        batch_size=args.batch_size,
    )

    print(f"n={len(pred_idx)} | pred_idx unique={np.unique(pred_idx).tolist()}")
    if args.out_preds:
        np.save(args.out_preds, pred_labels)
    if args.out_logits:
        np.save(args.out_logits, logits)


if __name__ == "__main__":
    main()

