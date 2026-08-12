from pathlib import Path
import numpy as np
import config as cfg
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from train_utils import exp_decay, standardize, grid_combos, print_confusion_matrix
import json
from training import train_one_fold, train_final

def load_subject(root, subject):
    """Format 2: root/<subject>/X.npy, y.npy"""
    d = Path(root) / subject
    return load_xy(d / "X.npy", d / "y.npy")

def load_subject_raw(root, subject):
    """Format 2: root/<subject>/X.npy, y.npy (no label remap)."""
    d = Path(root) / subject
    X = np.load(d / "X.npy").astype(np.float32)
    y = np.load(d / "y.npy").ravel()
    return X, y


def load_xy(x_path, y_path):
    """Format 1: explicit paths to X.npy and y.npy"""
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).ravel()
    _, y = np.unique(y, return_inverse=True)   # 1/2 or -1/1 -> 0/1
    return X, y.astype(np.int64)


def list_subjects(root):
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if (p / "X.npy").is_file())

def check_data(X, y, K, subject=None):
    """Fail loudly before anything expensive starts."""
    who = subject or 'data'
    if X.ndim != 3:
        raise ValueError(f"{who}: X must be (trials, channels, times), got {X.shape}")
    n, N, T = X.shape
    if len(y) != n:
        raise ValueError(f"{who}: X has {n} trials but y has {len(y)}")
    if cfg.EXPECT_CHANNELS and N != cfg.EXPECT_CHANNELS:
        raise ValueError(f"{who}: got {N} channels, expected {cfg.EXPECT_CHANNELS} "
                         f"(X={X.shape} — transposed?)")
    if N > T:
        print(f"WARNING {who}: channels({N}) > times({T}) — is X transposed?")
    if T < 90:
        raise ValueError(f"{who}: T={T} too short; MSFBCNN pools with kernel 75, need T>=90")
    if K > N:
        raise ValueError(f"{who}: K={K} exceeds n_channels={N}")

def out_paths(subject, K):
    """(cv_cache, result) — CV is keyed by split only, so inits share it."""
    Path(cfg.OUT_DIR).mkdir(parents=True, exist_ok=True)
    base = (f'{subject}_M{K}_split{cfg.SPLIT_SEED}' if subject
        else f'{cfg.DATASET_TAG}_M{K}_split{cfg.SPLIT_SEED}')
    return (Path(cfg.OUT_DIR) / f'cv_{base}.json',
            Path(cfg.OUT_DIR) / f'run_{base}_init{cfg.INIT_SEED}.json')

def _subjectwise_split_and_concat(loaded, test_size, seed):
    """
    Per-subject stratified train/test split, then concat across subjects.

    loaded: list[(subject_name, X, y_mapped)]
    returns: X_trainval, y_trainval, subj_trainval, X_test, y_test, subj_test, subject_to_id
    """
    subject_to_id = {s: i for i, (s, _, _) in enumerate(loaded)}
    X_tr_parts, y_tr_parts, subj_tr_parts = [], [], []
    X_te_parts, y_te_parts, subj_te_parts = [], [], []

    for subject, X, y in loaded:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        sid = subject_to_id[subject]
        X_tr_parts.append(X_trainval)
        y_tr_parts.append(y_trainval)
        subj_tr_parts.append(np.full(len(y_trainval), sid, dtype=np.int64))
        X_te_parts.append(X_test)
        y_te_parts.append(y_test)
        subj_te_parts.append(np.full(len(y_test), sid, dtype=np.int64))

    return (
        np.concatenate(X_tr_parts, axis=0),
        np.concatenate(y_tr_parts, axis=0),
        np.concatenate(subj_tr_parts, axis=0),
        np.concatenate(X_te_parts, axis=0),
        np.concatenate(y_te_parts, axis=0),
        np.concatenate(subj_te_parts, axis=0),
        subject_to_id,
    )

def _subjectwise_cv_splits(y_trainval, subj_trainval, n_splits, seed):
    """
    Subject-wise StratifiedKFold:
    - split within each subject by class
    - merge corresponding folds across subjects
    """
    fold_tr, fold_va = [[] for _ in range(n_splits)], [[] for _ in range(n_splits)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for sid in np.unique(subj_trainval):
        idx = np.flatnonzero(subj_trainval == sid)
        y_s = y_trainval[idx]
        for fold, (tr_rel, va_rel) in enumerate(skf.split(idx, y_s)):
            fold_tr[fold].append(idx[tr_rel])
            fold_va[fold].append(idx[va_rel])

    for fold in range(n_splits):
        yield np.concatenate(fold_tr[fold]), np.concatenate(fold_va[fold])

def process_multi_subjects(loaded_raw, K_channels, subject='ALL'):
    """
    Train a single global model on all subjects:
    - global label mapping across subjects
    - per-subject stratified split, then concat
    - subject-wise stratified CV folds, then concat
    """
    log = f'[{subject}] '
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # Ensure shapes are compatible across subjects
    dims = {(X.shape[1], X.shape[2]) for _, X, _ in loaded_raw}
    if len(dims) != 1:
        raise ValueError(f"{subject}: all subjects must share (channels, times); got {sorted(dims)}")

    # Global label mapping so class ids are consistent across subjects
    all_y = np.concatenate([y for _, _, y in loaded_raw], axis=0)
    classes, _ = np.unique(all_y, return_inverse=True)
    loaded = []
    for subject_name, X, y in loaded_raw:
        y_mapped = np.searchsorted(classes, y)
        loaded.append((subject_name, X, y_mapped.astype(np.int64)))

    n_classes = len(classes)
    cv_path, result_path = out_paths(subject, K_channels)

    (X_trainval, y_trainval, subj_trainval,
     X_test, y_test, subj_test,
     subject_to_id) = _subjectwise_split_and_concat(
        loaded, test_size=cfg.TEST_SIZE, seed=cfg.SPLIT_SEED
    )

    temp_sched = exp_decay(cfg.FIXED['start_temp'], cfg.FIXED['end_temp'],
                           cfg.MAX_EPOCHS, cfg.ANNEAL_EPOCHS)
    thresh_sched = exp_decay(cfg.THRESH_START, cfg.THRESH_END,
                             cfg.MAX_EPOCHS, cfg.ANNEAL_EPOCHS)

    # ---------------- CV: hyperparameter search (cached per split) ----------
    if cv_path.exists():
        print(f'{log}loading cached CV from {cv_path.name}')
        cached = json.load(open(cv_path))
        all_results, best = cached['all_results'], cached['best']
        valid = None
    else:
        all_results = []
        for hp in grid_combos(cfg.GRID):
            print(f'{log}===== combo: {hp} =====')
            accs, entropies, uniques = [], [], []

            for fold, (tr_idx, va_idx) in enumerate(
                _subjectwise_cv_splits(y_trainval, subj_trainval, cfg.N_FOLDS, cfg.SPLIT_SEED)
            ):
                X_tr, X_va, _, _ = standardize(X_trainval[tr_idx], X_trainval[va_idx])
                y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]
                subj_tr, subj_va = subj_trainval[tr_idx], subj_trainval[va_idx]

                val_acc, mean_H, n_unique = train_one_fold(
                    X_tr, y_tr, X_va, y_va,
                    K=K_channels, n_classes=n_classes, hp=hp,
                    temp_sched=temp_sched, thresh_sched=thresh_sched,
                    device=device, seed=cfg.SPLIT_SEED,      # NOT init seed
                    subj_tr=subj_tr, subj_va=subj_va,
                    use_subject_embedding=True,
                    n_subjects=len(subject_to_id),
                    subject_embed_dim=cfg.SUBJECT_EMBED_DIM,
                )
                accs.append(val_acc)
                entropies.append(mean_H)
                uniques.append(n_unique)
                print(f'{log}  fold {fold}: val_acc={val_acc:.4f} '
                      f'H={mean_H:.4f} unique={n_unique}/{K_channels}')

            all_results.append({
                'hp': hp,
                'mean_val_acc': float(np.mean(accs)),
                'std_val_acc': float(np.std(accs)),
                'mean_entropy': float(np.mean(entropies)),
                'mean_unique': float(np.mean(uniques)),
            })
            print(f'{log}  MEAN val_acc={np.mean(accs):.4f} ± {np.std(accs):.4f} '
                  f'| mean_unique={np.mean(uniques):.1f}')

        valid = [r for r in all_results
                 if r['mean_entropy'] <= cfg.FIXED['entropy_lim']
                 and r['mean_unique'] >= K_channels - 1]
        best = max(valid or all_results, key=lambda r: r['mean_val_acc'])

        json.dump({
            'all_results': all_results,
            'best': best,
            'K': K_channels,
            'subject': subject,
            'subjects': [s for s, _, _ in loaded],
            'subject_to_id': subject_to_id,
            'classes': classes.tolist(),
            'split_seed': cfg.SPLIT_SEED,
            'fixed': cfg.FIXED,
            'grid': cfg.GRID,
            'max_epochs': cfg.MAX_EPOCHS,
            'anneal_epochs': cfg.ANNEAL_EPOCHS,
        }, open(cv_path, 'w'), indent=2)

    print(f'{log}BEST hp: {best["hp"]}')
    print(f'{log}  val_acc {best["mean_val_acc"]:.3f} ± {best["std_val_acc"]:.3f} '
          f'| unique {best["mean_unique"]:.1f} | H {best["mean_entropy"]:.3f}')
    if valid is not None and not valid:
        print(f'{log}  WARNING: no combo met entropy/unique criteria — '
              f'selection may not have converged')

    # ---------------- final model at INIT_SEED ----------------------------
    Xtr_full, Xte_full, _, _ = standardize(X_trainval, X_test)   # train stats only

    res = train_final(
        Xtr_full, y_trainval, Xte_full, y_test,
        K=K_channels, n_classes=n_classes, hp=best['hp'],
        temp_sched=temp_sched, thresh_sched=thresh_sched,
        device=device, seed=cfg.INIT_SEED,
        subj_tr=subj_trainval, subj_te=subj_test,
        use_subject_embedding=True,
        n_subjects=len(subject_to_id),
        subject_embed_dim=cfg.SUBJECT_EMBED_DIM,
    )

    print(f'{log}[test] init {cfg.INIT_SEED}: train_acc={res["train_acc"]:.3f} '
          f'test_acc={res["test_acc"]:.3f} unique={res["n_unique"]} '
          f'meanH={res["mean_entropy"]:.3f}')
    print_confusion_matrix(res['confusion_matrix'], labels=list(range(n_classes)), prefix=log)
    print(f'{log}selected channels: {res["selected_channels"]}')

    result = {
        'subject': subject,
        'subjects': [s for s, _, _ in loaded],
        'subject_to_id': subject_to_id,
        'classes': classes.tolist(),
        'K': K_channels,
        'split_seed': cfg.SPLIT_SEED,
        'init_seed': cfg.INIT_SEED,
        'best_hp': best['hp'],
        'cv_val_acc_mean': best['mean_val_acc'],
        'cv_val_acc_std': best['std_val_acc'],
        'n_train': int(len(y_trainval)),
        'n_test': int(len(y_test)),
        'n_subjects': int(len(subject_to_id)),
        **res,
    }
    json.dump(result, open(result_path, 'w'), indent=2)
    print(f'{log}saved -> {result_path.name}\n')
    return result

def process_data(X, y, K_channels, subject=None):
    log = f'[{subject}] ' if subject else ''
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    n_classes = len(np.unique(y))
    cv_path, result_path = out_paths(subject, K_channels)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.SPLIT_SEED)

    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True,
                          random_state=cfg.SPLIT_SEED)

    temp_sched = exp_decay(cfg.FIXED['start_temp'], cfg.FIXED['end_temp'],
                           cfg.MAX_EPOCHS, cfg.ANNEAL_EPOCHS)
    thresh_sched = exp_decay(cfg.THRESH_START, cfg.THRESH_END,
                             cfg.MAX_EPOCHS, cfg.ANNEAL_EPOCHS)

    # ---------------- CV: hyperparameter search (cached per split) ----------
    if cv_path.exists():
        print(f'{log}loading cached CV from {cv_path.name}')
        cached = json.load(open(cv_path))
        all_results, best = cached['all_results'], cached['best']
        valid = None
    else:
        all_results = []
        for hp in grid_combos(cfg.GRID):
            print(f'{log}===== combo: {hp} =====')
            accs, entropies, uniques = [], [], []

            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval)):
                X_tr, X_va, _, _ = standardize(X_trainval[tr_idx], X_trainval[va_idx])
                y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]

                val_acc, mean_H, n_unique = train_one_fold(
                    X_tr, y_tr, X_va, y_va,
                    K=K_channels, n_classes=n_classes, hp=hp,
                    temp_sched=temp_sched, thresh_sched=thresh_sched,
                    device=device, seed=cfg.SPLIT_SEED,      # NOT init seed
                )
                accs.append(val_acc)
                entropies.append(mean_H)
                uniques.append(n_unique)
                print(f'{log}  fold {fold}: val_acc={val_acc:.4f} '
                      f'H={mean_H:.4f} unique={n_unique}/{K_channels}')

            all_results.append({
                'hp': hp,
                'mean_val_acc': float(np.mean(accs)),
                'std_val_acc': float(np.std(accs)),
                'mean_entropy': float(np.mean(entropies)),
                'mean_unique': float(np.mean(uniques)),
            })
            print(f'{log}  MEAN val_acc={np.mean(accs):.4f} ± {np.std(accs):.4f} '
                  f'| mean_unique={np.mean(uniques):.1f}')

        valid = [r for r in all_results
                 if r['mean_entropy'] <= cfg.FIXED['entropy_lim']
                 and r['mean_unique'] >= K_channels - 1]
        best = max(valid or all_results, key=lambda r: r['mean_val_acc'])

        json.dump({'all_results': all_results, 'best': best, 'K': K_channels,
                   'subject': subject, 'split_seed': cfg.SPLIT_SEED,
                   'fixed': cfg.FIXED, 'grid': cfg.GRID,
                   'max_epochs': cfg.MAX_EPOCHS, 'anneal_epochs': cfg.ANNEAL_EPOCHS},
                  open(cv_path, 'w'), indent=2)

    print(f'{log}BEST hp: {best["hp"]}')
    print(f'{log}  val_acc {best["mean_val_acc"]:.3f} ± {best["std_val_acc"]:.3f} '
          f'| unique {best["mean_unique"]:.1f} | H {best["mean_entropy"]:.3f}')
    if valid is not None and not valid:
        print(f'{log}  WARNING: no combo met entropy/unique criteria — '
              f'selection may not have converged')

    # ---------------- final model at INIT_SEED ----------------------------
    Xtr_full, Xte_full, _, _ = standardize(X_trainval, X_test)   # train stats only

    res = train_final(
        Xtr_full, y_trainval, Xte_full, y_test,
        K=K_channels, n_classes=n_classes, hp=best['hp'],
        temp_sched=temp_sched, thresh_sched=thresh_sched,
        device=device, seed=cfg.INIT_SEED,
    )

    print(f'{log}[test] init {cfg.INIT_SEED}: train_acc={res["train_acc"]:.3f} '
          f'test_acc={res["test_acc"]:.3f} unique={res["n_unique"]} '
          f'meanH={res["mean_entropy"]:.3f}')
    print_confusion_matrix(res['confusion_matrix'], labels=list(range(n_classes)), prefix=log)
    print(f'{log}selected channels: {res["selected_channels"]}')

    result = {
        'subject': subject,
        'K': K_channels,
        'split_seed': cfg.SPLIT_SEED,
        'init_seed': cfg.INIT_SEED,
        'best_hp': best['hp'],
        'cv_val_acc_mean': best['mean_val_acc'],
        'cv_val_acc_std': best['std_val_acc'],
        'n_train': int(len(y_trainval)),
        'n_test': int(len(y_test)),
        **res,
    }
    json.dump(result, open(result_path, 'w'), indent=2)
    print(f'{log}saved -> {result_path.name}\n')
    return result

if __name__ == "__main__":
    if cfg.DATA_ROOT and cfg.X_PATH:
        raise ValueError("set DATA_ROOT or X_PATH/Y_PATH, not both")
    if not cfg.DATA_ROOT and not (cfg.X_PATH and cfg.Y_PATH):
        raise ValueError("set DATA_ROOT, or both X_PATH and Y_PATH")

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    torch.manual_seed(cfg.INIT_SEED)
    torch.cuda.manual_seed_all(cfg.INIT_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if cfg.DATA_ROOT is not None:
        subjects = list_subjects(cfg.DATA_ROOT)
        if cfg.use_subject_embedding:
            loaded_raw = []
            for subject in subjects:  # validate everything first
                X, y = load_subject_raw(cfg.DATA_ROOT, subject)
                check_data(X, y, cfg.DEFAULT_K, subject=subject)
                loaded_raw.append((subject, X, y))
            print(f"validated {len(loaded_raw)} subjects: {[s for s, _, _ in loaded_raw]}\n")
            process_multi_subjects(loaded_raw, cfg.DEFAULT_K, subject='ALL')
        else:
            loaded = []
            for subject in subjects:  # validate everything first
                X, y = load_subject(cfg.DATA_ROOT, subject)
                K = cfg.SUBJECTS_BUDGET.get(subject, cfg.DEFAULT_K)
                check_data(X, y, K, subject=subject)
                loaded.append((subject, K))
            print(f"validated {len(loaded)} subjects: {[s for s, _ in loaded]}\n")
            for subject, K in loaded:
                X, y = load_subject(cfg.DATA_ROOT, subject)
                process_data(X, y, K, subject=subject)
    else:
        X, y = load_xy(cfg.X_PATH, cfg.Y_PATH)
        process_data(X, y, cfg.DEFAULT_K)
