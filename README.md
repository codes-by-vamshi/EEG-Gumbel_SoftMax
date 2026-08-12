# Gumbel-Softmax Channel Selection for EEG (SelectionNet)

This repo trains an EEG classifier with a **learned channel-selection layer** (Gumbel-Softmax / Concrete distribution) in front of an **MSFBCNN** backbone.

The main entrypoint is `gumbel-softmax.py`, which:
- loads `X`/`y` from `.npy` files (single dataset or multi-subject folder),
- runs **Stratified K-fold CV** to pick hyperparameters (cached per split),
- trains a final model and saves selected channels + metrics to `results/`.

## Data format

`X` must be a NumPy array with shape:
- `(n_trials, n_channels, n_times)`

`y` must be a 1D label array with length `n_trials`.

Two supported layouts:

1) **Single dataset (explicit paths)**
```
X.npy
y.npy
```

2) **Multi-subject root**
```
DATA_ROOT/
  SubjectA/
    X.npy
    y.npy
  SubjectB/
    X.npy
    y.npy
```

## Setup

Create an environment and install dependencies (CPU or CUDA build of PyTorch as appropriate for your machine):
```
pip install numpy scipy scikit-learn torch
```

## Run

1) Edit `config.py` to point to your data:
- Set **either** `DATA_ROOT` (multi-subject) **or** `X_PATH` + `Y_PATH` (single dataset).

2) Start training:
```
python gumbel-softmax.py
```

If you don’t have a GPU, set `device = "cpu"` in `config.py` (the code also falls back to CPU if CUDA is unavailable).

## Configuration notes (`config.py`)

- `DEFAULT_K`: number of channels to select (`M`).
- `SUBJECTS_BUDGET`: optional per-subject `K` override when using `DATA_ROOT`.
- `EXPECT_CHANNELS`: set to the expected number of channels to catch transposed inputs early.
- `SPLIT_SEED`: controls train/test split + CV folds (kept constant so CV cache is reusable).
- `INIT_SEED`: controls weight initialization for the final training run.
- `GRID`: hyperparameter grid searched in CV (`lr`, `lamba`, `weight_decay`).
- `OUT_DIR`: output directory (default: `results`).

## Outputs

Runs write JSON files to `results/`:
- `cv_<dataset>_M<K>_split<SPLIT_SEED>.json`: CV results (cached per split).
- `run_<dataset>_M<K>_split<SPLIT_SEED>_init<INIT_SEED>.json`: final run metrics.

The final run JSON includes:
- `train_acc`, `test_acc`
- `selected_channels` (0-based channel indices)
- `confusion_matrix`
- selection diagnostics like `mean_entropy` and `n_unique`

## Repo layout

- `gumbel-softmax.py`: CLI entrypoint (loading, CV, final training, saving outputs)
- `config.py`: all experiment configuration
- `training.py`: fold training + final training routines
- `SelectionNet.py`: SelectionNet (Gumbel-Softmax selection layer + MSFBCNN)
- `train_utils.py`: schedules, standardization, grid helpers

