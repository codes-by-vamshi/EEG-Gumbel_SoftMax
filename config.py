# For multi-subject data else make it None
DATA_ROOT = None # '/home/rajesh/Linear_models/mw_data/Linear_models/Pratheeksha'

# For single source data
X_PATH = None
Y_PATH = None

# Identifies the dataset in output filenames when using X_PATH/Y_PATH.
DATASET_TAG = 'exp1'

# Expected channel count; None to skip the check. Catches transposed X.
EXPECT_CHANNELS = 118

# Set True when using subject embeddings (i.e. multi-subject data)
use_subject_embedding = False
SUBJECT_EMBED_DIM = 4
# When using subject embeddings, optionally learn a separate selector (qz_loga) per subject.
# If False, the selector is shared across all subjects (one global set of selected channels).
use_subject_specific_selection = False

# Channel Budget for the Gumbel-Softmax model
DEFAULT_K = 8
# Make sure to match subject names with the folder names in the data root.
SUBJECTS_BUDGET = {
    'A2': 7, 'A3': 5, 'A4': 18, 'A5': 8, 'A8': 6,
    'A9': 8, 'A10': 16, 'A14': 4, 'A16': 12, 'A17': 7,
}

# Device
device = 'cuda:0'

# ---- seeds ---------------------------------------------------------------
SPLIT_SEED = 42      # test holdout + CV folds. Hold constant to reuse one split.
INIT_SEED  = 0       # weight init only. Vary this for initialization variance.

# ---- split ---------------------------------------------------------------
TEST_SIZE = 0.20
N_FOLDS   = 5

# ---- training ------------------------------------------------------------
FIXED = {
    'batch_size' : 4,
    'start_temp' : 10.0,
    'end_temp'   : 0.02,
    'sel_lr_mult': 40.0,
    'patience'   : 10,
    'stop_delta' : 1e-3,
    'entropy_lim': 0.05,
}

THRESH_START = 3.0
THRESH_END   = 1.1

GRID = {
    'lamba'        : [0.1, 0.3, 0.5],
    'lr'           : [1e-4, 5e-4],
    'weight_decay' : [5e-4],
}

# ---- schedule ------------------------------------------------------------
MAX_EPOCHS    = 30   # how long training runs
ANNEAL_EPOCHS = 30   # where temp/thresh finish decaying

# ---- output --------------------------------------------------------------
OUT_DIR = 'results'
