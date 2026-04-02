from pathlib import Path
import torch
import numpy as np

_COLAB_DRIVE = Path('/content/drive/MyDrive/DL_Final_Project')
_REPO_ROOT = Path(__file__).parent.parent

# DATA_DIR is in the repo itself so it is accessible after git clone
# without requiring Drive to be mounted.
DATA_DIR = _REPO_ROOT / 'data'

# Embeddings, results, and checkpoints live on Drive in Colab.
# Locally they fall back to outputs/ at the repo root (gitignored).
if _COLAB_DRIVE.exists():
    DRIVE_ROOT = _COLAB_DRIVE
else:
    DRIVE_ROOT = _REPO_ROOT / 'outputs'

EMBEDDING_DIR = DRIVE_ROOT / 'embeddings'
RESULTS_DIR = DRIVE_ROOT / 'results'
CHECKPOINT_DIR = DRIVE_ROOT / 'checkpoints'
FIGURES_DIR = RESULTS_DIR / 'figures'

# Device detection: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

ESM2_MODEL_NAME = 'esm2_t33_650M_UR50D'

ABAGYM_DATASETS = [
    'Ang2_2017_G6',
    'EGFR_2013_Cetuximab',
    'lysozyme_2019_D441',
    'VEGF_2017b_G6',
    'HER2_2021_trastuzumab',
]

N_MUTATIONS = 5318
N_SABDAB = 491

CDR_REGIONS = ['CDR_H1', 'CDR_H2', 'CDR_H3', 'CDR_L1', 'CDR_L2', 'CDR_L3']
FR_REGION = 'FR'


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
