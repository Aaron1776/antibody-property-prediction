# Contributing

## Collaboration Loop

Edit `.py` files locally or in your preferred editor, push to GitHub, then pull in Colab before running. All modeling logic lives in `src/`. Notebooks are thin orchestration layers that import from `src/` -- keep it that way.

```bash
# After editing src/ files locally:
git add src/
git commit -m "your message"
git push origin implementation

# In Colab, at the start of each session:
!git -C /content/antibody-property-prediction pull origin implementation
```

## Branch Structure

- `main`: stable, reflects the last clean state
- `implementation`: all active development happens here

Never push directly to `main`.

## Google Drive Setup

Both collaborators use a shared Drive folder. The non-owner (Lucas) should add a shortcut from "Shared with me" to My Drive root so both use the same path.

Both collaborators set:
```
DRIVE_ROOT = Path('/content/drive/MyDrive/DL_Final_Project')
```

Drive folder structure:
```
/MyDrive/DL_Final_Project/
    data/
        abagym_antibody.csv
        abagym_sequences.csv
        sabdab_affinity.csv
    embeddings/
        # sequence-level (5318, 2560) and (5318, 960):
        esm2_abagym.pt, esm2_abagym_index.json
        esm2_abagym_wildtype.pt, esm2_abagym_wildtype_index.json
        esm2_abagym_delta.pt, esm2_abagym_delta_index.json
        esm2_sabdab.pt, esm2_sabdab_index.json
        ablang2_abagym.pt, ablang2_abagym_index.json
        ablang2_abagym_wildtype.pt, ablang2_abagym_wildtype_index.json
        ablang2_abagym_delta.pt, ablang2_abagym_delta_index.json
        ablang2_sabdab.pt, ablang2_sabdab_index.json
        # residue-level (to be generated):
        esm2_abagym_residue_mutsite.pt   (5318, 1280)
        esm2_abagym_residue_wtsite.pt    (5318, 1280)
        ablang2_abagym_residue_mutsite.pt (5318, 480)
        ablang2_abagym_residue_wtsite.pt  (5318, 480)
    results/
        figures/
    checkpoints/
    wandb/
```

## Colab Session Setup

Every session starts with the same setup cells. The only line that differs per collaborator is `DRIVE_ROOT`.

### Cell 1: Mount Drive and clone/pull repo

```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess, os

REPO_URL = 'https://github.com/Aaron1776/antibody-property-prediction.git'
REPO_DIR = '/content/antibody-property-prediction'
BRANCH = 'implementation'

if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-b', BRANCH, REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull', 'origin', BRANCH], check=True)

import sys
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print("Repo ready.")
```

### Cell 2: Set paths

```python
from pathlib import Path

# Change this line only -- use your own Drive path if different
DRIVE_ROOT = Path('/content/drive/MyDrive/DL_Final_Project')

DATA_DIR = DRIVE_ROOT / 'data'
EMBEDDING_DIR = DRIVE_ROOT / 'embeddings'
RESULTS_DIR = DRIVE_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
CHECKPOINT_DIR = DRIVE_ROOT / 'checkpoints'

for d in [DATA_DIR, EMBEDDING_DIR, RESULTS_DIR, FIGURES_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("Paths set.")
```

### Cell 3: Install dependencies

```python
# HMMER must come before ANARCI or ANARCI will silently fail at runtime
!apt-get install -y hmmer

!pip install -q fair-esm ablang2 anarci wandb
```

### Cell 4: IPython upgrade (required for autoreload on Python 3.12)

```python
!pip install -q --upgrade ipython
```

### Cell 5: Enable autoreload and clear pycache

```python
%load_ext autoreload
%autoreload 2

import subprocess
subprocess.run(['find', '/content/antibody-property-prediction', '-type', 'd',
                '-name', '__pycache__', '-exec', 'rm', '-rf', '{}', '+'],
               capture_output=True)
print("Autoreload enabled, pycache cleared.")
```

### Cell 6: GPU check

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cell 7: W&B login

```python
import wandb
wandb.login()
```

## W&B Setup

Both collaborators should be added to the W&B project `antibody-property-prediction`. Use consistent run naming:

```
{model}_{strategy}_lambda{lambda_cdr}_{timestamp}
# e.g.: ablang2_delta_residue_lambda0.1_20240415
```

## Saving Work Back from Colab

Push `.py` changes from Colab:

```python
import subprocess, os
os.chdir('/content/antibody-property-prediction')
subprocess.run(['git', 'config', 'user.email', 'your@email.com'])
subprocess.run(['git', 'config', 'user.name', 'Your Name'])
subprocess.run(['git', 'add', 'src/'])
subprocess.run(['git', 'commit', '-m', 'your commit message'])
subprocess.run(['git', 'push', 'origin', 'implementation'])
```

Notebooks are not tracked by git (too large, outputs change). Sync notebooks to Drive manually if needed.

## Key Dependency Notes

- **HMMER before ANARCI**: Install HMMER via apt before `pip install anarci`. If HMMER is absent, ANARCI installs silently but fails at runtime with `FileNotFoundError` on `hmmscan`.
- **PyTDC incompatibility**: PyTDC is incompatible with Colab's numpy. Do not use it. SAbDab data is loaded directly from the Zenodo CSV.
- **IPython upgrade**: Required for `%autoreload` to work correctly on Python 3.12 (Colab default as of late 2024).
