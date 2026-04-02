# Contributing

## Collaboration Loop

Edit `.py` files locally or in your preferred editor, push to GitHub, then pull before running. All modeling logic lives in `src/`. Notebooks are thin orchestration layers that import from `src/` -- keep it that way.

```bash
# After editing src/ files locally:
git add src/
git commit -m "your message"
git push origin implementation

# In Colab, at the start of each session:
!git -C /content/antibody-property-prediction pull origin implementation

# Locally, before running notebooks:
git pull origin implementation
```

## Branch Structure

- `main`: stable, reflects the last clean state
- `implementation`: all active development happens here

Never push directly to `main`.

## Google Drive Setup

Both collaborators use a shared Drive folder. `DRIVE_ROOT` is resolved automatically by `src/config.py` -- no manual path setup needed in the notebooks. Resolution order:
1. Colab: `/content/drive/MyDrive/DL_Final_Project/Antibody_Project`
2. Local Mac with Google Drive Desktop: auto-detected via glob (any Google account)
3. Fallback: `outputs/` at repo root (gitignored)

### Collaborator Setup

Complete these steps once before running notebooks locally. Colab-only collaborators only need steps 1-2.

1. **Get repo access**: ask the owner to add you as a collaborator on GitHub
2. **Get Drive access**: ask the owner to share the `DL_Final_Project/Antibody_Project` folder with your Google account (Editor access)
3. **Add Drive shortcut**: in Google Drive, open "Shared with me", right-click the folder, select Organize → Add shortcut, and place it at **My Drive root** (not inside any subfolder). This is required so the path matches what Colab expects.
4. **Install Google Drive Desktop**: download and install from drive.google.com/drive/download. Sign in with the same Google account. Wait for the initial sync to complete.
5. **Clone the repo locally**:
   ```bash
   git clone -b implementation https://github.com/Aaron1776/antibody-property-prediction.git
   cd antibody-property-prediction
   pip install -r requirements.txt
   ```
6. **Verify**: open a notebook and run the setup cells. Cell 2 should print a Drive root path ending in `DL_Final_Project/Antibody_Project` -- not `outputs/`. If it prints `outputs/`, the shortcut is not in the right place or Drive Desktop has not finished syncing.

Drive folder structure:
```
/MyDrive/DL_Final_Project/Antibody_Project/
    data/
        abagym_antibody.csv
        abagym_sequences.csv
        sabdab_affinity.csv
    embeddings/
        # ESM-2 sequence-level
        esm2_abagym.pt                    (5318, 2560)
        esm2_abagym_wildtype.pt           (5, 2560)
        esm2_abagym_wildtype_index.json
        esm2_abagym_delta.pt              (5318, 2560)
        esm2_sabdab.pt                    (491, 2560)
        esm2_sabdab_index.json
        # ESM-2 residue-level
        esm2_abagym_residue_mutsite.pt    (5318, 1280)
        esm2_abagym_residue_wtsite.pt     (5318, 1280)
        esm2_abagym_residue_delta.pt      (5318, 1280)
        # AbLang2 sequence-level
        ablang2_abagym.pt                 (5318, 960)
        ablang2_abagym_wildtype.pt        (5, 960)
        ablang2_abagym_wildtype_index.json
        ablang2_abagym_delta.pt           (5318, 960)
        ablang2_sabdab.pt                 (491, 960)
        ablang2_sabdab_index.json
        # AbLang2 residue-level
        ablang2_abagym_residue_mutsite.pt (5318, 480)
        ablang2_abagym_residue_wtsite.pt  (5318, 480)
        ablang2_abagym_residue_delta.pt   (5318, 480)
    results/
        figures/
    checkpoints/
    wandb/
```

## Session Setup

Notebooks run unchanged on Colab or locally. The setup cells detect the environment automatically -- no per-collaborator edits needed.

### Cell 1: Environment detection, sys.path, repo

```python
import subprocess, os, sys
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

    REPO_URL = 'https://github.com/Aaron1776/antibody-property-prediction.git'
    REPO_DIR = '/content/antibody-property-prediction'
    BRANCH   = 'implementation'

    if not os.path.exists(REPO_DIR):
        subprocess.run(['git', 'clone', '-b', BRANCH, REPO_URL, REPO_DIR], check=True)
    else:
        subprocess.run(['git', '-C', REPO_DIR, 'pull', 'origin', BRANCH], check=True)

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
else:
    REPO_DIR = str(Path('..').resolve())
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

print(f"Environment: {'Colab' if IN_COLAB else 'local'}")
print(f"Repo: {REPO_DIR}")
```

### Cell 2: Paths (auto-resolved by src/config.py)

```python
from src.config import DRIVE_ROOT, EMBEDDING_DIR, RESULTS_DIR, FIGURES_DIR, CHECKPOINT_DIR

for d in [EMBEDDING_DIR, RESULTS_DIR, FIGURES_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Drive root: {DRIVE_ROOT}")
print("Paths set.")
```

### Cell 3: Install dependencies

```python
if IN_COLAB:
    import subprocess
    subprocess.run(['apt-get', 'install', '-y', 'hmmer'], check=True)
    subprocess.run(['pip', 'install', '-q', '--upgrade', 'ipython'], check=True)
    subprocess.run(['pip', 'install', '-q', 'fair-esm', 'ablang2', 'anarci', 'wandb'], check=True)
else:
    print("Local run -- installation skipped.")
```

### Cell 4: Autoreload and pycache

```python
%load_ext autoreload
%autoreload 2

if IN_COLAB:
    import subprocess
    subprocess.run(
        ['find', REPO_DIR, '-type', 'd', '-name', '__pycache__', '-exec', 'rm', '-rf', '{}', '+'],
        capture_output=True,
    )

print("Autoreload enabled.")
```

### Cell 5: Device check

```python
import torch
from src.config import DEVICE

print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cell 6: W&B login

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
