"""Training loop for AbAgym (Task 1) and SAbDab (Task 2).

This module is a stub. The training loop will be iterated on in NB05.
Class signatures and docstrings are complete; implementations are placeholders.

Key design decisions to implement:
- Batching for AbAgym must mix datasets to avoid HER2-only batches
  (HER2 contributes zero constraint gradient -- all CDR H3, no FR).
- Early stopping based on validation Spearman (Task 1) or Pearson (Task 2).
- W&B logging for all runs.
- lambda_cdr = 0 must produce identical results to unconstrained baseline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import AbAgymDataset, SAbDabDataset
from src.models.mlp import MLP
from src.training.losses import cdr_constraint_loss, combined_loss


@dataclass
class TrainConfig:
    """Configuration for a single training run.

    Attributes
    ----------
    model_name:
        'esm2' or 'ablang2'.
    embedding_strategy:
        String key from EmbeddingStrategy enum, e.g. 'delta_sequence'.
    lr:
        Learning rate.
    epochs:
        Maximum number of training epochs.
    batch_size:
        Number of samples per batch.
    hidden_dims:
        MLP hidden layer sizes, e.g. [256, 128].
    dropout:
        Dropout probability for MLP.
    lambda_cdr:
        CDR constraint loss weight. 0.0 = unconstrained baseline.
        Sweep: [0, 0.1, 0.5, 1.0].
    seed:
        Random seed for reproducibility.
    patience:
        Early stopping patience (epochs without improvement).
    wandb_project:
        W&B project name.
    wandb_run_name:
        W&B run name. Convention: {model}_{strategy}_lambda{lambda_cdr}_{date}.
    """
    model_name: str
    embedding_strategy: str
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    lambda_cdr: float = 0.0
    seed: int = 42
    patience: int = 10
    wandb_project: str = 'antibody-property-prediction'
    wandb_run_name: Optional[str] = None


def train_abagym(
    config: TrainConfig,
    train_dataset: AbAgymDataset,
    val_dataset: AbAgymDataset,
    input_dim: int,
    device: str,
) -> Dict:
    """Train MLP on AbAgym mutation effect prediction.

    Handles constraint loss if config.lambda_cdr > 0. Logs metrics to W&B.

    Parameters
    ----------
    config:
        TrainConfig instance.
    train_dataset:
        AbAgymDataset for training.
    val_dataset:
        AbAgymDataset for validation.
    input_dim:
        MLP input dimension (depends on model and embedding strategy).
    device:
        'cuda' or 'cpu'.

    Returns
    -------
    dict with keys: 'model', 'train_history', 'val_history', 'best_epoch',
    'best_val_spearman', 'per_dataset_spearman'.

    Implementation notes:
    - Use a stratified or shuffled sampler to ensure batches mix datasets.
      Pure random shuffle should work for most batches, but monitor whether
      any batch ends up being all HER2 (184 samples -- possible at large batch
      sizes or unlucky shuffles).
    - Evaluate with evaluate_abagym() after each epoch.
    - Report HER2 Spearman separately (bimodal score distribution, all CDR H3).

    TODO: implement in NB05.
    """
    raise NotImplementedError("Implement training loop in NB05.")


def train_sabdab(
    config: TrainConfig,
    train_dataset: SAbDabDataset,
    val_dataset: SAbDabDataset,
    input_dim: int,
    device: str,
) -> Dict:
    """Train MLP on SAbDab binding affinity prediction.

    No constraint loss (SAbDab has no CDR/FR mapping).

    Parameters
    ----------
    config:
        TrainConfig instance.
    train_dataset:
        SAbDabDataset for training.
    val_dataset:
        SAbDabDataset for validation.
    input_dim:
        MLP input dimension.
    device:
        'cuda' or 'cpu'.

    Returns
    -------
    dict with keys: 'model', 'train_history', 'val_history', 'best_epoch',
    'best_val_pearson', 'best_val_rmse'.

    TODO: implement in NB05.
    """
    raise NotImplementedError("Implement training loop in NB05.")


def evaluate_abagym(
    model: MLP,
    dataset: AbAgymDataset,
    device: str,
) -> Dict:
    """Evaluate model on AbAgym. Returns Spearman per-dataset and aggregate.

    Parameters
    ----------
    model:
        Trained MLP.
    dataset:
        AbAgymDataset (typically validation or test split).
    device:
        'cuda' or 'cpu'.

    Returns
    -------
    dict with keys:
        'aggregate': float (Spearman r across all 5 datasets)
        'per_dataset': dict mapping DMS_name -> Spearman r
        'HER2': float (Spearman r for HER2 only, reported separately)

    TODO: implement in NB05.
    """
    raise NotImplementedError("Implement evaluation in NB05.")


def evaluate_sabdab(
    model: MLP,
    dataset: SAbDabDataset,
    device: str,
) -> Dict:
    """Evaluate model on SAbDab. Returns Pearson correlation and RMSE.

    Parameters
    ----------
    model:
        Trained MLP.
    dataset:
        SAbDabDataset (typically validation or test split).
    device:
        'cuda' or 'cpu'.

    Returns
    -------
    dict with keys: 'pearson': float, 'rmse': float.

    TODO: implement in NB05.
    """
    raise NotImplementedError("Implement evaluation in NB05.")
