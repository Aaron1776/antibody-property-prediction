"""Training loop for AbAgym (Task 1) and SAbDab (Task 2).

Key design decisions:
- Batching for AbAgym shuffles the full training set so batches mix datasets.
  HER2 contributes 184 samples (~3.5% of total); at batch_size=64, the expected
  number of HER2 samples per batch is ~2.2. Pure random shuffle is sufficient.
- Early stopping based on validation Spearman (Task 1).
- W&B logging for all runs.
- lambda_cdr = 0 produces identical results to the unconstrained baseline
  because combined_loss(task_loss, constraint_loss, 0.0) == task_loss.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ABAGYM_DATASETS, set_seed
from src.data.datasets import AbAgymDataset, SAbDabDataset
from src.models.mlp import MLP
from src.training.losses import (
    cdr_constraint_loss,
    pairwise_cdr_constraint_loss,
    combined_loss,
)

import wandb

_HER2_NAME = 'HER2_2021_trastuzumab'
_EVAL_BATCH_SIZE = 512


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
    constraint_type:
        'batch_mean' (original) or 'pairwise' (ranking loss extension).
        Ignored when lambda_cdr = 0.
    constraint_margin:
        Margin for pairwise constraint. Ignored for batch_mean.
        Default 0.1.
    seed:
        Random seed for reproducibility.
    patience:
        Early stopping patience (epochs without improvement on val Spearman).
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
    constraint_type: str = 'batch_mean'
    constraint_margin: float = 0.1
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

    Handles CDR constraint loss if config.lambda_cdr > 0.
    Logs all metrics to W&B. Applies early stopping on aggregate
    validation Spearman correlation. Restores the best-epoch weights
    before returning.

    Parameters
    ----------
    config:
        TrainConfig instance.
    train_dataset:
        AbAgymDataset (or Subset thereof) for training.
    val_dataset:
        AbAgymDataset (or Subset thereof) for validation.
    input_dim:
        MLP input dimension. Must match the chosen model and strategy.
    device:
        'cuda', 'mps', or 'cpu'.

    Returns
    -------
    dict with keys:
        'model'                -- trained MLP at best epoch (restored)
        'train_history'        -- list of per-epoch mean train MSE
        'val_history'          -- list of per-epoch aggregate val Spearman
        'best_epoch'           -- epoch with highest val Spearman (1-indexed)
        'best_val_spearman'    -- aggregate val Spearman at best epoch
        'per_dataset_spearman' -- per-DMS Spearman at best epoch (val set)
    """
    set_seed(config.seed)

    # Build model
    model = MLP(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mse_fn = nn.MSELoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # W&B setup
    run_name = config.wandb_run_name or (
        f"{config.model_name}_{config.embedding_strategy}"
        f"_lambda{config.lambda_cdr}"
    )
    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config={
            'model_name': config.model_name,
            'embedding_strategy': config.embedding_strategy,
            'lr': config.lr,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'hidden_dims': config.hidden_dims,
            'dropout': config.dropout,
            'lambda_cdr': config.lambda_cdr,
            'constraint_type': config.constraint_type,
            'constraint_margin': config.constraint_margin,
            'seed': config.seed,
            'input_dim': input_dim,
        },
        reinit=False,
    )

    train_history: List[float] = []
    val_history: List[float] = []
    best_val_spearman: float = -np.inf
    best_epoch: int = 0
    best_state = None
    patience_counter: int = 0

    epoch_bar = tqdm(range(config.epochs), desc=f"{run_name}", unit="epoch")

    for epoch in epoch_bar:
        # ---- Training pass ----
        model.train()
        epoch_mse = 0.0
        n_batches = 0

        for x, y, meta in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x).squeeze(-1)  # (batch,)

            task_loss = mse_fn(preds, y)

            if config.lambda_cdr > 0:
                if config.constraint_type == 'pairwise':
                    constraint = pairwise_cdr_constraint_loss(
                        preds, meta['region'], margin=config.constraint_margin
                    )
                else:
                    constraint = cdr_constraint_loss(preds, meta['region'])
                loss = combined_loss(task_loss, constraint, config.lambda_cdr)
            else:
                loss = task_loss

            loss.backward()
            optimizer.step()

            epoch_mse += task_loss.item()
            n_batches += 1

        avg_train_mse = epoch_mse / n_batches
        train_history.append(avg_train_mse)

        # ---- Validation pass ----
        val_metrics = evaluate_abagym(model, val_dataset, device)
        val_spearman = val_metrics['aggregate']
        val_history.append(val_spearman)

        # ---- Update epoch bar ----
        epoch_bar.set_postfix(
            train_mse=f"{avg_train_mse:.4f}",
            val_rho=f"{val_spearman:.4f}",
            best=f"{best_val_spearman:.4f}",
            patience=f"{patience_counter}/{config.patience}",
        )

        # ---- W&B logging ----
        log_dict: Dict = {
            'epoch': epoch + 1,
            'train_mse': avg_train_mse,
            'val_spearman': val_spearman,
            'val_spearman_excl_her2': val_metrics['exclude_her2'],
            'val_spearman_HER2': val_metrics['HER2'],
        }
        for ds_name, r in val_metrics['per_dataset'].items():
            log_dict[f'val_spearman/{ds_name}'] = r
        wandb.log(log_dict, step=epoch + 1)

        # ---- Early stopping ----
        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                epoch_bar.write(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best epoch: {best_epoch} (val ρ={best_val_spearman:.4f})"
                )
                break

    # Restore best weights
    model.load_state_dict(best_state)

    # Final evaluation at best weights
    final_val_metrics = evaluate_abagym(model, val_dataset, device)

    wandb.summary['best_epoch'] = best_epoch
    wandb.summary['best_val_spearman'] = best_val_spearman
    wandb.summary['best_val_spearman_excl_her2'] = final_val_metrics['exclude_her2']
    for ds_name, r in final_val_metrics['per_dataset'].items():
        wandb.summary[f'best_val_spearman/{ds_name}'] = r
    wandb.finish()

    return {
        'model': model,
        'train_history': train_history,
        'val_history': val_history,
        'best_epoch': best_epoch,
        'best_val_spearman': best_val_spearman,
        'per_dataset_spearman': final_val_metrics['per_dataset'],
    }


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
        'cuda', 'mps', or 'cpu'.

    Returns
    -------
    dict with keys: 'model', 'train_history', 'val_history', 'best_epoch',
    'best_val_pearson', 'best_val_rmse'.

    TODO: implement.
    """
    raise NotImplementedError("Implement SAbDab training loop.")


def evaluate_abagym(
    model: MLP,
    dataset: AbAgymDataset,
    device: str,
) -> Dict:
    """Evaluate model on AbAgym. Returns Spearman per-dataset and aggregate.

    Runs the model in eval mode (no gradient tracking). Uses a fixed batch
    size of 512 for speed.

    Parameters
    ----------
    model:
        MLP (in any state; switched to eval mode internally).
    dataset:
        AbAgymDataset or Subset for evaluation.
    device:
        'cuda', 'mps', or 'cpu'.

    Returns
    -------
    dict with keys:
        'aggregate'    -- float, Spearman r across all 5 datasets pooled
        'exclude_her2' -- float, Spearman r excluding HER2 samples
        'per_dataset'  -- dict mapping DMS_name (str) -> Spearman r (float)
        'HER2'         -- float, Spearman r for HER2 only
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=_EVAL_BATCH_SIZE, shuffle=False)

    all_preds: List[float] = []
    all_labels: List[float] = []
    all_dms_names: List[str] = []

    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            preds = model(x).squeeze(-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
            all_dms_names.extend(meta['dms_name'])

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    names_arr = np.array(all_dms_names)

    # Per-dataset Spearman
    per_dataset: Dict[str, float] = {}
    for ds_name in np.unique(names_arr):
        mask = names_arr == ds_name
        r, _ = spearmanr(labels_arr[mask], preds_arr[mask])
        per_dataset[str(ds_name)] = float(r)

    # Aggregate across all 5 datasets (samples pooled)
    agg_r, _ = spearmanr(labels_arr, preds_arr)

    # Aggregate excluding HER2
    non_her2_mask = names_arr != _HER2_NAME
    if non_her2_mask.sum() > 1:
        excl_r, _ = spearmanr(labels_arr[non_her2_mask], preds_arr[non_her2_mask])
    else:
        excl_r = float('nan')

    her2_r = per_dataset.get(_HER2_NAME, float('nan'))

    return {
        'aggregate': float(agg_r),
        'exclude_her2': float(excl_r),
        'per_dataset': per_dataset,
        'HER2': her2_r,
    }


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
        'cuda', 'mps', or 'cpu'.

    Returns
    -------
    dict with keys: 'pearson': float, 'rmse': float.

    TODO: implement.
    """
    raise NotImplementedError("Implement SAbDab evaluation.")
