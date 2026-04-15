"""Loss functions for training.

Task loss: MSE for regression (both Task 1 and Task 2).

CDR constraint loss (Task 1 only):
    Encodes the biological prior that CDR mutations should have higher
    predicted effect magnitude than framework mutations.

    Two formulations are available:

    Batch-mean (original):
        constraint_loss = ReLU(mean(|FR_predicted|) - mean(|CDR_predicted|))
        Fires when group means violate the prior. One gradient term per batch.

    Pairwise ranking (extension):
        constraint_loss = mean over all (FR, CDR) pairs of
            ReLU(|FR_predicted| - |CDR_predicted| + margin)
        Fires on every individual FR-CDR pair. O(n_fr x n_cdr) gradient
        terms per batch. Enforces a minimum margin between CDR and FR
        absolute predictions.

    Applied to AbAgym (Task 1) only. Not applied to SAbDab (no CDR mapping).
"""

from typing import List, Union

import torch
import torch.nn.functional as F

from src.config import CDR_REGIONS, FR_REGION


def cdr_constraint_loss(
    predictions: torch.Tensor,
    regions: Union[List[str], torch.Tensor],
) -> torch.Tensor:
    """Compute the CDR constraint loss for a batch.

    Penalizes batches where the model predicts larger absolute effects for
    framework mutations than for CDR mutations.

    Parameters
    ----------
    predictions:
        (batch_size,) float tensor of scalar predictions.
    regions:
        List or array of region labels for each sample in the batch.
        Valid values: 'CDR_H1', 'CDR_H2', 'CDR_H3', 'CDR_L1', 'CDR_L2',
        'CDR_L3', 'FR'.

    Returns
    -------
    Scalar tensor. Returns 0.0 (as a tensor) if the batch has no FR
    mutations or no CDR mutations, since the group comparison is
    undefined.

    Notes
    -----
    The constraint compares the MEAN absolute prediction of FR mutations
    against the MEAN absolute prediction of CDR mutations. This is a
    batch-level statistic, not a pairwise per-sample penalty.

    Lambda = 0 must reproduce the unconstrained baseline exactly
    (combined_loss handles this by multiplying constraint by lambda).
    """
    if not isinstance(regions, list):
        regions = list(regions)

    fr_mask = torch.tensor(
        [r == FR_REGION for r in regions],
        dtype=torch.bool,
        device=predictions.device,
    )
    cdr_mask = torch.tensor(
        [r in CDR_REGIONS for r in regions],
        dtype=torch.bool,
        device=predictions.device,
    )

    if fr_mask.sum() == 0 or cdr_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    fr_abs_mean = predictions[fr_mask].abs().mean()
    cdr_abs_mean = predictions[cdr_mask].abs().mean()

    return F.relu(fr_abs_mean - cdr_abs_mean)


def pairwise_cdr_constraint_loss(
    predictions: torch.Tensor,
    regions: Union[List[str], torch.Tensor],
    margin: float = 0.1,
) -> torch.Tensor:
    """Pairwise ranking CDR constraint loss.

    For every (FR, CDR) pair in the batch, penalizes cases where the
    absolute FR prediction exceeds the absolute CDR prediction by more
    than -margin (i.e., where |FR| > |CDR| - margin).

    loss = mean_{i in FR, j in CDR} ReLU(|pred_i| - |pred_j| + margin)

    This enforces that every CDR prediction exceeds every FR prediction
    by at least `margin`. It is zero only when all |CDR_pred| >= all
    |FR_pred| + margin.

    Compared to the batch-mean formulation:
    - Fires on O(n_fr x n_cdr) pairs instead of one group-mean comparison
    - Provides a richer gradient signal per batch
    - Enforces a minimum separation (margin) rather than just mean ordering

    Parameters
    ----------
    predictions:
        (batch_size,) float tensor of scalar predictions.
    regions:
        List or array of region labels for each sample in the batch.
        Valid values: 'CDR_H1', 'CDR_H2', 'CDR_H3', 'CDR_L1', 'CDR_L2',
        'CDR_L3', 'FR'.
    margin:
        Minimum required gap: |CDR_pred| must exceed |FR_pred| by at
        least this value to contribute zero loss. Default 0.1.

    Returns
    -------
    Scalar tensor. Returns 0.0 (as a tensor) if the batch has no FR
    mutations or no CDR mutations.
    """
    if not isinstance(regions, list):
        regions = list(regions)

    fr_mask = torch.tensor(
        [r == FR_REGION for r in regions],
        dtype=torch.bool,
        device=predictions.device,
    )
    cdr_mask = torch.tensor(
        [r in CDR_REGIONS for r in regions],
        dtype=torch.bool,
        device=predictions.device,
    )

    if fr_mask.sum() == 0 or cdr_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    fr_abs = predictions[fr_mask].abs()    # (n_fr,)
    cdr_abs = predictions[cdr_mask].abs()  # (n_cdr,)

    # All pairwise differences: (n_fr, n_cdr)
    diffs = fr_abs.unsqueeze(1) - cdr_abs.unsqueeze(0)

    return F.relu(diffs + margin).mean()


def combined_loss(
    task_loss: torch.Tensor,
    constraint_loss: torch.Tensor,
    lambda_cdr: float,
) -> torch.Tensor:
    """Combine task loss and CDR constraint loss.

    total_loss = task_loss + lambda_cdr * constraint_loss

    When lambda_cdr = 0, returns task_loss unchanged (the constraint term
    is exactly zero). This is the required behavior for sanity checking that
    lambda=0 reproduces the unconstrained baseline.

    Parameters
    ----------
    task_loss:
        Scalar tensor. MSE or other regression loss.
    constraint_loss:
        Scalar tensor from cdr_constraint_loss().
    lambda_cdr:
        Non-negative scalar weight. Sweep: [0, 0.1, 0.5, 1.0].

    Returns
    -------
    Scalar tensor.
    """
    return task_loss + lambda_cdr * constraint_loss
