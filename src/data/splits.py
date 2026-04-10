"""Stratified train/val/test split for AbAgym.

The split is stratified within each antibody dataset: 80/10/10 within each
DMS_name, then pooled. This ensures all five antibodies are represented in
every split and prevents a model from being evaluated on antibodies it has
never seen (which would be a different, harder task).

The split is deterministic given a fixed random_state. Both notebooks (oscar
and lucas) must use the same random_state and call this function on the same
full DataFrame to guarantee identical splits.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def make_stratified_splits(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
    dataset_col: str = 'DMS_name',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 80/10/10 split within each antibody dataset.

    Splits are performed independently within each unique value of
    dataset_col, then the per-antibody splits are pooled. This ensures
    each antibody contributes proportionally to all three splits.

    Parameters
    ----------
    df:
        Full AbAgym antibody DataFrame (5318 rows).
    val_frac:
        Fraction of each antibody's mutations to use for validation.
    test_frac:
        Fraction of each antibody's mutations to use for test.
    random_state:
        Random seed for reproducibility. Must be the same across both
        collaborator notebooks.
    dataset_col:
        Column identifying which antibody each mutation belongs to.

    Returns
    -------
    train_idx, val_idx, test_idx:
        Integer index arrays into df (not row labels). Use df.iloc[train_idx]
        to subset.
    """
    rng = np.random.default_rng(random_state)

    train_idx, val_idx, test_idx = [], [], []

    for ds_name, group in df.groupby(dataset_col):
        idx = group.index.to_numpy()
        rng.shuffle(idx)

        n = len(idx)
        n_test = max(1, round(n * test_frac))
        n_val  = max(1, round(n * val_frac))

        test_idx.append(idx[:n_test])
        val_idx.append(idx[n_test:n_test + n_val])
        train_idx.append(idx[n_test + n_val:])

    train_idx = np.concatenate(train_idx)
    val_idx   = np.concatenate(val_idx)
    test_idx  = np.concatenate(test_idx)

    return train_idx, val_idx, test_idx
