"""Figure generation for embedding EDA and training analysis.

All functions save at 300 DPI to output_dir. All functions are stubs;
implementations will be filled in as notebooks 04 and 06 are executed.

Style conventions:
- No emojis in titles, labels, or captions
- Use plain text alternatives (e.g. "CDR" not "CDR loop")
- Consistent color palette across model comparison figures
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_delta_norm_by_dataset(
    delta_tensor: torch.Tensor,
    dms_names: List[str],
    model_name: str,
    output_dir: Path,
):
    """Violin or box plots of delta embedding norms, one panel per dataset.

    Shows the distribution of ||delta||_2 for each of the 5 AbAgym datasets.
    Used to verify that delta embeddings have discriminable variance across
    datasets and to compare ESM-2 vs AbLang2.

    TODO: implement in NB04.
    """
    raise NotImplementedError("Implement in NB04.")


def plot_delta_norm_cdr_vs_fr(
    delta_tensor: torch.Tensor,
    regions: List[str],
    model_name: str,
    output_dir: Path,
):
    """Compare delta norms for CDR vs FR mutations.

    Visualizes whether the model encodes the CDR prior (CDR norms > FR norms)
    or the inverse (FR norms > CDR norms, as found for ESM-2).
    Include Mann-Whitney U test result in the figure.

    TODO: implement in NB04.
    """
    raise NotImplementedError("Implement in NB04.")


def plot_delta_pca(
    delta_tensor: torch.Tensor,
    labels: np.ndarray,
    color_by: str,
    model_name: str,
    output_dir: Path,
    title_suffix: str = '',
):
    """2D PCA scatter plot of delta embeddings.

    Parameters
    ----------
    color_by:
        What to color points by: 'dms_score', 'dataset', 'chain', 'region'.
    title_suffix:
        Appended to the plot title for identification.

    Known findings to verify:
    - ESM-2: cross shape in PCA caused by chain identity (H vs L)
    - ESM-2: PC1+PC2 explain ~26% of variance
    - HER2: no cross shape (all heavy chain)

    TODO: implement in NB04.
    """
    raise NotImplementedError("Implement in NB04.")


def plot_delta_variance_ratio(
    raw_tensor: torch.Tensor,
    delta_tensor: torch.Tensor,
    dms_names: List[str],
    model_name: str,
    output_dir: Path,
):
    """Bar chart of CoV (raw embeddings) vs CoV (delta embeddings) per dataset.

    Illustrates the discriminability gain from delta computation.
    Known ESM-2 finding: CoV 177x-301x higher for deltas than raw embeddings.

    TODO: implement in NB04.
    """
    raise NotImplementedError("Implement in NB04.")


def plot_spearman_by_dataset(
    results: Dict[str, float],
    model_name: str,
    strategy: str,
    output_dir: Path,
):
    """Bar chart of Spearman correlation per dataset.

    HER2 should be displayed separately or annotated as an outlier
    (bimodal score distribution, all CDR H3).

    TODO: implement in NB06.
    """
    raise NotImplementedError("Implement in NB06.")


def plot_lambda_sweep(
    results: Dict[float, Dict],
    model_name: str,
    strategy: str,
    output_dir: Path,
):
    """Line plot of Spearman correlation vs lambda for CDR constraint sweep.

    X-axis: lambda values [0, 0.1, 0.5, 1.0].
    Y-axis: aggregate Spearman (and optionally per-dataset lines).
    One line per dataset, plus aggregate.

    TODO: implement in NB06.
    """
    raise NotImplementedError("Implement in NB06.")


def plot_learning_curves(
    train_history: List[float],
    val_history: List[float],
    metric: str,
    model_name: str,
    run_name: str,
    output_dir: Path,
):
    """Training and validation loss (or metric) vs epoch.

    Parameters
    ----------
    metric:
        Name of the metric being plotted, e.g. 'MSE' or 'Spearman'.

    TODO: implement in NB06.
    """
    raise NotImplementedError("Implement in NB06.")


def plot_violation_heatmap(
    predictions: np.ndarray,
    regions: List[str],
    dms_names: List[str],
    model_name: str,
    lambda_cdr: float,
    output_dir: Path,
):
    """Heatmap of CDR constraint violation rates per dataset.

    Violation: a sample where |FR_predicted| > |CDR_predicted| in a given batch.
    Shows how often the constraint fires, broken down by dataset and CDR loop.

    TODO: implement in NB06.
    """
    raise NotImplementedError("Implement in NB06.")


def plot_embedding_strategy_comparison(
    results: Dict[str, Dict],
    model_name: str,
    output_dir: Path,
):
    """Compare Spearman correlations across embedding strategies (Exp 2-6).

    Groups by strategy with one bar per dataset, or one grouped bar per
    strategy. Enables direct comparison of which input formulation works best.

    TODO: implement in NB06.
    """
    raise NotImplementedError("Implement in NB06.")
