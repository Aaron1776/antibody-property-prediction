"""Figure generation for embedding EDA and training analysis.

All functions save at 300 DPI to output_dir.

Style conventions:
- No emojis in titles, labels, or captions
- Use plain text alternatives (e.g. "CDR" not "CDR loop")
- Consistent color palette across model comparison figures
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Color palette constants (used consistently across all figures)
# ---------------------------------------------------------------------------

_MODEL_COLORS: Dict[str, str] = {
    'ESM-2': '#4C72B0',
    'AbLang2': '#DD8452',
}

_REGION_COLORS: Dict[str, str] = {
    'CDR': '#2ca02c',
    'FR': '#d62728',
}

_REGION_DETAIL_COLORS: Dict[str, str] = {
    'CDR_H1': '#1f77b4',
    'CDR_H2': '#aec7e8',
    'CDR_H3': '#ff7f0e',
    'CDR_L1': '#ffbb78',
    'CDR_L2': '#2ca02c',
    'CDR_L3': '#98df8a',
    'FR': '#d62728',
}

_CHAIN_COLORS: Dict[str, str] = {
    'H': '#4C72B0',
    'L': '#DD8452',
}

_DATASET_COLORS: List[str] = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    return name.lower().replace('-', '').replace(' ', '_')


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / filename
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def _format_p(p: float) -> str:
    return f'p={p:.2e}' if p >= 1e-300 else 'p<1e-300'


# ---------------------------------------------------------------------------
# NB04: embedding EDA figures
# ---------------------------------------------------------------------------

def plot_delta_norm_by_dataset(
    delta_tensor: torch.Tensor,
    dms_names: List[str],
    model_name: str,
    output_dir: Path,
) -> None:
    """Violin plots of delta embedding norms, one panel per dataset.

    Shows the distribution of ||delta||_2 for each of the 5 AbAgym datasets.
    Used to verify that delta embeddings have discriminable variance and to
    compare ESM-2 vs AbLang2.
    """
    norms = torch.norm(delta_tensor.float(), dim=1).numpy()
    dms_arr = np.array(dms_names)
    datasets = sorted(set(dms_names))

    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 5), sharey=True)

    for ax, ds, c in zip(axes, datasets, _DATASET_COLORS):
        mask = dms_arr == ds
        ds_norms = norms[mask]
        parts = ax.violinplot(ds_norms, positions=[0], showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(c)
            pc.set_alpha(0.72)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)
        ax.set_xticks([0])
        ax.set_xticklabels([ds.replace('_', '\n')], fontsize=7)
        ax.set_title(f'N={mask.sum()}', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Delta norm (L2)', fontsize=11)
    fig.suptitle(
        f'{model_name} -- sequence delta norms by dataset',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    _save(fig, output_dir, f'delta_norm_by_dataset_{_sanitize_name(model_name)}.png')


def plot_delta_norm_cdr_vs_fr(
    delta_tensor: torch.Tensor,
    regions: List[str],
    model_name: str,
    output_dir: Path,
) -> None:
    """Violin plot comparing delta norms for CDR vs FR mutations.

    Left panel: binary CDR vs FR with Mann-Whitney annotation.
    Right panel: all 7 region types.

    Visualizes whether the model encodes the CDR prior (CDR > FR) or the
    inverse (FR > CDR, as found for ESM-2 sequence-level deltas).
    """
    norms = torch.norm(delta_tensor.float(), dim=1).numpy()
    regions_arr = np.array(regions)

    cdr_mask = regions_arr != 'FR'
    fr_mask = regions_arr == 'FR'
    cdr_norms = norms[cdr_mask]
    fr_norms = norms[fr_mask]

    _, p = mannwhitneyu(cdr_norms, fr_norms, alternative='two-sided')
    cdr_med = np.median(cdr_norms)
    fr_med = np.median(fr_norms)
    direction = 'FR > CDR' if fr_med > cdr_med else 'CDR > FR'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: binary CDR / FR
    for pos, (lbl, n) in enumerate(zip(['CDR (all)', 'FR'], [cdr_norms, fr_norms])):
        c = _REGION_COLORS['CDR'] if pos == 0 else _REGION_COLORS['FR']
        parts = ax1.violinplot(n, positions=[pos], showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(c)
            pc.set_alpha(0.75)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['CDR (all)', 'FR'], fontsize=11)
    ax1.set_ylabel('Delta norm (L2)', fontsize=11)
    ax1.set_title(
        f'{model_name} -- CDR vs FR\nMann-Whitney {_format_p(p)} ({direction})',
        fontsize=10,
    )
    ax1.annotate(
        f'CDR median: {cdr_med:.4f}\nFR  median: {fr_med:.4f}',
        xy=(0.97, 0.97), xycoords='axes fraction',
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85),
    )
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: all 7 region types
    region_order = ['CDR_H1', 'CDR_H2', 'CDR_H3', 'CDR_L1', 'CDR_L2', 'CDR_L3', 'FR']
    for pos, reg in enumerate(region_order):
        mask = regions_arr == reg
        if mask.sum() == 0:
            continue
        c = _REGION_DETAIL_COLORS.get(reg, '#888888')
        parts = ax2.violinplot(norms[mask], positions=[pos], showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(c)
            pc.set_alpha(0.75)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

    ax2.set_xticks(list(range(len(region_order))))
    ax2.set_xticklabels(region_order, rotation=30, ha='right', fontsize=9)
    ax2.set_title(f'{model_name} -- delta norms by CDR loop and FR', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, f'delta_norm_cdr_vs_fr_{_sanitize_name(model_name)}.png')


def plot_delta_pca(
    delta_tensor: torch.Tensor,
    labels: np.ndarray,
    color_by: str,
    model_name: str,
    output_dir: Path,
    title_suffix: str = '',
) -> None:
    """2D PCA scatter plot of delta embeddings.

    Parameters
    ----------
    color_by:
        What to color points by: 'dms_score', 'dataset', 'chain', 'region',
        'cdr_fr'.
    title_suffix:
        Appended to the plot title for identification (e.g. 'sequence-level').

    Known findings to verify:
    - ESM-2: cross shape in PCA caused by chain identity (H vs L)
    - ESM-2: PC1+PC2 explain ~26% of variance
    - HER2: no cross shape (all heavy chain)
    """
    X = delta_tensor.float().numpy()
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))

    if np.issubdtype(np.asarray(labels).dtype, np.floating):
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap='viridis', s=4, alpha=0.5, linewidths=0,
        )
        plt.colorbar(sc, ax=ax, label=color_by)
    else:
        labels = np.asarray(labels)
        unique_labels = sorted(set(labels))
        if color_by == 'chain':
            palette = _CHAIN_COLORS
        elif color_by == 'region':
            palette = _REGION_DETAIL_COLORS
        elif color_by == 'cdr_fr':
            palette = _REGION_COLORS
        else:
            palette = dict(zip(unique_labels, _DATASET_COLORS))

        for lbl in unique_labels:
            mask = labels == lbl
            c = palette.get(lbl, '#888888')
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=c, s=4, alpha=0.5, linewidths=0, label=lbl,
            )
        ax.legend(markerscale=3, fontsize=8, loc='best', framealpha=0.8)

    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=11)
    title = f'{model_name} delta embeddings -- PCA (color: {color_by})'
    if title_suffix:
        title += f' -- {title_suffix}'
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    suffix = f'_{title_suffix.replace(" ", "_")}' if title_suffix else ''
    _save(fig, output_dir, f'delta_pca_{_sanitize_name(model_name)}_{color_by}{suffix}.png')


def plot_delta_variance_ratio(
    raw_tensor: torch.Tensor,
    delta_tensor: torch.Tensor,
    dms_names: List[str],
    model_name: str,
    output_dir: Path,
) -> None:
    """Bar chart of CoV (raw embeddings) vs CoV (delta embeddings) per dataset.

    Left panel: raw CoV vs delta CoV on a log scale (illustrates magnitude
    of discriminability gain).
    Right panel: ratio (delta CoV / raw CoV) with per-dataset annotation.

    Known ESM-2 finding: CoV 177x-301x higher for deltas than raw embeddings.
    """
    dms_arr = np.array(dms_names)
    datasets = sorted(set(dms_names))
    raw_norms = torch.norm(raw_tensor.float(), dim=1).numpy()
    delta_norms = torch.norm(delta_tensor.float(), dim=1).numpy()

    raw_covs, delta_covs, ratios = [], [], []
    for ds in datasets:
        mask = dms_arr == ds
        r_cov = raw_norms[mask].std() / raw_norms[mask].mean()
        d_cov = delta_norms[mask].std() / delta_norms[mask].mean()
        raw_covs.append(r_cov)
        delta_covs.append(d_cov)
        ratios.append(d_cov / r_cov)

    x = np.arange(len(datasets))
    width = 0.35
    color = _MODEL_COLORS.get(model_name, '#4C72B0')
    xlabels = [d.replace('_', '\n') for d in datasets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x - width/2, raw_covs, width, label='Raw embeddings', color='#aec7e8')
    ax1.bar(x + width/2, delta_covs, width, label='Delta embeddings', color=color, alpha=0.85)
    ax1.set_yscale('log')
    ax1.set_ylabel('CoV (std / mean of L2 norms)', fontsize=10)
    ax1.set_title(f'{model_name} -- discriminability: raw vs delta', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    bars = ax2.bar(x, ratios, color=color, alpha=0.85)
    for i, r in enumerate(ratios):
        ax2.text(i, r + max(ratios) * 0.01, f'{r:.0f}x', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('CoV ratio (delta / raw)', fontsize=10)
    ax2.set_title(f'{model_name} -- delta CoV / raw CoV per dataset', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, f'delta_variance_ratio_{_sanitize_name(model_name)}.png')


def plot_model_comparison_norms(
    esm2_norms: np.ndarray,
    ablang2_norms: np.ndarray,
    regions: List[str],
    level: str,
    output_dir: Path,
) -> None:
    """Scatter of ESM-2 delta norms vs AbLang2 delta norms, colored by CDR/FR.

    Answers: do the two models agree on which mutations produce large vs small
    delta norms? High Spearman r = models encode similar perturbation magnitude.
    Low r = complementary information.

    Parameters
    ----------
    level:
        'sequence' or 'residue' -- used in title and filename.
    """
    r, p = spearmanr(esm2_norms, ablang2_norms)
    regions_arr = np.array(regions)
    cdr_fr = np.where(regions_arr == 'FR', 'FR', 'CDR')

    fig, ax = plt.subplots(figsize=(7, 6))

    for lbl, c in _REGION_COLORS.items():
        mask = cdr_fr == lbl
        ax.scatter(
            esm2_norms[mask], ablang2_norms[mask],
            c=c, s=3, alpha=0.4, linewidths=0, label=f'{lbl} (N={mask.sum()})',
        )

    lim_min = min(esm2_norms.min(), ablang2_norms.min())
    lim_max = max(esm2_norms.max(), ablang2_norms.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=0.8, alpha=0.5, label='y=x')

    ax.set_xlabel('ESM-2 delta norm (L2)', fontsize=11)
    ax.set_ylabel('AbLang2 delta norm (L2)', fontsize=11)
    ax.set_title(
        f'Model agreement -- {level}-level delta norms\nSpearman r={r:.4f}, {_format_p(p)}',
        fontsize=11,
    )
    ax.legend(markerscale=4, fontsize=9)

    plt.tight_layout()
    _save(fig, output_dir, f'model_comparison_norms_{level}.png')


# ---------------------------------------------------------------------------
# NB04: DMS score distributions
# ---------------------------------------------------------------------------

def plot_dms_score_distributions(
    dms_scores: np.ndarray,
    dms_names: np.ndarray,
    output_dir: Path,
) -> None:
    """Histogram of MinMax-normalized DMS scores for each dataset.

    Answers: what does the label distribution look like per antibody?
    Flags bimodal or skewed distributions that may affect training and evaluation.

    Parameters
    ----------
    dms_scores:
        Per-mutation DMS scores, MinMax-normalized per dataset to [0, 1].
    dms_names:
        Dataset label for each mutation (same length as dms_scores).
    """
    datasets = list(dict.fromkeys(dms_names))
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)

    for ax, ds in zip(axes, datasets):
        mask = dms_names == ds
        scores = dms_scores[mask]
        color = _DATASET_COLORS[datasets.index(ds) % len(_DATASET_COLORS)]
        ax.hist(scores, bins=30, color=color, alpha=0.8, edgecolor='white', linewidth=0.4)
        ax.set_title(ds.replace('_', '\n'), fontsize=8)
        ax.set_xlabel('DMS score (normalized)', fontsize=8)
        ax.set_ylabel('Count' if ax is axes[0] else '', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(
            0.97, 0.97, f'N={mask.sum()}',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
        )

    fig.suptitle('DMS score distributions by dataset', fontsize=11, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, 'dms_score_distributions.png')


# ---------------------------------------------------------------------------
# NB06: training analysis figures (stubs)
# ---------------------------------------------------------------------------

def plot_spearman_by_dataset(
    results: Dict[str, float],
    model_name: str,
    strategy: str,
    output_dir: Path,
) -> None:
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
) -> None:
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
) -> None:
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
) -> None:
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
) -> None:
    """Compare Spearman correlations across embedding strategies (Exp 2-6).

    Groups by strategy with one bar per dataset, or one grouped bar per
    strategy. Enables direct comparison of which input formulation works best.

    TODO: implement in NB06.
    """
    raise NotImplementedError("Implement in NB06.")
