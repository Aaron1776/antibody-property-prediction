"""Delta embedding computation.

Works on pre-cached tensors. No model forward passes happen here.

Sequence-level delta:
    delta[i] = mutant_tensor[i] - wildtype_tensor[wt_row]
    where wt_row is the row in the wildtype tensor corresponding to the
    antibody that mutation i belongs to.

Residue-level delta:
    delta[i] = mutsite_tensor[i] - wtsite_tensor[i]
    Both inputs are (N, dim) tensors already aligned by mutation index.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch


def compute_delta_sequence(
    mutant_tensor: torch.Tensor,
    wildtype_tensor: torch.Tensor,
    dms_names: List[str],
    wt_index: Dict[str, int],
) -> torch.Tensor:
    """Compute sequence-level delta embeddings.

    Subtracts the wildtype embedding for the correct antibody from each
    mutant embedding.

    Parameters
    ----------
    mutant_tensor:
        (N, dim) tensor of mutant sequence embeddings.
    wildtype_tensor:
        (n_antibodies, dim) tensor of wildtype sequence embeddings.
    dms_names:
        List of length N. Each entry is the DMS dataset name for that mutation,
        used to look up the corresponding wildtype row.
    wt_index:
        Dict mapping DMS dataset name -> row index in wildtype_tensor.
        Loaded from e.g. esm2_abagym_wildtype_index.json.

    Returns
    -------
    torch.Tensor of shape (N, dim), dtype float32.

    Raises
    ------
    AssertionError if any dms_name in dms_names is not found in wt_index.
    """
    # Build inverse mapping: name -> row
    dms_to_wt_row = {name: row for row, name in wt_index.items()}

    missing = [n for n in set(dms_names) if n not in dms_to_wt_row]
    assert not missing, f"DMS names not found in wildtype index: {missing}"

    wt_rows = [dms_to_wt_row[name] for name in dms_names]
    delta = mutant_tensor - wildtype_tensor[wt_rows]

    return delta


def compute_delta_residue(
    mutsite_tensor: torch.Tensor,
    wtsite_tensor: torch.Tensor,
) -> torch.Tensor:
    """Compute residue-level delta embeddings.

    Simple element-wise subtraction. Both inputs must already be aligned
    by mutation index (row i of mutsite corresponds to row i of wtsite).

    Parameters
    ----------
    mutsite_tensor:
        (N, dim) tensor of mutant residue embeddings at each mutation site.
    wtsite_tensor:
        (N, dim) tensor of wildtype residue embeddings at each mutation site.

    Returns
    -------
    torch.Tensor of shape (N, dim), dtype float32.
    """
    assert mutsite_tensor.shape == wtsite_tensor.shape, (
        f"Shape mismatch: {mutsite_tensor.shape} vs {wtsite_tensor.shape}"
    )
    return mutsite_tensor - wtsite_tensor


def load_cached_deltas(
    embedding_dir: Path,
    model_name: str,
    level: str,
) -> torch.Tensor:
    """Load a pre-cached delta tensor from Drive.

    Parameters
    ----------
    embedding_dir:
        Path to Drive embeddings/ folder.
    model_name:
        'esm2' or 'ablang2'.
    level:
        'sequence' or 'residue'.

    Returns
    -------
    torch.Tensor with shapes:
        sequence-level: ESM-2 (5318, 2560), AbLang2 (5318, 960)
        residue-level: ESM-2 (5318, 1280), AbLang2 (5318, 480)

    Raises
    ------
    AssertionError if the file does not exist or shape is unexpected.
    """
    assert model_name in ('esm2', 'ablang2'), (
        f"model_name must be 'esm2' or 'ablang2', got {model_name!r}"
    )
    assert level in ('sequence', 'residue'), (
        f"level must be 'sequence' or 'residue', got {level!r}"
    )

    if level == 'sequence':
        filename = f'{model_name}_abagym_delta.pt'
    else:
        # residue-level delta is computed from mutsite and wtsite tensors,
        # not stored as a single pre-cached file
        raise ValueError(
            "Residue-level deltas are computed from mutsite/wtsite tensors "
            "via compute_delta_residue(). Load mutsite and wtsite separately "
            "using their filenames: "
            f"{model_name}_abagym_residue_mutsite.pt and "
            f"{model_name}_abagym_residue_wtsite.pt"
        )

    path = Path(embedding_dir) / filename
    assert path.exists(), f"Delta tensor not found: {path}"

    tensor = torch.load(path, map_location='cpu')

    # Validate expected shapes
    expected_n = 5318
    expected_dim = {'esm2': 2560, 'ablang2': 960}[model_name]
    assert tensor.shape == (expected_n, expected_dim), (
        f"Unexpected shape {tensor.shape}, expected ({expected_n}, {expected_dim})"
    )

    return tensor
