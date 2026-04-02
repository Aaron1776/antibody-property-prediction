"""AbLang2 embedding extraction.

Token structure (w_extra_tkns=False):
    [H_1, H_2, ..., H_n, SEP(25), L_1, ..., L_m, PAD(21)...]

No BOS or EOS tokens. Both chains in a SINGLE forward pass.
Input format: 'VH|VL' pipe-separated string.

Special tokens:
    SEP = ablang.tokenizer.sep_token (= 25, read programmatically)
    PAD = ablang.AbRep.padding_tkn (= 21, read programmatically)
    Never hardcode these values.

Chain boundary detection (vectorized):
    sep_mask = (tokens == sep_id)
    sep_cumsum = sep_mask.cumsum(dim=1)
    H mask: (sep_cumsum == 0) & (tokens != pad_id)
    L mask: (sep_cumsum >= 1) & (tokens != sep_id) & (tokens != pad_id)

Per-residue hidden dim: 480 (NOT 768 -- empirically confirmed. The original
spec assumed 768. This was wrong. Always derive via get_hidden_dim().)

Residue token positions:
    Heavy chain mutation at seq_idx: token_pos = seq_idx
    Light chain mutation at seq_idx: token_pos = len(heavy_seq) + 1 + seq_idx
    (+1 accounts for the SEP token between chains)

Sequence-level representation: concat(mean_pool(H), mean_pool(L)) = 960-dim.
Residue-level representation: token at mutation position = 480-dim.
"""

from typing import List, Tuple

import numpy as np
import torch

# TODO: verify exact import and load call in NB02 (model exploration)
# import ablang2


def load_ablang2(device: str):
    """Load AbLang2-paired model.

    Returns the ablang model object. Device is passed as a string at load
    time. Standard .to(device) also works since AbRep is an nn.Module.

    TODO: verify load call in NB02 -- expected:
        ablang2.pretrained(
            model_to_use='ablang2-paired',
            random_init=False,
            ncpu=1,
            device=str(device)
        )
    """
    raise NotImplementedError(
        "Verify AbLang2 load API in NB02 before implementing."
    )


def get_special_tokens(ablang_model) -> dict:
    """Return special token ids, read programmatically from the model.

    Returns
    -------
    {'sep': ablang_model.tokenizer.sep_token, 'pad': ablang_model.AbRep.padding_tkn}

    Never hardcode these values. As of prior work they are SEP=25, PAD=21,
    but always read from the model object.

    TODO: confirm attribute paths in NB02.
    """
    return {
        'sep': ablang_model.tokenizer.sep_token,
        'pad': ablang_model.AbRep.padding_tkn,
    }


def get_hidden_dim(ablang_model, device: str) -> int:
    """Return the per-residue hidden dimension by running a dummy forward pass.

    Expected result: 480. This is how we discovered the true hidden dim --
    the original project spec assumed 768, which was wrong.

    Always derive the hidden dim at runtime rather than hardcoding it.

    TODO: implement after NB02 confirms the forward pass API. The dummy
    input should be a minimal valid sequence, e.g. one short heavy+light pair.
    """
    raise NotImplementedError(
        "Implement after NB02 confirms forward pass API. Expected output: 480."
    )


def get_chain_masks(
    tokens: torch.Tensor,
    sep_id: int,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized chain boundary detection.

    Parameters
    ----------
    tokens:
        (batch, seq_len) integer tensor of token ids.
    sep_id:
        Separator token id (read from ablang.tokenizer.sep_token).
    pad_id:
        Padding token id (read from ablang.AbRep.padding_tkn).

    Returns
    -------
    (heavy_mask, light_mask): boolean tensors of shape (batch, seq_len).
    heavy_mask is True for heavy chain residue positions (before SEP, not PAD).
    light_mask is True for light chain residue positions (after SEP, not SEP, not PAD).
    """
    sep_mask = (tokens == sep_id)
    sep_cumsum = sep_mask.cumsum(dim=1)

    heavy_mask = (sep_cumsum == 0) & (tokens != pad_id)
    light_mask = (sep_cumsum >= 1) & (~sep_mask) & (tokens != pad_id)

    return heavy_mask, light_mask


def embed_sequences_pooled(
    ablang_model,
    sequences: List[Tuple[str, str]],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Embed a list of (heavy_seq, light_seq) tuples. Returns (N, 960) tensor.

    Both chains are embedded in a single forward pass per sequence.
    Input format: 'VH|VL' pipe-separated string.
    Final representation: concat(mean_pool(H), mean_pool(L)).
    Chain boundaries determined by get_chain_masks().

    Parameters
    ----------
    sequences:
        List of (heavy_seq, light_seq) tuples.
    batch_size:
        Number of sequences per forward pass.
    device:
        'cuda' or 'cpu'.

    Returns
    -------
    torch.Tensor of shape (N, 960), dtype float32, on CPU.

    TODO: implement after NB02 confirms tokenizer call signature and
    last_hidden_states shape and type.
    """
    raise NotImplementedError(
        "Implement after NB02 model exploration confirms API details."
    )


def embed_sequences_residue(
    ablang_model,
    sequences: List[Tuple[str, str]],
    site_indices: np.ndarray,
    chains: np.ndarray,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Extract per-residue embeddings at mutation sites. Returns (N, 480).

    For each sequence/mutation pair, extracts the token embedding at the
    mutation site.

    Token positions:
        Heavy chain mutation at seq_idx: token_pos = seq_idx
        Light chain mutation at seq_idx: token_pos = len(heavy_seq) + 1 + seq_idx
        (+1 for SEP token between chains)

    Parameters
    ----------
    sequences:
        List of (heavy_seq, light_seq) tuples.
    site_indices:
        (N,) array of 0-based seq_idx values.
    chains:
        (N,) array of 'H' or 'L'.
    batch_size:
        Number of sequences per forward pass.
    device:
        'cuda' or 'cpu'.

    Returns
    -------
    torch.Tensor of shape (N, 480), dtype float32, on CPU.

    TODO: implement after NB02 confirms token indexing with a known mutation.
    """
    raise NotImplementedError(
        "Implement after NB02 model exploration confirms residue indexing."
    )


def profile_throughput(ablang_model, device: str):
    """Profile forward pass throughput at varying sequence lengths and batch sizes.

    Tests sequence lengths [50, 100, 150, 200, 250] at batch sizes [1, 8, 32, 64].
    Prints a table of seconds-per-sample and peak VRAM usage.

    Used in NB02 to characterize model speed before designing the embedding
    generation pipeline.

    TODO: implement after NB02 confirms the forward pass API.
    """
    raise NotImplementedError(
        "Implement in context of NB02 model exploration."
    )
