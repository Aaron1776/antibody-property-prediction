"""ESM-2 embedding extraction.

Token structure (w_extra_tkns=False equivalent for ESM-2):
    [BOS, res_1, res_2, ..., res_n, EOS, PAD...]

Residue i in the amino acid string maps to token position i + 1 (BOS offset).
BOS = alphabet.cls_idx, EOS = alphabet.eos_idx, PAD = alphabet.padding_idx.
All three are excluded from mean pooling.

ESM-2 encodes heavy and light chains in SEPARATE forward passes.
Sequence-level representation: concat(mean_pool(H), mean_pool(L)) = 2560-dim.
Residue-level representation: token at position seq_idx + 1 = 1280-dim.
"""

from typing import List, Tuple

import numpy as np
import torch

# TODO: verify exact import path for fair-esm in NB02 (model exploration)
# import esm


def load_esm2(device: str):
    """Load ESM-2 650M, move to device, set eval mode.

    Returns
    -------
    (model, alphabet, batch_converter)

    The repr layer is model.num_layers (= 33). Never hardcode this.

    TODO: verify load call in NB02 -- expected:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
    """
    # TODO: implement after verifying API in NB02
    raise NotImplementedError(
        "Verify ESM-2 load API in NB02 before implementing."
    )


def get_repr_layer(model) -> int:
    """Return the representation layer index.

    Always use model.num_layers. Never hardcode 33.
    """
    return model.num_layers


def get_token_offsets(alphabet) -> dict:
    """Return token offset information for residue index lookup.

    For ESM-2, residue i in the amino acid string maps to token position i + 1
    because of the leading BOS token. Returns a dict for clarity:
        {'bos_idx': alphabet.cls_idx, 'eos_idx': alphabet.eos_idx,
         'pad_idx': alphabet.padding_idx, 'residue_offset': 1}

    TODO: confirm cls_idx, eos_idx, padding_idx attribute names in NB02.
    """
    return {
        'bos_idx': alphabet.cls_idx,
        'eos_idx': alphabet.eos_idx,
        'pad_idx': alphabet.padding_idx,
        'residue_offset': 1,
    }


def embed_sequences_pooled(
    model,
    alphabet,
    batch_converter,
    sequences: List[Tuple[str, str]],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Embed a list of (heavy_seq, light_seq) tuples, return (N, 2560) tensor.

    Each pair is embedded with heavy and light chains in SEPARATE forward
    passes. Final representation is concat(mean_pool(H), mean_pool(L)).
    BOS, EOS, and PAD tokens are excluded from mean pooling.

    Parameters
    ----------
    sequences:
        List of (heavy_seq, light_seq) tuples.
    batch_size:
        Number of sequences per forward pass. Tune based on available VRAM.
    device:
        'cuda' or 'cpu'.

    Returns
    -------
    torch.Tensor of shape (N, 2560), dtype float32, on CPU.

    TODO: implement after verifying batch_converter input format and
    representations dict structure in NB02.
    """
    raise NotImplementedError(
        "Implement after NB02 model exploration confirms API details."
    )


def embed_sequences_residue(
    model,
    alphabet,
    batch_converter,
    sequences: List[Tuple[str, str]],
    site_indices: np.ndarray,
    chains: np.ndarray,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Extract per-residue embeddings at mutation sites. Returns (N, 1280).

    For each sequence/mutation pair, extracts the token embedding at the
    mutation site. Token position = seq_idx + 1 (BOS offset).

    Parameters
    ----------
    sequences:
        List of (heavy_seq, light_seq) tuples.
    site_indices:
        (N,) array of 0-based seq_idx values from abagym_sequences mapping.
    chains:
        (N,) array of 'H' or 'L' indicating which chain carries the mutation.
    batch_size:
        Number of sequences per forward pass.
    device:
        'cuda' or 'cpu'.

    Returns
    -------
    torch.Tensor of shape (N, 1280), dtype float32, on CPU.

    Note: for H chain mutations, embed the heavy sequence and extract token
    at seq_idx + 1. For L chain mutations, embed the light sequence and
    extract token at seq_idx + 1. The two chains are always separate passes.

    TODO: implement after NB02 confirms token indexing with a known mutation
    (e.g., first mutation in Ang2_2017_G6).
    """
    raise NotImplementedError(
        "Implement after NB02 model exploration confirms residue indexing."
    )


def profile_throughput(model, alphabet, batch_converter, device: str):
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
