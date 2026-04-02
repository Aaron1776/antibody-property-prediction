"""ESM-2 embedding extraction.

Token structure:
    [BOS, res_1, res_2, ..., res_n, EOS, PAD...]

Residue i in the amino acid string maps to token position i + 1 (BOS offset).
BOS = alphabet.cls_idx, EOS = alphabet.eos_idx, PAD = alphabet.padding_idx.
All three are excluded from mean pooling.

ESM-2 embeds heavy and light chains in SEPARATE forward passes.
Sequence-level representation: concat(mean_pool(H), mean_pool(L)) = 2560-dim.
Residue-level representation: token at position seq_idx + 1 = 1280-dim.

Load:
    import esm
    model, alphabet = getattr(esm.pretrained, 'esm2_t33_650M_UR50D')()
    batch_converter = alphabet.get_batch_converter()
    REPR_LAYER = model.num_layers  # 33 for 650M -- never hardcode
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.utils import save_index, load_index


def load_esm2(device: str):
    """Load ESM-2 650M, move to device, set eval mode.

    Returns
    -------
    (model, alphabet, batch_converter)

    The repr layer is model.num_layers (= 33). Never hardcode this.
    """
    import esm as esm_lib

    model, alphabet = getattr(esm_lib.pretrained, 'esm2_t33_650M_UR50D')()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(device)
    return model, alphabet, batch_converter


def get_repr_layer(model) -> int:
    """Return the representation layer index. Always model.num_layers."""
    return model.num_layers


def get_token_offsets(alphabet) -> dict:
    """Return token offset information for residue index lookup.

    For ESM-2, residue i in the amino acid string maps to token position i + 1
    because of the leading BOS token.

    Returns
    -------
    dict with keys: bos_idx, eos_idx, pad_idx, residue_offset
    """
    return {
        'bos_idx':        alphabet.cls_idx,
        'eos_idx':        alphabet.eos_idx,
        'pad_idx':        alphabet.padding_idx,
        'residue_offset': 1,
    }


def embed_batch(
    batch: List[Tuple],
    model,
    alphabet,
    batch_converter,
    device: str,
    repr_layer: int,
) -> Dict:
    """Embed a single batch of (identifier, sequence) tuples.

    Parameters
    ----------
    batch : list of (identifier, amino_acid_sequence) tuples.
    model : loaded ESM-2 model in eval mode.
    alphabet : ESM-2 alphabet object.
    batch_converter : alphabet.get_batch_converter().
    device : 'cuda' or 'cpu'.
    repr_layer : transformer layer to extract from (use get_repr_layer(model)).

    Returns
    -------
    dict {identifier: tensor of shape (embedding_dim,)} on CPU.
    """
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    with torch.no_grad():
        output = model(tokens, repr_layers=[repr_layer], return_contacts=False)

    representations = output['representations'][repr_layer]

    mask = (
        (tokens != alphabet.cls_idx) &
        (tokens != alphabet.eos_idx) &
        (tokens != alphabet.padding_idx)
    )  # (batch, seq_len)

    mask_expanded  = mask.unsqueeze(-1).float()
    residue_sum    = (representations * mask_expanded).sum(dim=1)
    residue_counts = mask_expanded.sum(dim=1)
    pooled         = residue_sum / residue_counts  # (batch, dim)

    return {identifier: pooled[i].cpu()
            for i, (identifier, _) in enumerate(batch)}


def embed_sequences(
    id_seq_pairs: List[Tuple],
    model,
    alphabet,
    batch_converter,
    device: str,
    batch_size: int,
    repr_layer: int,
) -> Dict:
    """Embed all sequences without checkpointing. Use for small jobs only.

    Parameters
    ----------
    id_seq_pairs : list of (identifier, sequence) tuples.

    Returns
    -------
    dict {identifier: tensor of shape (embedding_dim,)} on CPU.
    """
    results = {}
    with tqdm(total=len(id_seq_pairs), unit='seq') as pbar:
        for start in range(0, len(id_seq_pairs), batch_size):
            batch = id_seq_pairs[start: start + batch_size]
            results.update(
                embed_batch(batch, model, alphabet, batch_converter, device, repr_layer)
            )
            pbar.update(len(batch))
    return results


def embed_with_checkpoint(
    id_seq_pairs: List[Tuple],
    model,
    alphabet,
    batch_converter,
    device: str,
    ckpt_emb_path,
    ckpt_idx_path,
    batch_size: int,
    repr_layer: int,
    checkpoint_every: int = 10,
) -> Dict:
    """Embed all sequences with checkpoint/resume support. Use for large jobs.

    On first run: processes all sequences, saving a checkpoint every
    checkpoint_every batches. On resume: loads checkpoint, skips already-
    processed sequences, and continues from where it left off.

    Parameters
    ----------
    id_seq_pairs : list of (identifier, sequence) tuples.
    ckpt_emb_path : path to save/load checkpoint tensor.
    ckpt_idx_path : path to save/load checkpoint index JSON.
    checkpoint_every : save checkpoint after this many batches.

    Returns
    -------
    dict {identifier: tensor of shape (embedding_dim,)} on CPU.
    """
    ckpt_emb_path = Path(ckpt_emb_path)
    ckpt_idx_path = Path(ckpt_idx_path)

    if ckpt_emb_path.exists() and ckpt_idx_path.exists():
        ckpt_tensor = torch.load(ckpt_emb_path, map_location='cpu')
        ckpt_index  = load_index(ckpt_idx_path)
        results  = {int(id_): ckpt_tensor[pos]
                    for pos, id_ in ckpt_index.items()}
        done_ids = set(results.keys())
        print(f"Checkpoint found: {len(done_ids)} sequences already processed.")
    else:
        results  = {}
        done_ids = set()
        print("No checkpoint found. Starting from row 0.")

    remaining = [(id_, seq) for id_, seq in id_seq_pairs if id_ not in done_ids]
    print(f"Sequences remaining: {len(remaining)}")

    if not remaining:
        return results

    batch_count = 0
    with tqdm(total=len(id_seq_pairs), initial=len(done_ids), unit='seq') as pbar:
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start: start + batch_size]
            results.update(
                embed_batch(batch, model, alphabet, batch_converter, device, repr_layer)
            )
            batch_count += 1
            pbar.update(len(batch))

            if batch_count % checkpoint_every == 0:
                sorted_ids  = sorted(results.keys())
                ckpt_t      = torch.stack([results[k] for k in sorted_ids])
                ckpt_idx    = {i: id_ for i, id_ in enumerate(sorted_ids)}
                torch.save(ckpt_t, ckpt_emb_path)
                save_index(ckpt_idx, ckpt_idx_path)

    return results


def embed_sequences_pooled(
    model,
    alphabet,
    batch_converter,
    sequences: List[Tuple[str, str]],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Embed a list of (heavy_seq, light_seq) tuples. Returns (N, 2560) tensor.

    Heavy and light chains are embedded in SEPARATE forward passes, then
    concatenated: concat(mean_pool(H), mean_pool(L)).

    Parameters
    ----------
    sequences : list of (heavy_seq, light_seq) tuples.
    batch_size : sequences per forward pass (per chain, not per pair).
    device : 'cuda' or 'cpu'.

    Returns
    -------
    torch.Tensor of shape (N, 2560), dtype float32, on CPU.
    """
    repr_layer = get_repr_layer(model)

    heavy_pairs = [(i, h) for i, (h, _) in enumerate(sequences)]
    light_pairs = [(i, l) for i, (_, l) in enumerate(sequences)]

    heavy_pairs_sorted = sorted(heavy_pairs, key=lambda x: len(x[1]))
    light_pairs_sorted = sorted(light_pairs, key=lambda x: len(x[1]))

    heavy_results = embed_sequences(
        heavy_pairs_sorted, model, alphabet, batch_converter,
        device, batch_size, repr_layer
    )
    light_results = embed_sequences(
        light_pairs_sorted, model, alphabet, batch_converter,
        device, batch_size, repr_layer
    )

    embeddings = []
    for i in range(len(sequences)):
        combined = torch.cat([heavy_results[i], light_results[i]], dim=0)
        embeddings.append(combined.unsqueeze(0))

    return torch.cat(embeddings, dim=0)


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
    sequences : list of (heavy_seq, light_seq) tuples, one per mutation.
    site_indices : (N,) array of 0-based seq_idx values from the mapping CSV.
    chains : (N,) array of 'H' or 'L'.
    batch_size : sequences per forward pass.
    device : 'cuda' or 'cpu'.

    Returns
    -------
    torch.Tensor of shape (N, 1280), dtype float32, on CPU.

    Notes
    -----
    For H chain mutations: embed the heavy sequence, extract token at
    seq_idx + 1. For L chain mutations: embed the light sequence, extract
    token at seq_idx + 1. The +1 is the BOS token offset.

    Token at seq_idx + 1 should correspond to the amino acid at seq_idx
    in the input string. Verify with a known mutation in NB02 before relying
    on this for training.
    """
    repr_layer = get_repr_layer(model)
    n = len(sequences)
    assert len(site_indices) == n and len(chains) == n

    all_embeddings = []

    with tqdm(total=n, unit='seq') as pbar:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seqs    = sequences[start:end]
            batch_indices = site_indices[start:end]
            batch_chains  = chains[start:end]

            # Build (identifier, sequence) pairs -- embed the mutated chain only
            pairs = []
            for j, ((h_seq, l_seq), chain) in enumerate(
                    zip(batch_seqs, batch_chains)):
                seq = h_seq if chain == 'H' else l_seq
                pairs.append((j, seq))

            _, _, tokens = batch_converter(pairs)
            tokens = tokens.to(device)

            with torch.no_grad():
                output = model(tokens, repr_layers=[repr_layer],
                               return_contacts=False)

            representations = output['representations'][repr_layer]
            # (batch, seq_len, dim)

            for j, seq_idx in enumerate(batch_indices):
                token_pos = int(seq_idx) + 1  # +1 for BOS token
                emb = representations[j, token_pos, :].cpu()
                all_embeddings.append(emb.unsqueeze(0))

            pbar.update(len(batch_seqs))

    return torch.cat(all_embeddings, dim=0)


def profile_throughput(model, alphabet, batch_converter, device: str):
    """Profile forward pass throughput at varying lengths and batch sizes.

    Tests sequence lengths [50, 100, 150, 200, 250] at batch sizes
    [1, 8, 32, 64]. Prints seconds-per-sample and peak VRAM.

    Used in NB02 to characterize model speed before designing the generation
    pipeline.
    """
    import time

    repr_layer = get_repr_layer(model)
    lengths    = [50, 100, 150, 200, 250]
    batch_sizes = [1, 8, 32, 64]
    aa = 'A'

    print(f'{"length":>8} {"batch":>6} {"sec/sample":>12} {"peak VRAM GB":>14}')
    print('-' * 46)

    for length in lengths:
        for bs in batch_sizes:
            seq = aa * length
            batch = [(f'test_{j}', seq) for j in range(bs)]

            if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
                torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            embed_batch(batch, model, alphabet, batch_converter, device, repr_layer)
            elapsed = time.time() - t0

            sec_per_sample = elapsed / bs

            if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
                peak_vram = torch.cuda.max_memory_allocated() / 1e9
            else:
                peak_vram = float('nan')

            print(f'{length:>8} {bs:>6} {sec_per_sample:>12.4f} {peak_vram:>14.2f}')
