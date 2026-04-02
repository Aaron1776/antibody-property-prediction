"""AbLang2 embedding extraction.

Token structure (w_extra_tkns=False):
    [H_1, H_2, ..., H_n, SEP(25), L_1, ..., L_m, PAD(21)...]

No BOS or EOS tokens. Both chains processed in a SINGLE forward pass.
Input format: 'VH|VL' pipe-separated string.

Special tokens (read programmatically, never hardcode):
    SEP = ablang.tokenizer.sep_token       (= 25)
    PAD = ablang.AbRep.padding_tkn         (= 21)

Chain boundary: separator is at position len(heavy_seq) exactly.
Chain boundary detection is vectorized across the batch via cumulative sum:
    sep_mask   = (tokens == sep_id)
    sep_cumsum = sep_mask.cumsum(dim=1)
    H mask: (sep_cumsum == 0) & ~pad_mask
    L mask: (sep_cumsum >= 1) & ~sep_mask & ~pad_mask

Per-residue hidden dim: 480. Confirmed empirically from last_hidden_states
shape. The original project spec assumed 768 -- that was wrong.
Always derive EMBEDDING_DIM at runtime via get_hidden_dim().

Residue token positions:
    Heavy chain mutation at seq_idx: token_pos = seq_idx       (no offset)
    Light chain mutation at seq_idx: token_pos = len(heavy) + 1 + seq_idx
                                                (+1 for SEP token)

Sequence-level representation: concat(mean_pool(H), mean_pool(L)) = 960-dim.
Residue-level representation: token at mutation position = 480-dim.

Load:
    import ablang2
    ablang = ablang2.pretrained(
        model_to_use='ablang2-paired',
        random_init=False,
        ncpu=1,
        device=str(DEVICE)
    )
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.utils import save_index, load_index


def load_ablang2(device: str):
    """Load AbLang2-paired model.

    Device is passed as a string at load time. Standard .to(device) also
    works since AbRep is an nn.Module, but the API accepts device at init.

    Returns
    -------
    ablang model object with attributes: tokenizer, AbRep.
    """
    import ablang2 as ablang2_lib

    ablang = ablang2_lib.pretrained(
        model_to_use='ablang2-paired',
        random_init=False,
        ncpu=1,
        device=str(device),
    )
    return ablang


def get_special_tokens(ablang_model) -> dict:
    """Return special token ids, read programmatically from the model.

    Returns
    -------
    dict with keys:
        'sep': ablang_model.tokenizer.sep_token   (= 25)
        'pad': ablang_model.AbRep.padding_tkn     (= 21)

    Never hardcode these values. As of prior work they are SEP=25, PAD=21,
    but always read from the model object.
    """
    return {
        'sep': ablang_model.tokenizer.sep_token,
        'pad': ablang_model.AbRep.padding_tkn,
    }


def get_hidden_dim(ablang_model, device: str) -> int:
    """Return the per-residue hidden dimension from a dummy forward pass.

    Expected result: 480. The original project spec assumed 768 -- that
    was wrong. Always derive at runtime.

    Returns
    -------
    int: last_hidden_states.shape[-1]
    """
    dummy_seq = 'EVQLVESGGGLVQ|DIQMTQSPSSLSA'
    tokens = ablang_model.tokenizer(
        [dummy_seq], pad=True, w_extra_tkns=False, device=str(device)
    )
    with torch.no_grad():
        out = ablang_model.AbRep(tokens).last_hidden_states
    return out.shape[-1]


def get_chain_masks(
    tokens: torch.Tensor,
    sep_id: int,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized chain boundary detection for a batch of tokenized sequences.

    Parameters
    ----------
    tokens : (batch, seq_len) integer tensor of token ids.
    sep_id : separator token id (from ablang.tokenizer.sep_token).
    pad_id : padding token id (from ablang.AbRep.padding_tkn).

    Returns
    -------
    (heavy_mask, light_mask): boolean tensors of shape (batch, seq_len).
    heavy_mask: True for heavy chain residue positions (before SEP, not PAD).
    light_mask: True for light chain residue positions (after SEP, not SEP,
                not PAD).
    """
    sep_mask   = (tokens == sep_id)
    pad_mask   = (tokens == pad_id)
    sep_cumsum = sep_mask.cumsum(dim=1)

    heavy_mask = (sep_cumsum == 0) & ~pad_mask
    light_mask = (sep_cumsum >= 1) & ~sep_mask & ~pad_mask

    return heavy_mask, light_mask


def embed_batch(batch: List[Tuple], ablang_model, device: str) -> Dict:
    """Embed a single batch of (identifier, (heavy_seq, light_seq)) tuples.

    Parameters
    ----------
    batch : list of (identifier, (heavy_seq, light_seq)) tuples.
    ablang_model : loaded ablang2 model object.
    device : 'cuda' or 'cpu'.

    Returns
    -------
    dict {identifier: tensor of shape (960,)} on CPU.
    """
    sep_id = ablang_model.tokenizer.sep_token
    pad_id = ablang_model.AbRep.padding_tkn

    identifiers = [item[0] for item in batch]
    seqs        = [f"{item[1][0]}|{item[1][1]}" for item in batch]

    tokens = ablang_model.tokenizer(
        seqs, pad=True, w_extra_tkns=False, device=str(device)
    )
    # tokens shape: (batch, seq_len)

    with torch.no_grad():
        representations = ablang_model.AbRep(tokens).last_hidden_states
    # representations shape: (batch, seq_len, embedding_dim)

    heavy_mask, light_mask = get_chain_masks(tokens, sep_id, pad_id)

    # Sanity: every row must have exactly one separator
    sep_counts = (tokens == sep_id).sum(dim=1)
    assert (sep_counts == 1).all(), (
        f"Expected one separator per row, got: {sep_counts.tolist()}"
    )

    h_mask_exp = heavy_mask.unsqueeze(-1).float()
    l_mask_exp = light_mask.unsqueeze(-1).float()

    h_sum    = (representations * h_mask_exp).sum(dim=1)
    h_count  = h_mask_exp.sum(dim=1)
    h_pooled = h_sum / h_count   # (batch, dim)

    l_sum    = (representations * l_mask_exp).sum(dim=1)
    l_count  = l_mask_exp.sum(dim=1)
    l_pooled = l_sum / l_count   # (batch, dim)

    combined = torch.cat([h_pooled, l_pooled], dim=1)  # (batch, 2*dim)

    return {identifier: combined[i].cpu()
            for i, identifier in enumerate(identifiers)}


def embed_sequences(
    id_seq_pairs: List[Tuple],
    ablang_model,
    device: str,
    batch_size: int,
) -> Dict:
    """Embed all sequences without checkpointing. Use for small jobs only.

    Parameters
    ----------
    id_seq_pairs : list of (identifier, (heavy_seq, light_seq)) tuples.

    Returns
    -------
    dict {identifier: tensor of shape (960,)} on CPU.
    """
    results = {}
    for start in range(0, len(id_seq_pairs), batch_size):
        batch = id_seq_pairs[start: start + batch_size]
        results.update(embed_batch(batch, ablang_model, device))
    return results


def embed_with_checkpoint(
    id_seq_pairs: List[Tuple],
    ablang_model,
    device: str,
    ckpt_emb_path,
    ckpt_idx_path,
    batch_size: int,
    checkpoint_every: int = 10,
) -> Dict:
    """Embed all sequences with checkpoint/resume support. Use for large jobs.

    Parameters
    ----------
    id_seq_pairs : list of (identifier, (heavy_seq, light_seq)) tuples.
    ckpt_emb_path : path to save/load checkpoint tensor.
    ckpt_idx_path : path to save/load checkpoint index JSON.
    checkpoint_every : save checkpoint after this many batches.

    Returns
    -------
    dict {identifier: tensor of shape (960,)} on CPU.
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
            results.update(embed_batch(batch, ablang_model, device))
            batch_count += 1
            pbar.update(len(batch))

            if batch_count % checkpoint_every == 0:
                sorted_ids = sorted(results.keys())
                ckpt_t     = torch.stack([results[k] for k in sorted_ids])
                ckpt_idx   = {i: id_ for i, id_ in enumerate(sorted_ids)}
                torch.save(ckpt_t, ckpt_emb_path)
                save_index(ckpt_idx, ckpt_idx_path)

    return results


def embed_sequences_pooled(
    ablang_model,
    sequences: List[Tuple[str, str]],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Embed a list of (heavy_seq, light_seq) tuples. Returns (N, 960) tensor.

    Both chains are processed in a single forward pass per sequence.
    Final representation: concat(mean_pool(H), mean_pool(L)).

    Parameters
    ----------
    sequences : list of (heavy_seq, light_seq) tuples.
    batch_size : sequences per forward pass.
    device : 'cuda' or 'cpu'.

    Returns
    -------
    torch.Tensor of shape (N, 960), dtype float32, on CPU.
    """
    pairs = [(i, (h, l)) for i, (h, l) in enumerate(sequences)]
    results = embed_sequences(pairs, ablang_model, device, batch_size)

    return torch.stack([results[i] for i in range(len(sequences))])


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
        (+1 for the SEP token between chains)

    Parameters
    ----------
    sequences : list of (heavy_seq, light_seq) tuples, one per mutation.
    site_indices : (N,) array of 0-based seq_idx values.
    chains : (N,) array of 'H' or 'L'.
    batch_size : sequences per forward pass.
    device : 'cuda' or 'cpu'.

    Returns
    -------
    torch.Tensor of shape (N, 480), dtype float32, on CPU.

    Notes
    -----
    Both chains are tokenized together as 'VH|VL'. The token at seq_idx
    for heavy or len(heavy)+1+seq_idx for light is extracted directly from
    the joint representation. Verify with a known mutation in NB02 before
    relying on this for training.
    """
    sep_id = ablang_model.tokenizer.sep_token
    pad_id = ablang_model.AbRep.padding_tkn

    n = len(sequences)
    assert len(site_indices) == n and len(chains) == n

    all_embeddings = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_seqs    = sequences[start:end]
        batch_indices = site_indices[start:end]
        batch_chains  = chains[start:end]

        seqs = [f"{h}|{l}" for h, l in batch_seqs]
        tokens = ablang_model.tokenizer(
            seqs, pad=True, w_extra_tkns=False, device=str(device)
        )

        with torch.no_grad():
            representations = ablang_model.AbRep(tokens).last_hidden_states
        # (batch, seq_len, dim)

        for j, (seq_idx, chain) in enumerate(zip(batch_indices, batch_chains)):
            h_seq, _ = batch_seqs[j]
            if chain == 'H':
                token_pos = int(seq_idx)
            else:
                token_pos = len(h_seq) + 1 + int(seq_idx)  # +1 for SEP

            emb = representations[j, token_pos, :].cpu()
            all_embeddings.append(emb.unsqueeze(0))

    return torch.cat(all_embeddings, dim=0)


def profile_throughput(ablang_model, device: str):
    """Profile forward pass throughput at varying lengths and batch sizes.

    Tests sequence lengths [50, 100, 150, 200, 250] at batch sizes
    [1, 8, 32, 64]. Prints seconds-per-sample and peak VRAM.
    """
    import time

    lengths     = [50, 100, 150, 200, 250]
    batch_sizes = [1, 8, 32, 64]
    aa = 'A'

    print(f'{"length":>8} {"batch":>6} {"sec/sample":>12} {"peak VRAM GB":>14}')
    print('-' * 46)

    for length in lengths:
        for bs in batch_sizes:
            h_seq = aa * length
            l_seq = aa * length
            batch = [(j, (h_seq, l_seq)) for j in range(bs)]

            if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
                torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            embed_batch(batch, ablang_model, device)
            elapsed = time.time() - t0

            sec_per_sample = elapsed / bs

            if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
                peak_vram = torch.cuda.max_memory_allocated() / 1e9
            else:
                peak_vram = float('nan')

            print(f'{length:>8} {bs:>6} {sec_per_sample:>12.4f} {peak_vram:>14.2f}')
