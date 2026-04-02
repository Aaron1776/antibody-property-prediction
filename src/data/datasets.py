"""PyTorch Dataset classes for the experiment matrix.

Each dataset loads embedding tensors from Drive at init time and serves
them from memory. No model forward passes happen here.

Embedding strategies correspond to experiments 2-6 in the experiment matrix:
  DELTA_SEQUENCE          -- Exp 4: mean_pool(mutant) - mean_pool(wt)
  DELTA_RESIDUE           -- Exp 2: mutant_residue[mut_pos] - wt_residue[mut_pos]
  DELTA_RESIDUE_PLUS_WILD -- Exp 3: concat(delta_residue, mean_pool(wt_sequence))
  DELTA_RESIDUE_REDUCED   -- Exp 5: reduce(delta_residue) via PCA or learned projection
  DELTA_RESIDUE_POOLED    -- Exp 6: TBD (formulation not yet decided)
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd
import json


class EmbeddingStrategy(Enum):
    DELTA_SEQUENCE = 'delta_sequence'
    DELTA_RESIDUE = 'delta_residue'
    DELTA_RESIDUE_PLUS_WILD = 'delta_residue_plus_wild'
    DELTA_RESIDUE_REDUCED = 'delta_residue_reduced'
    DELTA_RESIDUE_POOLED = 'delta_residue_pooled'  # TBD


class AbAgymDataset(Dataset):
    """AbAgym mutation effect prediction dataset.

    Loads pre-cached embedding tensors and serves (input, label, metadata)
    triples. Metadata includes DMS_name and region for constraint loss
    computation during training.

    Parameters
    ----------
    antibody_df:
        DataFrame from src.data.abagym.load_abagym_antibody().
    embedding_dir:
        Path to Drive embeddings/ folder.
    strategy:
        EmbeddingStrategy determining which cached tensors to load and how
        to combine them.
    model_name:
        'esm2' or 'ablang2'. Determines which embedding files to load.
    label_col:
        Column in antibody_df to use as the regression label.
        Default: 'MinMax_normalized_DMS_score'.

    Returns from __getitem__
    ------------------------
    (input_tensor, label, metadata) where:
    - input_tensor: 1-D float tensor, shape depends on strategy and model
    - label: float scalar
    - metadata: dict with keys 'dms_name' (str) and 'region' (str)

    Input tensor shapes by strategy and model:
    - DELTA_SEQUENCE:           ESM-2 = 2560, AbLang2 = 960
    - DELTA_RESIDUE:            ESM-2 = 1280, AbLang2 = 480
    - DELTA_RESIDUE_PLUS_WILD:  ESM-2 = 1280 + 2560 = 3840, AbLang2 = 480 + 960 = 1440
    - DELTA_RESIDUE_REDUCED:    depends on reduction dimension
    - DELTA_RESIDUE_POOLED:     TBD
    """

    def __init__(
        self,
        antibody_df: pd.DataFrame,
        embedding_dir: Path,
        strategy: EmbeddingStrategy,
        model_name: str,
        label_col: str = 'MinMax_normalized_DMS_score',
    ):
        assert model_name in ('esm2', 'ablang2'), (
            f"model_name must be 'esm2' or 'ablang2', got {model_name!r}"
        )
        if strategy == EmbeddingStrategy.DELTA_RESIDUE_POOLED:
            raise NotImplementedError(
                "DELTA_RESIDUE_POOLED formulation is not yet decided. "
                "See Experiment 6 in PROGRESS.md."
            )

        self.df = antibody_df.reset_index(drop=True)
        self.strategy = strategy
        self.model_name = model_name
        self.label_col = label_col
        embedding_dir = Path(embedding_dir)

        # Load required tensors based on strategy
        if strategy == EmbeddingStrategy.DELTA_SEQUENCE:
            self.delta = self._load_tensor(
                embedding_dir, f'{model_name}_abagym_delta.pt'
            )
            self.delta_index = self._load_index(
                embedding_dir, f'{model_name}_abagym_delta_index.json'
            )

        elif strategy == EmbeddingStrategy.DELTA_RESIDUE:
            self.delta_residue = self._load_tensor(
                embedding_dir, f'{model_name}_abagym_residue_mutsite.pt'
            )
            self.wt_residue = self._load_tensor(
                embedding_dir, f'{model_name}_abagym_residue_wtsite.pt'
            )

        elif strategy == EmbeddingStrategy.DELTA_RESIDUE_PLUS_WILD:
            self.delta_residue = self._load_tensor(
                embedding_dir, f'{model_name}_abagym_residue_mutsite.pt'
            )
            self.wt_residue = self._load_tensor(
                embedding_dir, f'{model_name}_abagym_residue_wtsite.pt'
            )
            self.wt_sequence = self._load_tensor(
                embedding_dir, f'{model_name}_abagym_wildtype.pt'
            )
            self.wt_sequence_index = self._load_index(
                embedding_dir, f'{model_name}_abagym_wildtype_index.json'
            )

        elif strategy == EmbeddingStrategy.DELTA_RESIDUE_REDUCED:
            # TODO: load residue tensors; reduction applied externally or via
            # a projection layer. For now, load the raw residue tensors.
            self.delta_residue = self._load_tensor(
                embedding_dir, f'{model_name}_abagym_residue_mutsite.pt'
            )
            self.wt_residue = self._load_tensor(
                embedding_dir, f'{model_name}_abagym_residue_wtsite.pt'
            )

        self.labels = torch.tensor(
            self.df[label_col].values, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        row = self.df.iloc[idx]
        label = self.labels[idx]
        metadata = {'dms_name': row['DMS_name'], 'region': row['region']}

        if self.strategy == EmbeddingStrategy.DELTA_SEQUENCE:
            # Delta embeddings are pre-computed and stored in order matching df
            input_tensor = self.delta[idx]

        elif self.strategy == EmbeddingStrategy.DELTA_RESIDUE:
            input_tensor = self.delta_residue[idx] - self.wt_residue[idx]

        elif self.strategy == EmbeddingStrategy.DELTA_RESIDUE_PLUS_WILD:
            delta_res = self.delta_residue[idx] - self.wt_residue[idx]
            # Look up wildtype sequence embedding for this antibody
            wt_row = self.wt_sequence_index[row['DMS_name']]
            wt_seq_emb = self.wt_sequence[wt_row]
            input_tensor = torch.cat([delta_res, wt_seq_emb])

        elif self.strategy == EmbeddingStrategy.DELTA_RESIDUE_REDUCED:
            # Raw delta residue; reduction handled by model or upstream
            input_tensor = self.delta_residue[idx] - self.wt_residue[idx]

        return input_tensor, label, metadata

    @staticmethod
    def _load_tensor(embedding_dir: Path, filename: str) -> torch.Tensor:
        path = embedding_dir / filename
        assert path.exists(), f"Embedding file not found: {path}"
        return torch.load(path, map_location='cpu')

    @staticmethod
    def _load_index(embedding_dir: Path, filename: str) -> dict:
        path = embedding_dir / filename
        assert path.exists(), f"Index file not found: {path}"
        with open(path) as f:
            return json.load(f)


class SAbDabDataset(Dataset):
    """SAbDab binding affinity prediction dataset.

    Uses delta sequence embeddings only. There are no individual mutations
    in SAbDab -- the task is to predict pKd from a full antibody sequence.
    Delta sequence strategy does not apply here (no wildtype/mutant pairs).

    In practice, this dataset uses the sequence-level embeddings directly
    (not a delta), since SAbDab has unique antibodies, not mutation pairs.
    The 'delta sequence' naming in the embedding files refers to the
    concat(mean_pool(H), mean_pool(L)) representation.

    Parameters
    ----------
    sabdab_df:
        DataFrame from src.data.sabdab.load_sabdab().
    embedding_dir:
        Path to Drive embeddings/ folder.
    model_name:
        'esm2' or 'ablang2'.
    label_col:
        Column to use as label. Default: 'pKd'.

    Returns from __getitem__
    ------------------------
    (input_tensor, label, metadata) where:
    - input_tensor: 1-D float tensor (2560 for ESM-2, 960 for AbLang2)
    - label: float scalar (pKd value)
    - metadata: dict with key 'antibody_id' (str)
    """

    def __init__(
        self,
        sabdab_df: pd.DataFrame,
        embedding_dir: Path,
        model_name: str,
        label_col: str = 'pKd',
    ):
        assert model_name in ('esm2', 'ablang2'), (
            f"model_name must be 'esm2' or 'ablang2', got {model_name!r}"
        )

        self.df = sabdab_df.reset_index(drop=True)
        self.model_name = model_name
        self.label_col = label_col
        embedding_dir = Path(embedding_dir)

        self.embeddings = self._load_tensor(
            embedding_dir, f'{model_name}_sabdab.pt'
        )
        self.labels = torch.tensor(
            self.df[label_col].values, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        label = self.labels[idx]
        metadata = {'antibody_id': self.df.iloc[idx]['Antibody_ID']}
        return self.embeddings[idx], label, metadata

    @staticmethod
    def _load_tensor(embedding_dir: Path, filename: str) -> torch.Tensor:
        path = embedding_dir / filename
        assert path.exists(), f"Embedding file not found: {path}"
        return torch.load(path, map_location='cpu')
