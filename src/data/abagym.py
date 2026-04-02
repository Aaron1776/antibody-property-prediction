import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import CDR_REGIONS, FR_REGION

_VALID_REGIONS = set(CDR_REGIONS) | {FR_REGION}
_EXPECTED_COLUMNS = 14
_EXPECTED_MUTATIONS = 5318
_EXPECTED_ANTIBODIES = 5


def load_abagym_antibody(data_dir: Path) -> pd.DataFrame:
    """Load abagym_antibody.csv.

    Returns a DataFrame with 5,318 rows and 14 columns. Asserts row count,
    column count, and that the region column contains only valid CDR/FR labels.

    Columns: DMS_name, PDB_file, chains, site, wildtype, mutation, mut_names,
    DMS_score, MinMax_normalized_DMS_score, Rank_quartile_normalized_DMS_score,
    closest_interface_atom_distance, mutant_heavy_seq, mutant_light_seq, region.

    Note: the 'site' column is a string (e.g. '100A'). Never cast to int.
    """
    path = Path(data_dir) / 'abagym_antibody.csv'
    df = pd.read_csv(path)

    assert len(df) == _EXPECTED_MUTATIONS, (
        f"Expected {_EXPECTED_MUTATIONS} rows, got {len(df)}"
    )
    assert df.shape[1] == _EXPECTED_COLUMNS, (
        f"Expected {_EXPECTED_COLUMNS} columns, got {df.shape[1]}"
    )

    invalid = set(df['region'].unique()) - _VALID_REGIONS
    assert not invalid, f"Unexpected region values: {invalid}"

    # Ensure site is treated as string, never coerced to numeric
    df['site'] = df['site'].astype(str)

    return df


def load_abagym_sequences(data_dir: Path) -> pd.DataFrame:
    """Load abagym_sequences.csv.

    Returns a DataFrame with 5 rows (one per antibody). The mapping_H and
    mapping_L columns are JSON strings; call json.loads() to parse them.

    Mapping dict structure:
        keys: PDB position label (string, e.g. '100A')
        values: {imgt_pos: int, imgt_ins: str, region: str, seq_idx: int}

    seq_idx is the 0-based index into the amino acid sequence string.
    """
    path = Path(data_dir) / 'abagym_sequences.csv'
    df = pd.read_csv(path)

    assert len(df) == _EXPECTED_ANTIBODIES, (
        f"Expected {_EXPECTED_ANTIBODIES} rows, got {len(df)}"
    )

    return df


def get_mutation_site_index(
    sequences_df: pd.DataFrame,
    dms_name: str,
    chain: str,
    site: str,
) -> int:
    """Return the 0-based amino acid sequence index for a single mutation site.

    Parameters
    ----------
    sequences_df:
        DataFrame from load_abagym_sequences().
    dms_name:
        Dataset name, e.g. 'Ang2_2017_G6'.
    chain:
        'H' for heavy chain mutation, 'L' for light chain mutation.
    site:
        PDB position label as string, e.g. '100A' or '28'.

    Returns
    -------
    int
        0-based index into the amino acid sequence string. This is NOT the
        token position in a model's tokenized output -- model-specific offsets
        (e.g. BOS token in ESM-2) are handled in the embedding modules.
    """
    row = sequences_df[sequences_df['dms_name'] == dms_name]
    assert len(row) == 1, f"dms_name {dms_name!r} not found in sequences_df"
    row = row.iloc[0]

    mapping_key = 'mapping_H' if chain == 'H' else 'mapping_L'
    mapping = json.loads(row[mapping_key])

    assert site in mapping, (
        f"Site {site!r} not found in {mapping_key} for {dms_name}"
    )

    return mapping[site]['seq_idx']


def get_all_mutation_site_indices(
    antibody_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
) -> np.ndarray:
    """Return a (5318,) array of 0-based seq_idx values for all mutations.

    This is the primary function for residue-level embedding extraction.
    For each row in antibody_df, looks up the seq_idx for the mutation site
    from the appropriate chain mapping in sequences_df.

    Parameters
    ----------
    antibody_df:
        DataFrame from load_abagym_antibody().
    sequences_df:
        DataFrame from load_abagym_sequences().

    Returns
    -------
    np.ndarray of shape (5318,) with dtype int64.
    """
    # Build lookup: {dms_name: {'H': mapping_dict, 'L': mapping_dict}}
    mappings = {}
    for _, row in sequences_df.iterrows():
        mappings[row['dms_name']] = {
            'H': json.loads(row['mapping_H']),
            'L': json.loads(row['mapping_L']),
        }

    indices = np.empty(len(antibody_df), dtype=np.int64)
    for i, row in enumerate(antibody_df.itertuples(index=False)):
        chain_map = mappings[row.DMS_name][row.chains]
        site = str(row.site)
        assert site in chain_map, (
            f"Site {site!r} not found for {row.DMS_name} chain {row.chains}"
        )
        indices[i] = chain_map[site]['seq_idx']

    return indices
