"""AbAgym data loading and processing functions.

Two responsibilities:
  1. Data pipeline (run once via NB01): extract sequences from PDB files,
     run ANARCI annotation, reconstruct mutant sequences, save CSVs.
  2. Data loading (used by all downstream notebooks): load the saved CSVs
     and provide index lookups for residue-level embedding extraction.

ANARCI requires HMMER installed via apt before pip install anarci.
Install order: apt-get install -y hmmer  ->  pip install anarci
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import CDR_REGIONS, FR_REGION

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_REGIONS = set(CDR_REGIONS) | {FR_REGION}
_EXPECTED_COLUMNS = 14
_EXPECTED_MUTATIONS = 5318
_EXPECTED_ANTIBODIES = 5

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# IMGT CDR position boundaries, same numeric ranges for H and L chains.
# The region name encodes both chain and loop (e.g. CDR_H1, CDR_L3).
IMGT_CDR_BOUNDS = {
    'H': [('CDR_H1', 27, 38), ('CDR_H2', 56, 65), ('CDR_H3', 105, 117)],
    'L': [('CDR_L1', 27, 38), ('CDR_L2', 56, 65), ('CDR_L3', 105, 117)],
}

# AbAgym processed PDB file stems, keyed by DMS dataset name.
# These are the base filenames inside the PDB_files.zip from the AbAgym repo.
PDB_FILE_STEMS = {
    'Ang2_2017_G6':          'G6_27_30A_corrected_4zfg',
    'EGFR_2013_Cetuximab':   'Cetuximab_1yy9',
    'lysozyme_2019_D441':    'D441_1mlc',
    'VEGF_2017b_G6':         'G6_27_30A_corrected_4zff',
    'HER2_2021_trastuzumab': 'trastuzumab_8pwh',
}

# ---------------------------------------------------------------------------
# CSV loading (used by all downstream notebooks)
# ---------------------------------------------------------------------------

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

    df['site'] = df['site'].astype(str)
    return df


def load_abagym_sequences(data_dir: Path) -> pd.DataFrame:
    """Load abagym_sequences.csv.

    Returns a DataFrame with 5 rows (one per antibody). The mapping_H and
    mapping_L columns are JSON strings; call json.loads() to parse them.

    Mapping dict structure:
        keys: PDB position label (string, e.g. '100A')
        values: {imgt_pos: int, imgt_ins: str, region: str, seq_idx: int}

    seq_idx is the 0-based index into the full PDB chain amino acid sequence.
    It accounts for any non-variable-domain residues before the VH/VL start.
    """
    path = Path(data_dir) / 'abagym_sequences.csv'
    df = pd.read_csv(path)

    assert len(df) == _EXPECTED_ANTIBODIES, (
        f"Expected {_EXPECTED_ANTIBODIES} rows, got {len(df)}"
    )
    return df


# ---------------------------------------------------------------------------
# Residue index lookup (used by embedding modules)
# ---------------------------------------------------------------------------

def get_mutation_site_index(
    sequences_df: pd.DataFrame,
    dms_name: str,
    chain: str,
    site: str,
) -> int:
    """Return the 0-based amino acid index for a single mutation site.

    Parameters
    ----------
    sequences_df : DataFrame from load_abagym_sequences().
    dms_name : e.g. 'Ang2_2017_G6'.
    chain : 'H' or 'L'.
    site : PDB position label as string, e.g. '100A' or '28'.

    Returns
    -------
    int: 0-based index into the amino acid sequence string. This is NOT the
    token position in a model's tokenized output -- model-specific offsets
    (BOS token in ESM-2, separator offset in AbLang2) are handled in the
    embedding modules.
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

    Returns
    -------
    np.ndarray of shape (5318,) with dtype int64.
    """
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


# ---------------------------------------------------------------------------
# Data pipeline functions (run once in NB01 to produce the CSVs)
# ---------------------------------------------------------------------------

def get_region_label(chain_id: str, imgt_pos: int) -> str:
    """Return CDR/FR region label for a given chain and IMGT position.

    Insertion codes within a CDR range (e.g. IMGT 111A in CDR H3) are still
    CDR -- only the integer part of the IMGT position is checked.

    Parameters
    ----------
    chain_id : 'H' or 'L'
    imgt_pos : integer IMGT position number

    Returns
    -------
    str: one of 'CDR_H1', 'CDR_H2', 'CDR_H3', 'CDR_L1', 'CDR_L2',
         'CDR_L3', or 'FR'
    """
    for cdr_name, start, end in IMGT_CDR_BOUNDS.get(chain_id, []):
        if start <= imgt_pos <= end:
            return cdr_name
    return FR_REGION


def extract_chain_residues(
    pdb_path: str,
    chain_id: str,
) -> List[Tuple[str, str]]:
    """Extract residues from a single PDB chain using CA atoms.

    Returns a list of (pos_label, aa) tuples in PDB order, where:
      pos_label = residue_number + insertion_code.strip()
                  e.g. '100', '100A', '100B'
      aa        = single-letter amino acid code

    The 0-based index of each tuple in the returned list is that residue's
    seq_idx. This is the value stored in the mapping JSON.

    Uses only AbAgym's processed PDB files (from PDB_files.zip in the
    AbAgym repo). Do not use raw RCSB downloads -- AbAgym renames chains
    to H and L.
    """
    residues = []
    seen = set()

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            if line[13:15].strip() != 'CA':
                continue
            if line[21] != chain_id:
                continue

            res_num  = line[22:26].strip()
            ins_code = line[26]
            res_name = line[17:20].strip()

            pos_label = f'{res_num}{ins_code.strip()}'
            key = (res_num, ins_code)

            if key in seen:
                continue
            seen.add(key)

            aa = THREE_TO_ONE.get(res_name, 'X')
            residues.append((pos_label, aa))

    return residues


def build_anarci_mapping(
    dms_name: str,
    chain_id: str,
    pdb_residues: List[Tuple[str, str]],
) -> Tuple[Dict, int, int]:
    """Run ANARCI on a chain sequence and build PDB->IMGT->CDR/FR mapping.

    Parameters
    ----------
    dms_name : dataset name, used as sequence identifier for ANARCI.
    chain_id : 'H' or 'L'.
    pdb_residues : list of (pos_label, aa) tuples from extract_chain_residues().

    Returns
    -------
    (mapping, seqstart, seqend) where:
      mapping : dict {pdb_pos_label: {imgt_pos, imgt_ins, region, seq_idx}}
        - imgt_pos: int IMGT position number
        - imgt_ins: str insertion code within IMGT numbering (often '')
        - region: CDR/FR label string
        - seq_idx: 0-based index into the full chain amino acid string
      seqstart : int, first variable-domain residue index in pdb_residues
      seqend : int, last variable-domain residue index in pdb_residues

    Notes
    -----
    Only variable domain residues appear in the mapping. Constant domain
    residues are outside the ANARCI hit range and are not included.

    Amino acid mismatches between PDB and ANARCI are printed as warnings.
    These can occur at modified residues and are non-fatal.

    Requires ANARCI (pip install anarci) and HMMER (apt-get install -y hmmer).
    HMMER must be installed before anarci.
    """
    import anarci

    sequence = ''.join(aa for _, aa in pdb_residues)
    seq_id = f'{dms_name}_{chain_id}'

    results = anarci.anarci([(seq_id, sequence)], scheme='imgt')

    if results[0][0] is None:
        raise ValueError(f'ANARCI found no immunoglobulin domain in {seq_id}')

    numbering_list, seqstart, seqend = results[0][0][0]

    non_gap = [
        ((imgt_pos, imgt_ins), aa)
        for (imgt_pos, imgt_ins), aa in numbering_list
        if aa != '-'
    ]

    expected_len = seqend - seqstart + 1
    if len(non_gap) != expected_len:
        raise ValueError(
            f'{seq_id}: non-gap count {len(non_gap)} != expected {expected_len} '
            f'(seqstart={seqstart}, seqend={seqend})'
        )

    mapping = {}
    mismatches = []

    for i, ((imgt_pos, imgt_ins), anarci_aa) in enumerate(non_gap):
        seq_idx = seqstart + i
        pdb_pos_label, pdb_aa = pdb_residues[seq_idx]

        if pdb_aa != anarci_aa:
            mismatches.append((pdb_pos_label, pdb_aa, anarci_aa))

        mapping[pdb_pos_label] = {
            'imgt_pos': imgt_pos,
            'imgt_ins': imgt_ins.strip(),
            'region':   get_region_label(chain_id, imgt_pos),
            'seq_idx':  seq_idx,
        }

    if mismatches:
        print(f'WARNING: {len(mismatches)} AA mismatches in {seq_id}:')
        for pos, pdb_aa, anarci_aa in mismatches[:5]:
            print(f'  PDB {pos}: PDB={pdb_aa}, ANARCI={anarci_aa}')

    return mapping, seqstart, seqend


def reconstruct_mutant_sequence(
    heavy_seq: str,
    light_seq: str,
    chain: str,
    site: str,
    wildtype_aa: str,
    mutant_aa: str,
    mapping: Dict,
) -> Tuple[str, str]:
    """Apply a single amino acid substitution to the correct chain sequence.

    Parameters
    ----------
    heavy_seq : wildtype heavy chain sequence string.
    light_seq : wildtype light chain sequence string.
    chain : 'H' or 'L' -- which chain carries the mutation.
    site : PDB position label as string, e.g. '100A' or '28'.
    wildtype_aa : expected amino acid at this position (single letter).
    mutant_aa : replacement amino acid (single letter).
    mapping : dict with structure mapping[chain][site] -> {seq_idx, ...}
              (the loaded mapping dict for a single antibody, keyed by chain)

    Returns
    -------
    (mutant_heavy_seq, mutant_light_seq): tuple of str.
    The unchanged chain is returned as-is.

    Raises
    ------
    ValueError if site not in mapping, wildtype AA mismatch, or length change.
    """
    site = str(site)

    if chain not in mapping:
        raise ValueError(f'Chain {chain} not in mapping')
    if site not in mapping[chain]:
        raise ValueError(f'Site {site} not in mapping[{chain}]')

    seq_idx = mapping[chain][site]['seq_idx']
    seq = heavy_seq if chain == 'H' else light_seq

    actual_wt = seq[seq_idx]
    if actual_wt != wildtype_aa:
        raise ValueError(
            f'Wildtype mismatch at chain {chain} site {site} seq_idx {seq_idx}: '
            f'expected {wildtype_aa}, found {actual_wt}'
        )

    mutant_seq = seq[:seq_idx] + mutant_aa + seq[seq_idx + 1:]

    if len(mutant_seq) != len(seq):
        raise ValueError(
            f'Length changed after substitution: {len(seq)} -> {len(mutant_seq)}'
        )

    diffs = sum(a != b for a, b in zip(seq, mutant_seq))
    if diffs != 1:
        raise ValueError(
            f'Expected exactly 1 difference, found {diffs}'
        )

    if chain == 'H':
        return mutant_seq, light_seq
    else:
        return heavy_seq, mutant_seq
